use std::borrow::Cow;

use prover::common_constants;
use prover::risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;
use riscv_transpiler::ir::preprocess_bytecode;
use riscv_transpiler::ir::FullUnsignedMachineDecoderConfig;
use riscv_transpiler::ir::Instruction;
use riscv_transpiler::vm::DelegationsAndFamiliesCounters;
use riscv_transpiler::vm::RamWithRomRegion;
use riscv_transpiler::vm::SimpleSnapshotter;
use riscv_transpiler::vm::SimpleTape;
use riscv_transpiler::vm::State;
use riscv_transpiler::vm::VM;

use crate::rv32im::GuestResult;
use crate::rv32im::DEFAULT_CYCLES;

/// Either returns `None` if the string matches the one emitted by the panic raised if the
/// simulator encounters an exception or aborts the whole process.
fn check_panic_string<const ABORT: bool>(s: impl AsRef<str>) -> Option<GuestResult> {
    let s = s.as_ref();
    log::debug!("Panic string: {s:?}");
    if s.starts_with("Illegal instruction encounteted at PC =")
        || s.starts_with("Unaligned memory access at PC =")
    {
        return None;
    }

    if ABORT {
        log::error!("Target raised unhandled error: {s}");
        std::process::abort()
    } else {
        panic!("Target raised unhandled error: {s}")
    }
}

struct Binary<'d> {
    data: Cow<'d, [u8]>,
    text: Option<Cow<'d, [u8]>>,
}

pub(super) type DecoderConfig = FullUnsignedMachineDecoderConfig;
pub(super) type CountersT = DelegationsAndFamiliesCounters;

fn align<'d>(data: &'d [u8]) -> Cow<'d, [u8]> {
    let mult = data.len().next_multiple_of(4);

    if mult != data.len() {
        // Pad the data with 0 to keep alignment.
        let mut vec = Vec::with_capacity(mult);
        vec.extend_from_slice(data);
        vec.extend(std::iter::repeat_n(0u8, mult - data.len()));
        assert_eq!(vec.len() % 4, 0);
        Cow::Owned(vec)
    } else {
        Cow::Borrowed(data)
    }
}

fn assert_text_is_beginning_of_data(data: &[u8], text: &[u8]) {
    assert!(data.len() >= text.len());
    assert_eq!(&data[0..text.len()], text);
}

fn into_chunks(data: &[u8]) -> Vec<u32> {
    let (chunks, tail) = data.as_chunks::<4>();
    assert_eq!(tail.len(), 0);
    chunks.iter().copied().map(u32::from_le_bytes).collect()
}

impl<'d> Binary<'d> {
    fn new(data: &'d [u8], text: Option<&'d [u8]>) -> Self {
        let data = align(data);
        let text = text.map(align);
        Self { data, text }
    }

    fn data(&self) -> &[u8] {
        self.data.as_ref()
    }

    fn text(&self) -> Option<&[u8]> {
        self.text.as_deref()
    }

    fn data_chunks(&self) -> Vec<u32> {
        into_chunks(self.data())
    }

    fn text_chunks(&self) -> Option<Vec<u32>> {
        self.text().map(into_chunks)
    }

    fn instructions(&self) -> Vec<Instruction> {
        let chunks = self.text_chunks().unwrap_or_else(|| self.data_chunks());
        preprocess_bytecode::<DecoderConfig>(&chunks)
    }
}

fn run_vm_impl(data: &[u8], text: Option<&[u8]>) -> Option<GuestResult> {
    let binary = Binary::new(data, text);

    let instructions = binary.instructions();
    let tape = SimpleTape::new(&instructions);
    let mut ram = RamWithRomRegion::<{ common_constants::ROM_SECOND_WORD_BITS }>::from_rom_content(
        &binary.data_chunks(),
        crate::rv32im::common::constants::TOTAL_MEM_SIZE,
    );
    let mut state = State::initial_with_counters(CountersT::default());
    let mut snapshotter = SimpleSnapshotter::<CountersT, {common_constants::ROM_SECOND_WORD_BITS}>::new_with_cycle_limit(DEFAULT_CYCLES, state);
    let mut non_determinism = QuasiUARTSource::default();

    let cycles_bound = DEFAULT_CYCLES;
    log::debug!("Starting target VM...");
    let is_program_finished = VM::<CountersT>::run_basic_unrolled(
        &mut state,
        &mut ram,
        &mut snapshotter,
        &tape,
        cycles_bound,
        &mut non_determinism,
    );
    log::debug!("VM stopped. Program finished? {is_program_finished}");

    let registers = is_program_finished.then(|| {
        std::array::from_fn(|idx| {
            // We want registers A0-7, which are aliases to registers X10-17
            let reg_idx = idx + 10;
            state.registers[reg_idx].value
        })
    });

    #[cfg(feature = "prover")]
    {
        let chunks = binary.data_chunks();
        let text = binary.text_chunks();
        crate::rv32im::prover::prove_vm_result(
            &mut snapshotter,
            &mut state,
            &mut ram,
            text.as_deref().unwrap_or(&chunks),
            &tape,
            &chunks,
            cycles_bound,
        );
    }
    registers
}

pub fn run_vm<const ABORT: bool>(data: &[u8], text: Option<&[u8]>) -> Option<GuestResult> {
    match std::panic::catch_unwind(|| run_vm_impl(data, text)) {
        Ok(tr) => tr,
        Err(err) => match err.downcast::<String>() {
            Ok(s) => check_panic_string::<ABORT>(*s),
            Err(err) => match err.downcast::<&'static str>() {
                Ok(s) => check_panic_string::<ABORT>(*s),
                Err(_) => {
                    if ABORT {
                        log::error!("Unknown error type");
                        std::process::abort();
                    } else {
                        panic!("Unknown error type");
                    }
                }
            },
        },
    }
}
