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
fn check_panic_string(s: impl AsRef<str>) -> Option<GuestResult> {
    if s.as_ref()
        .starts_with("Illegal instruction encounteted at PC =")
    {
        return None;
    }

    log::error!("Target raised error: {}", s.as_ref());
    std::process::abort()
}

// We are assuming that the whole binary is just the .text section, which could be not.
// If the compiled binary has other sections of data that are part of the ROM we are going to miss
// them.
// For now, and until we see that's actually an issue, we are going to roll with the assumption
// since we are going to generate random programs anyway.

struct Binary<'d> {
    data: Cow<'d, [u8]>,
}

type DecoderConfig = FullUnsignedMachineDecoderConfig;
type CountersT = DelegationsAndFamiliesCounters;

impl<'d> Binary<'d> {
    fn new(data: &'d [u8]) -> Self {
        let delta = data.len() % 4;
        let data = if delta != 0 {
            // Pad the data with 0 to keep alignment.
            let mut vec = Vec::with_capacity(data.len() + delta);
            vec.extend_from_slice(data);
            vec.extend(std::iter::repeat_n(0u8, delta));
            assert_eq!(vec.len() % 4, 0);
            Cow::Owned(vec)
        } else {
            Cow::Borrowed(data)
        };
        Self { data }
    }

    fn data(&self) -> &[u8] {
        self.data.as_ref()
    }

    fn data_chunks(&self) -> Vec<u32> {
        let (chunks, tail) = self.data().as_chunks::<4>();
        assert_eq!(tail.len(), 0);
        chunks.iter().copied().map(u32::from_le_bytes).collect()
    }

    fn instructions(&self) -> Vec<Instruction> {
        preprocess_bytecode::<DecoderConfig>(&self.data_chunks())
    }
}

fn run_vm_impl(data: &[u8]) -> Option<GuestResult> {
    let binary = Binary::new(data);

    let instructions = binary.instructions();
    let tape = SimpleTape::new(&instructions);
    let mut ram = RamWithRomRegion::<{ common_constants::ROM_SECOND_WORD_BITS }>::from_rom_content(
        &binary.data_chunks(),
        crate::rv32im::common::constants::TOTAL_MEM_SIZE,
    );
    let mut state = State::initial_with_counters(CountersT::default());
    let mut snapshotter = SimpleSnapshotter::<CountersT, {common_constants::ROM_SECOND_WORD_BITS}>::new_with_cycle_limit(DEFAULT_CYCLES, state);
    let mut non_determinism = QuasiUARTSource::default();

    let is_program_finished = VM::<CountersT>::run_basic_unrolled(
        &mut state,
        &mut ram,
        &mut snapshotter,
        &tape,
        DEFAULT_CYCLES,
        &mut non_determinism,
    );

    is_program_finished.then(|| {
        std::array::from_fn(|idx| {
            // We want registers A0-7, which are aliases to registers X10-17
            let reg_idx = idx + 10;
            state.registers[reg_idx].value
        })
    })
}

pub fn run_vm(data: &[u8]) -> Option<GuestResult> {
    match std::panic::catch_unwind(|| run_vm_impl(data)) {
        Ok(tr) => tr,
        Err(err) => match err.downcast::<String>() {
            Ok(s) => check_panic_string(*s),
            Err(err) => match err.downcast::<&'static str>() {
                Ok(s) => check_panic_string(*s),
                Err(_) => {
                    log::error!("Unknown error type");
                    std::process::abort();
                }
            },
        },
    }
}
