use std::cell::RefCell;
use std::num::TryFromIntError;
use std::rc::Rc;

use unicorn_engine::uc_error;
use unicorn_engine::Arch;
use unicorn_engine::Mode;
use unicorn_engine::Prot;
use unicorn_engine::RegisterRISCV;
use unicorn_engine::Unicorn;

use crate::rv32im::GuestResult;
use crate::rv32im::DEFAULT_CYCLES;
use crate::rv32im::ENTRYPOINT;

// Taken from `examples/scripts/lds/memory.x`.
const ROM: u64 = 4 * 1024 * 1024;
const RAM: u64 = crate::rv32im::common::constants::TOTAL_MEM_SIZE as u64 - ROM;

fn configure_vm<'vm>(data: &[u8]) -> Result<Unicorn<'vm, ()>, uc_error> {
    let mut vm = Unicorn::new(Arch::RISCV, Mode::RISCV32)?;
    log::debug!("Created vm: {vm:?}");
    for (base, size, perms) in [
        // ROM section
        (ENTRYPOINT as u64, ROM, Prot::EXEC),
        // RAM section, right after the ROM
        (ROM, RAM, Prot::READ | Prot::WRITE),
    ] {
        log::debug!("Creating memory map at address 0x{base} with {size} bytes");
        vm.mem_map(base, size, perms)?;
    }
    vm.mem_write(ENTRYPOINT as u64, data)?;
    log::debug!("Wrote program to entrypoint");
    Ok(vm)
}

#[derive(Debug)]
pub enum Error {
    Unicorn(uc_error),
    U32(TryFromIntError),
    UnexpectedRegisterListSize(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unicorn(e) => write!(f, "unicorn error: {e}"),
            Error::U32(e) => write!(f, "u64 to u32 conversion error: {e}"),
            Error::UnexpectedRegisterListSize(s) => write!(f, "expected 8 registers, but got {s}"),
        }
    }
}

impl From<uc_error> for Error {
    fn from(value: uc_error) -> Self {
        Self::Unicorn(value)
    }
}

impl From<TryFromIntError> for Error {
    fn from(value: TryFromIntError) -> Self {
        Self::U32(value)
    }
}

/// Runs the given binary in an unicorn VM and returns the result following the same ABI.
pub fn run_on_unicorn(data: &[u8]) -> Result<Option<GuestResult>, Error> {
    // Set to true if unicorn encounters instructions that are not support by the target, like 2
    // byte instructions or RV32A instructions.
    // If after execution this flag is true we report that the oracle failed, regardless of what
    // actually happened.
    let unsupported_instructions = Rc::new(RefCell::new(false));
    // Clone so we have a different variable go into the closure.
    let ui = unsupported_instructions.clone();
    let prev_pc = Rc::new(RefCell::new(None));
    let mut vm = configure_vm(data)?;
    let hook_id = vm.add_code_hook(ENTRYPOINT as u64, RAM, |vm, addr, size| {
        log::debug!("CODE HOOK!! (0x{addr:08x}) ({size})");
        {
            let prev = prev_pc.borrow();
            if *prev == Some(addr) {
                vm.emu_stop().unwrap();
            }
        }
        let _ = prev_pc.borrow_mut().insert(addr);
        if size == 2 {
            *ui.borrow_mut() = true;
            return;
        }
        let mut instr = [0, 0, 0, 0];
        match vm.mem_read(addr, &mut instr) {
            Ok(_) => {}
            Err(err) => {
                log::error!("Error in hook at 0x{addr:016x}: {err}");
                return;
            }
        };
        let instr = u32::from_le_bytes(instr);
        log::debug!("instr = 0x{instr:08x}");
    })?;
    log::debug!("Unicorn VM configured");

    if let Err(err) = vm.emu_start(ENTRYPOINT as u64, data.len() as u64, 0, DEFAULT_CYCLES) {
        // If unicorn fails during execution we consider it a 'success' that returns no output.
        vm.remove_hook(hook_id)?;
        log::debug!("Unicorn failed while executing: {err}");
        return Ok(None);
    }
    log::debug!("Execution completed");
    if *unsupported_instructions.borrow() {
        log::debug!("Unicorn encountered instructions that are not supported by the target");
        return Ok(None);
    }
    vm.remove_hook(hook_id)?;
    [
        RegisterRISCV::A0,
        RegisterRISCV::A1,
        RegisterRISCV::A2,
        RegisterRISCV::A3,
        RegisterRISCV::A4,
        RegisterRISCV::A5,
        RegisterRISCV::A6,
        RegisterRISCV::A7,
    ]
    .into_iter()
    .map(|reg| -> Result<u32, Error> { Ok(vm.reg_read(reg)?.try_into()?) })
    .collect::<Result<Vec<u32>, _>>()?
    .try_into()
    .map_err(|v: Vec<u32>| Error::UnexpectedRegisterListSize(v.len()))
    .map(Some)
}
