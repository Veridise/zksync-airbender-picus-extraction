use std::num::TryFromIntError;

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
const RAM: u64 = 1024 * 1024 * 1024 - ROM;

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
        log::debug!("Created memory map at address 0x{base} with {size} bytes");
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
pub fn run_on_unicorn(data: &[u8]) -> Result<GuestResult, Error> {
    let mut vm = configure_vm(data)?;
    log::debug!("Unicorn VM configured");
    vm.emu_start(ENTRYPOINT as u64, data.len() as u64, 0, DEFAULT_CYCLES)?;
    log::debug!("Execution completed");
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
}
