use clap::ValueEnum;
use prover::risc_v_simulator::runner::run_simple_simulator;
use risc_v_simulator::sim::BinarySource;
use risc_v_simulator::sim::SimulatorConfig;

use crate::rv32im::unicorn::run_on_unicorn;

mod unicorn;

/// Available fuzzing modes
#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Dumb fuzzing that runs the prover with random binary inputs.
    Dumb,
    /// Runs the input against the target and an unicorn VM, comparing the results.
    Unicorn,
}

/// Configures the fuzzer to expect only one input and stop after that regardless of status.
fn configure_singleton_test_mode() {
    std::env::set_var("AFL_FUZZER_LOOPCOUNT", "1");
}

pub fn run(mode: Mode, test_one: bool) {
    if test_one {
        configure_singleton_test_mode();
    }

    match mode {
        Mode::Dumb => dumb_fuzzer(test_one),
        Mode::Unicorn => oracle_fuzzer(test_one),
    }
}

/// Default amount of cycles, taken from `tools/cli/src/prover_utils.rs`.
const DEFAULT_CYCLES: usize = 32_000_000;
const ENTRYPOINT: u32 = 0;

type GuestResult = [u32; 8];

fn run_on_put(data: &[u8]) -> GuestResult {
    run_simple_simulator(SimulatorConfig {
        bin: BinarySource::Slice(data),
        entry_point: ENTRYPOINT,
        cycles: DEFAULT_CYCLES,
        diagnostics: None,
    })
}

fn dumb_fuzzer(print_result: bool) {
    afl::fuzz!(|data| {
        let result = run_on_put(data);
        if print_result {
            log::info!("result = {result:?}");
        }
    })
}

fn oracle_fuzzer(print_result: bool) {
    afl::fuzz!(|data| {
        let oracle_result = match run_on_unicorn(data) {
            Ok(or) => or,
            Err(err) => {
                // Stop if the oracle failed.
                if print_result {
                    log::info!("Oracle failed. Skipping...");
                    log::info!("Oracle failure: {err}");
                }
                return;
            }
        };

        if print_result {
            log::info!("Oracle: {oracle_result:?}");
        }

        let target_result = run_on_put(data);

        if print_result {
            log::info!("Target: {target_result:?}");
        }

        assert_eq!(
            oracle_result, target_result,
            "Oracle and result produced different register outputs!"
        );
    })
}
