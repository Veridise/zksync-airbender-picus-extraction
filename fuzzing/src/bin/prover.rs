use std::process::ExitCode;

use clap::Parser;
use fuzzing::prover::run;
use fuzzing::prover::Cli;
use fuzzing::setup_logging;

fn main() -> ExitCode {
    setup_logging();
    let cli = Cli::parse();
    match run(cli) {
        Ok(_) => {
            println!("Command finished!");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Fuzzer encountered a fatal error: {err}");
            ExitCode::FAILURE
        }
    }
}
