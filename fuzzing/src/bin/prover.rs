use clap::Parser;
use fuzzing::prover::run;
use fuzzing::prover::Cli;
use fuzzing::setup_logging;

fn main() {
    setup_logging();
    let cli = Cli::parse();
    run(cli);
}
