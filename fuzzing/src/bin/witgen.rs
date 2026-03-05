use clap::Parser;
use fuzzing::setup_logging;
use fuzzing::witgen::run;
use fuzzing::witgen::targets::Circuits;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    circuit: Circuits,
}

fn main() {
    setup_logging();
    let cli = Cli::parse();
    run(cli.circuit);
}
