use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use llzk_backend::{dump_add_sub_lui_auipc_mop, setup_logging};

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
enum Circuits {
    AddSubLuiAuipcMop,
}

const CIRCUITS: &[(Circuits, fn(&str) -> Result<()>)] =
    &[(Circuits::AddSubLuiAuipcMop, dump_add_sub_lui_auipc_mop)];

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate LLZK IR from the circuits
    DumpLLZK {
        #[arg(short, long)]
        output: String,
        #[arg(long)]
        circuit: Circuits,
    },
}

fn main() -> Result<()> {
    setup_logging();
    let cli = Cli::parse();
    match &cli.command {
        Commands::DumpLLZK { output, circuit } => {
            CIRCUITS
                .iter()
                .find_map(|(name, handler)| (name == circuit).then_some(handler))
                .expect("circuit without a handler function")(output)?;
        }
    }
    Ok(())
}
