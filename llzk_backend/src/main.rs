use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use llzk_backend::dump_add_sub_lui_auipc_mop;
use llzk_backend::output_format::OutputFormat;
use llzk_backend::setup_logging;

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
enum Circuits {
    AddSubLuiAuipcMop,
}

const CIRCUITS: &[(Circuits, fn(&str, OutputFormat, u8) -> Result<()>)] =
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
        #[arg(short, long, default_value_t = OutputFormat::Pcl)]
        format: OutputFormat,
        #[arg(short = 'O', default_value_t = 1)]
        opt_level: u8,
    },
}

fn main() -> Result<()> {
    setup_logging();
    let cli = Cli::parse();
    match &cli.command {
        Commands::DumpLLZK {
            output,
            circuit,
            format,
            opt_level,
        } => {
            CIRCUITS
                .iter()
                .find_map(|(name, handler)| (name == circuit).then_some(handler))
                .expect("circuit without a handler function")(
                output, *format, *opt_level
            )?;
        }
    }
    Ok(())
}
