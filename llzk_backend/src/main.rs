use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use llzk_backend::config::LlzkStructLayout;
use llzk_backend::config::OptLevel;
use llzk_backend::gen_add_sub_lui_auipc_mop;
use llzk_backend::gen_jump_branch_slt;
use llzk_backend::gen_load_store_subword_only;
use llzk_backend::output_format::OutputFormat;

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
enum Circuits {
    AddSubLuiAuipcMop,
    JumpBranchSlt,
    LoadStoreSubwordOnly,
}

type CircuitFnTuple = (
    Circuits,
    fn(&str, OutputFormat, OptLevel, LlzkStructLayout) -> Result<()>,
);
const CIRCUITS: &[CircuitFnTuple] = &[
    (Circuits::AddSubLuiAuipcMop, gen_add_sub_lui_auipc_mop),
    (Circuits::JumpBranchSlt, gen_jump_branch_slt),
    (Circuits::LoadStoreSubwordOnly, gen_load_store_subword_only),
];

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate the specified output for the specified circuit
    GenCircuit {
        /// Output directory or output file name
        #[arg(short, long)]
        output: String,
        #[arg(long)]
        circuit: Circuits,
        #[arg(short, long, default_value_t = OutputFormat::Pcl)]
        format: OutputFormat,
        #[arg(short = 'O', default_value_t = OptLevel::O1)]
        opt_level: OptLevel,
        #[arg(long, default_value_t = LlzkStructLayout::ComputeConstrain)]
        layout: LlzkStructLayout,
    },
}

pub fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .init();
}

fn main() -> Result<()> {
    setup_logging();
    let cli = Cli::parse();
    match &cli.command {
        Commands::GenCircuit {
            output,
            circuit,
            format,
            opt_level,
            layout,
        } => {
            CIRCUITS
                .iter()
                .find_map(|(name, handler)| (name == circuit).then_some(handler))
                .expect("circuit without a handler function")(
                output, *format, *opt_level, *layout,
            )?;
        }
    }
    Ok(())
}
