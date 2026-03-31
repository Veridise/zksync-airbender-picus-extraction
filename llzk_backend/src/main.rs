use anyhow::Result;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use llzk_backend::config::ConstraintLoweringMode;
use llzk_backend::config::DebugLocationStyle;
use llzk_backend::config::LlzkStructLayout;
use llzk_backend::config::OptLevel;
use llzk_backend::config::UnusedVariablePolicy;
use llzk_backend::output_format::OutputFormat;
use llzk_backend::CircuitGenerationConfig;

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
enum Circuits {
    AddSubLuiAuipcMop,
    JumpBranchSlt,
    LoadStoreSubwordOnly,
    LoadStoreWordOnly,
    MulDiv,
    ShiftBinaryCsr,
    UnifiedReducedMachine,
}

type CircuitFnTuple = (Circuits, fn(&CircuitGenerationConfig) -> Result<()>);
const CIRCUITS: &[CircuitFnTuple] = &[
    (
        Circuits::AddSubLuiAuipcMop,
        CircuitGenerationConfig::gen_add_sub_lui_auipc_mop,
    ),
    (
        Circuits::JumpBranchSlt,
        CircuitGenerationConfig::gen_jump_branch_slt,
    ),
    (
        Circuits::LoadStoreSubwordOnly,
        CircuitGenerationConfig::gen_load_store_subword_only,
    ),
    (
        Circuits::LoadStoreWordOnly,
        CircuitGenerationConfig::gen_load_store_word_only,
    ),
    (Circuits::MulDiv, CircuitGenerationConfig::gen_mul_div),
    (
        Circuits::ShiftBinaryCsr,
        CircuitGenerationConfig::gen_shift_binary_csr,
    ),
    (
        Circuits::UnifiedReducedMachine,
        CircuitGenerationConfig::gen_unified_reduced_machine,
    ),
];

#[derive(Args, Clone)]
struct GenerateArgs {
    /// Output directory or output file name
    #[arg(short, long)]
    output: String,
    #[arg(short, long, default_value_t = OutputFormat::Pcl)]
    format: OutputFormat,
    #[arg(short = 'O', default_value_t = OptLevel::O1)]
    opt_level: OptLevel,
    #[arg(long, default_value_t = LlzkStructLayout::ComputeConstrain)]
    layout: LlzkStructLayout,
    #[arg(long, default_value_t = DebugLocationStyle::FileLineCol)]
    debug_location_style: DebugLocationStyle,
    #[arg(long, default_value_t = ConstraintLoweringMode::Logical)]
    constraint_lowering_mode: ConstraintLoweringMode,
    #[arg(long, default_value_t = UnusedVariablePolicy::Warn)]
    unused_variable_policy: UnusedVariablePolicy,
    #[arg(long, default_value_t = false)]
    emit_suspicious_unused: bool,
}

impl GenerateArgs {
    fn generation_config(&self) -> CircuitGenerationConfig {
        CircuitGenerationConfig {
            output: self.output.clone(),
            format: self.format,
            opt_level: self.opt_level,
            layout: self.layout,
            debug_location_style: self.debug_location_style,
            constraint_lowering_mode: self.constraint_lowering_mode,
            unused_variable_policy: self.unused_variable_policy,
            emit_suspicious_unused: self.emit_suspicious_unused,
        }
    }
}

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
        #[arg(long)]
        circuit: Circuits,
        #[command(flatten)]
        args: GenerateArgs,
    },
    /// Generate outputs for all supported circuits
    GenAllCircuits {
        #[command(flatten)]
        args: GenerateArgs,
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
        Commands::GenCircuit { circuit, args } => {
            let config = args.generation_config();
            CIRCUITS
                .iter()
                .find_map(|(name, handler)| (name == circuit).then_some(handler))
                .expect("circuit without a handler function")(&config)?;
        }
        Commands::GenAllCircuits { args } => {
            let config = args.generation_config();
            for (_, handler) in CIRCUITS {
                handler(&config)?;
            }
        }
    }
    Ok(())
}
