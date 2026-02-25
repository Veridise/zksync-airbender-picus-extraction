use std::collections::HashMap;
use std::path::Path;

use clap::{Parser, Subcommand};
use llzk_backend::builder::{Builder, OpsBuilder, StructBuilder};
use prover::cs::definitions::Variable;
use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::{
    add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode, add_sub_lui_auipc_mop_table_addition_fn};
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::{cs::cs::cs_reference::BasicAssembly, field::Mersenne31Field};
use prover::cs::cs::circuit::Circuit;
use add_sub_lui_auipc_mop;
use add_sub_lui_auipc_mop::{TRACE_LEN_LOG2, ROM_ADDRESS_SPACE_SECOND_WORD_BITS};
use anyhow::{anyhow, Result};
use llzk::prelude::StructDefOpLike;
use llzk_backend::codegen::AddConstraints;
use llzk::prelude::*;
use llzk_backend::codegen::GenerateLlzk;
use std::fs::File;
use std::io::Write;
use std::fs;

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
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .init();
    let cli = Cli::parse();
    match &cli.command {
        Commands::DumpLLZK { output } => {
            let mut cs = BasicAssembly::<Mersenne31Field>::new();

            add_sub_lui_auipc_mop_table_addition_fn(&mut cs);
            add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(&mut cs);

            let (circuit_output, _maybe_wit_placer) = cs.finalize();

            // from [add_sub_lui_auipc_mop::get_circuit_for_rom_bound]
            let max_bytecode_size_in_words = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;
            // taken from add_sub_lui_auipc_mop::get_circuit:
            let compiler = OneRowCompiler::<Mersenne31Field>::default();
            // We don't want the compiled artifact as much as we want the constraints
            // that the compilation process adds.
            let _compiled = compiler.compile_executor_circuit_assuming_preprocessed_bytecode(
                circuit_output.clone(),
                max_bytecode_size_in_words,
                TRACE_LEN_LOG2 as usize,
            );

            // Generate an empty LLZK module
            let ctx = LlzkContext::new();
            let module = llzk_module(Location::unknown(&ctx));

            // Add the circuit output to it.
            circuit_output.generate_in_module(&ctx, &module)?;

            // Verify the module
            verify_operation_with_diags(&module.as_operation())?;

            // Write to file
            let outpath = Path::new(output).join("add_sub_lui_auipc_mop.llzk");
            // Ensure parent directories exist
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent).map_err(anyhow::Error::from)?;
            }
            let mut file = File::create(&outpath).map_err(anyhow::Error::from)?;
            write!(file, "{}", module.as_operation())?;
            println!("{} {}", "Written successfully:", outpath.display());

            // Also transform to PCL
            // TODO: Need to get the PCL pass exposed
            // llzk::passes::register_all_llzk_passes();
            // let pm = PassManager::new(&ctx);
            // pm.add_pass(llzk::passes::create_pcl_to_llzk_pass());
            // pm.run(&mut module).expect("failed to convert to PCL");
        }
    }
    Ok(())
}
