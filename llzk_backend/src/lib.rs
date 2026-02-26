use anyhow::Result;
use llzk::prelude::*;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::cs_reference::BasicAssembly;
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::field::Mersenne31Field;
use std::fs::File;
use std::fs::{self};
use std::io::Write as _;
use std::path::Path;

use crate::codegen::GenerateLlzk as _;

mod builder;
mod codegen;

pub fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .init();
}

pub fn dump_add_sub_lui_auipc_mop(output: &str) -> Result<()> {
    use add_sub_lui_auipc_mop::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use add_sub_lui_auipc_mop::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn;

    dump_llzk_command(
        "add_sub_lui_auipc_mop",
        output,
        (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            add_sub_lui_auipc_mop_table_addition_fn(cs);
            add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(cs);
        },
    )
}

fn dump_llzk_command(
    name: &str,
    output: &str,
    bytecode_size: usize,
    trace_len_log2: usize,
    synthesis_fn: impl Fn(&mut BasicAssembly<Mersenne31Field>),
) -> Result<()> {
    let mut cs = BasicAssembly::<Mersenne31Field>::new();

    synthesis_fn(&mut cs);

    let (circuit_output, _maybe_wit_placer) = cs.finalize();

    // taken from add_sub_lui_auipc_mop::get_circuit:
    let compiler = OneRowCompiler::<Mersenne31Field>::default();
    // We don't want the compiled artifact as much as we want the constraints
    // that the compilation process adds.
    let _compiled = compiler.compile_executor_circuit_assuming_preprocessed_bytecode(
        circuit_output.clone(),
        bytecode_size,
        trace_len_log2,
    );

    // Generate an empty LLZK module
    let ctx = LlzkContext::new();
    let module = llzk_module(Location::unknown(&ctx));

    // Add the circuit output to it.
    circuit_output.generate_in_module(&ctx, &module, name)?;

    // Verify the module
    verify_operation_with_diags(&module.as_operation())?;

    // Write to file
    let outpath = Path::new(output).join(format!("{name}.llzk"));
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

    Ok(())
}
