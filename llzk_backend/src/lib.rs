use anyhow::Result;
use clap::ValueEnum;
use llzk::prelude::*;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::cs_reference::BasicAssembly;
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::field::Mersenne31Field;
use std::fs::File;
use std::fs::{self};
use std::io::Write;
use std::path::Path;

use crate::builder::ModuleBuilder;
use crate::codegen::EmitLLZKInModule as _;
use crate::codegen::NamedCircuitOutput;
use crate::output_format::OutputFormat;

use llzk::targets::pcl::translate_module;

mod builder;
mod codegen;
mod field;
mod lookups;
// mod expr;
pub mod output_format;

pub fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .init();
}

/// Generate the `add_sub_lui_auipc_mop` circuit.
pub fn gen_add_sub_lui_auipc_mop(
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
) -> Result<()> {
    use add_sub_lui_auipc_mop::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use add_sub_lui_auipc_mop::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn;

    generate_circuit_command(
        "add_sub_lui_auipc_mop",
        output,
        format,
        opt_level,
        (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            add_sub_lui_auipc_mop_table_addition_fn(cs);
            add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(cs);
        },
    )
}

/// Generate the `jump_branch_slt` circuit with `SUPPORT_SIGNED=true`
/// (all invocations appear use this configuration).
pub fn gen_jump_branch_slt(output: &str, format: OutputFormat, opt_level: OptLevel) -> Result<()> {
    use jump_branch_slt::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use jump_branch_slt::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_table_addition_fn;

    generate_circuit_command(
        "jump_branch_slt",
        output,
        format,
        opt_level,
        (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            jump_branch_slt_table_addition_fn(cs);
            jump_branch_slt_circuit_with_preprocessed_bytecode::<_, _, true>(cs);
        },
    )
}

/// A wrapper for the two circuit outputs, that being MLIR formats (LLZK and PCL IR)
/// and PCL code.
enum GenCircuitResult<'ctx> {
    Mlir(&'ctx Module<'ctx>),
    Pcl(String),
}

impl<'ctx> GenCircuitResult<'ctx> {
    /// Construct a new result from the given MLIR module based on the expected
    /// output format.
    pub fn new(format: OutputFormat, module: &'ctx Module<'ctx>) -> Result<Self> {
        Ok(match format {
            OutputFormat::Llzk | OutputFormat::PclMlir => Self::Mlir(module),
            OutputFormat::Pcl => Self::Pcl(translate_module(module)?),
        })
    }

    /// Write the result to the given file.
    pub fn dump<F: Write>(&self, file: &mut F) -> Result<()> {
        match self {
            GenCircuitResult::Mlir(module) => write!(file, "{}", module.as_operation())?,
            GenCircuitResult::Pcl(picus_program) => write!(file, "{}", picus_program)?,
        }
        Ok(())
    }
}

fn generate_circuit_command(
    name: &str,
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
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
    let mut module = llzk_module(Location::unknown(&ctx));
    let builder = ModuleBuilder::new(&ctx, &module);

    println!("Circuit Output:\n{:#?}", circuit_output);
    println!("Compiled:\n{:#?}", _compiled);

    // Add the circuit output to it.
    let named_circuit_output = NamedCircuitOutput::new(circuit_output, name);
    named_circuit_output.emit_llzk(&builder)?;

    // Verify the module
    verify_operation_with_diags(&module.as_operation())?;

    // Run optimizer
    run_optimizer_pipeline(&ctx, &mut module, format, opt_level)?;

    // Verify again
    verify_operation_with_diags(&module.as_operation())?;

    // Convert to the correct output format
    let res = GenCircuitResult::new(format, &module)?;

    // Write to file
    write_result(&res, format, output, name)?;

    Ok(())
}

fn write_result<'ctx>(
    res: &GenCircuitResult<'ctx>,
    format: OutputFormat,
    output: &str,
    name: &str,
) -> Result<()> {
    match output {
        // Stdout.
        "-" => {
            let mut file = std::io::stdout();
            res.dump(&mut file)?;
            eprintln!("Written successfully!");
        }
        // A file.
        output
            if [".llzk", ".mlir", ".pcl"]
                .into_iter()
                .any(|suffix| output.ends_with(suffix)) =>
        {
            let outpath = Path::new(output);
            let mut file = File::create(outpath).map_err(anyhow::Error::from)?;
            res.dump(&mut file)?;
            println!("Written successfully: {}", outpath.display());
        }
        // A directory.
        output => {
            // Write to file
            let file_name = format!("{}.{}", name, format.extension());
            let outpath = Path::new(output).join(file_name);
            // Ensure parent directories exist
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent).map_err(anyhow::Error::from)?;
            }
            let mut file = File::create(&outpath).map_err(anyhow::Error::from)?;
            res.dump(&mut file)?;
            println!("Written successfully: {}", outpath.display());
        }
    }
    Ok(())
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OptLevel {
    /// No optimizations
    #[value(name = "0")]
    O0,
    /// Basic MLIR optimizations
    #[value(name = "1")]
    O1,
    /// MLIR and LLZK optimizations
    #[value(name = "2")]
    O2,
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            self.to_possible_value()
                .expect("ValueEnum variant should always have a PossibleValue")
                .get_name(),
        )
    }
}

fn run_optimizer_pipeline(
    ctx: &Context,
    module: &mut Module,
    format: OutputFormat,
    opt_level: OptLevel,
) -> Result<()> {
    let pm = PassManager::new(ctx);
    // First cleanup the IR
    match opt_level {
        OptLevel::O0 => {} // No opt.
        OptLevel::O1 => {
            pm.add_pass(melior_passes::create_cse());
            pm.add_pass(melior_passes::create_canonicalizer());
        }
        OptLevel::O2 => {
            pm.add_pass(melior_passes::create_canonicalizer());
            pm.add_pass(llzk::passes::create_redundant_read_and_write_elimination_pass());
            pm.add_pass(melior_passes::create_cse());
            pm.add_pass(melior_passes::create_canonicalizer());
        }
    }
    // Then convert to the output format
    match format {
        OutputFormat::Llzk => {} // LLZK is the default
        OutputFormat::PclMlir | OutputFormat::Pcl => {
            // Convert to PCL IR
            pm.add_pass(llzk::passes::create_array_to_scalar_pass());
            pm.add_pass(llzk::passes::create_pcl_lowering_pass());
        }
    }

    pm.run(module)?;
    Ok(())
}
