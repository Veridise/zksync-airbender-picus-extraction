use anyhow::Result;
use llzk::prelude::*;
use prover::common_constants;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::circuit::CircuitOutput;
use prover::cs::cs::circuit::ShuffleRamMemQuery;
use prover::cs::cs::cs_reference::BasicAssembly;
use prover::cs::cs::placeholder::Placeholder;
use prover::cs::cs::witness_placer::graph_description::RawExpression;
use prover::cs::definitions::Variable;
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::field::Mersenne31Field;
use std::collections::HashMap;
use std::fs::File;
use std::fs::{self};
use std::io::Write;
use std::path::Path;

use crate::builder::ModuleEnv;
use crate::codegen::CircuitBundle;
use crate::codegen::EmitLlzkInModule as _;
use crate::config::LlzkStructLayout;
use crate::config::OptLevel;
use crate::output_format::OutputFormat;
use crate::witness::WitnessComputation;

use llzk::targets::pcl::translate_module;

mod builder;
mod codegen;
pub mod config;
mod constraints;
mod field;
mod lookups;
#[cfg(test)]
mod test_helpers;
mod witness;
// mod expr;
pub mod output_format;

/// Generate the `add_sub_lui_auipc_mop` circuit.
pub fn gen_add_sub_lui_auipc_mop(
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
    layout: LlzkStructLayout,
) -> Result<()> {
    use add_sub_lui_auipc_mop::dump_ssa_form;
    use add_sub_lui_auipc_mop::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use add_sub_lui_auipc_mop::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn;
    let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

    generate_circuit_command(
        "add_sub_lui_auipc_mop",
        output,
        format,
        opt_level,
        layout,
        bytecode_size,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            add_sub_lui_auipc_mop_table_addition_fn(cs);
            add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(cs);
        },
        dump_ssa_form,
    )
}

/// Generate the `jump_branch_slt` circuit with `SUPPORT_SIGNED=true`
/// (all invocations appear use this configuration).
pub fn gen_jump_branch_slt(
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
    layout: LlzkStructLayout,
) -> Result<()> {
    use jump_branch_slt::dump_ssa_form;
    use jump_branch_slt::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use jump_branch_slt::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_table_addition_fn;
    let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

    generate_circuit_command(
        "jump_branch_slt",
        output,
        format,
        opt_level,
        layout,
        bytecode_size,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            jump_branch_slt_table_addition_fn(cs);
            jump_branch_slt_circuit_with_preprocessed_bytecode::<_, _, true>(cs);
        },
        dump_ssa_form,
    )
}

/// Generate the `load_store_subword_only` circuit.
pub fn gen_load_store_subword_only(
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
    layout: LlzkStructLayout,
) -> Result<()> {
    use load_store_subword_only::dump_ssa_form;
    use load_store_subword_only::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
    use load_store_subword_only::TRACE_LEN_LOG2;
    use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_circuit_with_preprocessed_bytecode;
    use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_table_addition_fn;
    let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

    generate_circuit_command(
        "load_store_subword_only",
        output,
        format,
        opt_level,
        layout,
        bytecode_size,
        TRACE_LEN_LOG2 as usize,
        |cs| {
            subword_only_load_store_table_addition_fn(cs);
            subword_only_load_store_circuit_with_preprocessed_bytecode::<
                _,
                _,
                { common_constants::ROM_SECOND_WORD_BITS },
            >(cs);
        },
        dump_ssa_form,
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

/// Build, lower, and serialize one LLZK circuit family from the given synthesis
/// and witness SSA functions.
#[allow(clippy::too_many_arguments)]
fn generate_circuit_command(
    name: &str,
    output: &str,
    format: OutputFormat,
    opt_level: OptLevel,
    layout: LlzkStructLayout,
    bytecode_size: usize,
    trace_len_log2: usize,
    synthesis_fn: impl Fn(&mut BasicAssembly<Mersenne31Field>),
    witness_ssa_fn: impl FnOnce(&[u32]) -> Vec<Vec<RawExpression<Mersenne31Field>>>,
) -> Result<()> {
    let mut cs = BasicAssembly::<Mersenne31Field>::new();
    // Placeholder ROM image used during LLZK extraction.
    //
    // The LLZK backend currently emits circuit-family IR rather than program-specific IR, so it
    // does not receive a concrete bytecode image from the CLI. Some unrolled circuit helpers
    // still require a fixed-size ROM slice during setup because they build bytecode-dependent
    // lookup tables such as `AlignedRomRead`. We use an all-zeroes one because:
    // - circuits that only care about ROM size use it only to satisfy their length checks; and
    // - circuits with bytecode-backed lookup tables will call external functions as placeholders
    //   for the real lookups, so they will still not use the all-zero image contents.
    let bytecode = vec![0u32; bytecode_size];

    synthesis_fn(&mut cs);

    let (circuit_output, _maybe_wit_placer) = cs.finalize();
    let substitutions = merge_llzk_placeholder_aliases(&circuit_output);

    // From this point we intentionally build two different artifacts from the same circuit:
    // - `compiled_artifact` is the column-layout view with constraint expressions over logical
    //   variables used to emit LLZK constraints (i.e., by `@constrain`).
    // - `witness_ssa_fn(&bytecode)` is the witness-evaluation program used by `@compute`. It is a
    //   sequence of typed [`RawExpression`] blocks that describes how to derive witness values and
    //   write them back into logical variables.
    //
    // We also preserve the circuit's placeholder substitution map and augment it with additional
    // aliases. Several shuffle-RAM witness placeholders are already represented by explicit LLZK
    // inputs/outputs via `shuffle_ram_queries`, but the core circuit code does not record them
    // in `substitutions`.
    let compiler = OneRowCompiler::<Mersenne31Field>::default();
    // The compilation process here also adds constraints.
    let compiled_artifact = compiler.compile_executor_circuit_assuming_preprocessed_bytecode(
        circuit_output.clone(),
        bytecode_size,
        trace_len_log2,
    );
    let witness = match layout {
        LlzkStructLayout::ComputeConstrain
        | LlzkStructLayout::Product
        | LlzkStructLayout::ComputeOnly => Some(WitnessComputation::new(
            compiled_artifact.clone(),
            witness_ssa_fn(&bytecode),
            substitutions,
        )),
        LlzkStructLayout::ConstrainOnly => None,
    };

    // Generate an empty LLZK module
    let ctx = LlzkContext::new();
    let mut module = llzk_module(Location::unknown(&ctx));
    let env: ModuleEnv<'_, Mersenne31Field> = ModuleEnv::new(&ctx, &module);

    // Add the circuit output to it.
    let circuit_bundle = CircuitBundle::new(name, layout, circuit_output, witness)?;
    circuit_bundle.emit_llzk(&env)?;

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

/// Merge the core circuit substitutions with the extra placeholder aliases that LLZK can derive
/// from the extracted shuffle-RAM queries.
///
/// The shared circuit library already records substitutions for executor-state placeholders (e.g.,
/// mapping [`Placeholder::PcInit`] to a [`Variable`]), but some witness placeholders are only
/// visible indirectly through [`ShuffleRamMemQuery`] values. LLZK treats those query values as
/// struct inputs/outputs, so we need these aliases for witness generation so the `@compute` and
/// `@constrain` logic target the same set of LLZK args/members.
fn merge_llzk_placeholder_aliases<F: prover::field::PrimeField>(
    circuit_output: &CircuitOutput<F>,
) -> HashMap<(Placeholder, usize), Variable> {
    let mut substitutions = circuit_output.substitutions.clone();
    for (key, variable) in
        derive_shuffle_ram_placeholder_aliases(&circuit_output.shuffle_ram_queries)
    {
        substitutions.entry(key).or_insert(variable);
    }
    substitutions
}

/// Derive aliases for shuffle-RAM placeholders that are already exposed to the [`CircuitOutput`].
///
/// The source circuit code uses the same register variables for multiple placeholders, and they
/// differ between the [`CircuitOutput`] and [`WitnessComputation`], so adding these aliases
/// allows `@compute` and `@constrain` to reference the same final LLZK members/arguments for
/// computation and constraints.
///
/// Sources:
/// - In `get_rs1_as_shuffle_ram` and `get_rs2_as_shuffle_ram` (`cs/src/machine/utils.rs`), the
///   registers allocated from `FirstRegMem` and `SecondRegMem` are passed directly into
///   `form_mem_op_for_register_only`, so the placeholder value and
///   [`ShuffleRamMemQuery::read_value`] are literally the same two variables.
/// - In the legacy destination-write helpers `set_rd_with_mask_as_shuffle_ram` and
///   `set_rd_without_mask_as_shuffle_ram`, the register allocated from `WriteRdReadSetWitness`
///   becomes the returned query's `read_value`, again without creating a second variable.
/// - In the newer decode/reduced-machine paths (`decode_and_read_operands.rs` /
///   `reduced_machine_ops.rs`), the placeholders `ShuffleRamReadValue(0)`,
///   `ShuffleRamReadValue(1)`, and `ShuffleRamReadValue(2)` are each allocated first and then
///   written directly into `ShuffleRamMemQuery.read_value`.
/// - In the unrolled load/store families, `WriteRegMemReadWitness` is assigned into the same
///   `rd_or_store_ram_access_query_read_value` limbs that are later added as shuffle-RAM query 2,
///   so aliasing it to query 2's `read_value` preserves the existing witness flow.
fn derive_shuffle_ram_placeholder_aliases(
    queries: &[ShuffleRamMemQuery],
) -> HashMap<(Placeholder, usize), Variable> {
    let mut aliases = HashMap::new();

    // `ShuffleRamReadValue(i)` is only defined for `i in {0, 1, 2}`.
    for (query_index, query) in queries.iter().take(3).enumerate() {
        insert_register_alias(
            &mut aliases,
            Placeholder::ShuffleRamReadValue(query_index),
            query.read_value,
        );
    }
    // query 0 is the RS1 read slot (`FirstRegMem` / `ShuffleRamReadValue(0)`):

    if let Some(query) = queries.first() {
        insert_register_alias(&mut aliases, Placeholder::FirstRegMem, query.read_value);
    }
    // query 1 is the RS2 read slot (`SecondRegMem` / `ShuffleRamReadValue(1)`)
    if let Some(query) = queries.get(1) {
        insert_register_alias(&mut aliases, Placeholder::SecondRegMem, query.read_value);
    }
    // query 2 is the destination prior-value slot (`WriteRdReadSetWitness`,
    //   `WriteRegMemReadWitness`, and `ShuffleRamReadValue(2)`)
    if let Some(query) = queries.get(2) {
        insert_register_alias(
            &mut aliases,
            Placeholder::WriteRdReadSetWitness,
            query.read_value,
        );
        insert_register_alias(
            &mut aliases,
            Placeholder::WriteRegMemReadWitness,
            query.read_value,
        );
    }

    aliases
}

/// Insert both limbs of a register-valued placeholder alias.
fn insert_register_alias(
    aliases: &mut HashMap<(Placeholder, usize), Variable>,
    placeholder: Placeholder,
    register: [Variable; 2],
) {
    for (subindex, variable) in register.into_iter().enumerate() {
        aliases.entry((placeholder, subindex)).or_insert(variable);
    }
}

/// Write `res` to the specified `output` destination.
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

#[cfg(test)]
mod tests {
    use super::*;
    use prover::cs::cs::circuit::ShuffleRamQueryType;

    fn register_query(local_timestamp_in_cycle: usize, read_value: [u64; 2]) -> ShuffleRamMemQuery {
        ShuffleRamMemQuery {
            query_type: ShuffleRamQueryType::RegisterOnly {
                register_index: Variable(100 + local_timestamp_in_cycle as u64),
            },
            local_timestamp_in_cycle,
            read_value: [Variable(read_value[0]), Variable(read_value[1])],
            write_value: [Variable(read_value[0]), Variable(read_value[1])],
        }
    }

    #[test]
    fn shuffle_placeholder_aliases_cover_legacy_register_reads() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_query(1, [20, 21]),
            register_query(2, [30, 31]),
        ]);

        assert_eq!(aliases[&(Placeholder::FirstRegMem, 0)], Variable(10));
        assert_eq!(aliases[&(Placeholder::FirstRegMem, 1)], Variable(11));
        assert_eq!(aliases[&(Placeholder::SecondRegMem, 0)], Variable(20));
        assert_eq!(aliases[&(Placeholder::SecondRegMem, 1)], Variable(21));
        assert_eq!(
            aliases[&(Placeholder::WriteRdReadSetWitness, 0)],
            Variable(30)
        );
        assert_eq!(
            aliases[&(Placeholder::WriteRegMemReadWitness, 1)],
            Variable(31)
        );
    }

    #[test]
    fn shuffle_placeholder_aliases_cover_supported_shuffle_reads() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_query(1, [20, 21]),
            register_query(2, [30, 31]),
            register_query(3, [40, 41]),
        ]);

        assert_eq!(
            aliases[&(Placeholder::ShuffleRamReadValue(0), 0)],
            Variable(10)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamReadValue(1), 1)],
            Variable(21)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamReadValue(2), 0)],
            Variable(30)
        );
        assert!(!aliases.contains_key(&(Placeholder::ShuffleRamReadValue(3), 0)));
        assert!(!aliases.contains_key(&(Placeholder::ShuffleRamReadValue(3), 1)));
    }
}
