use anyhow::Result;
use llzk::prelude::*;
use melior::ir::operation::OperationLike;
use melior::ir::operation::OperationPrintingFlags;
use prover::common_constants;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::circuit::CircuitOutput;
use prover::cs::cs::circuit::ShuffleRamMemQuery;
use prover::cs::cs::circuit::ShuffleRamQueryType;
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
use crate::codegen::SpecialCsrPropertiesMetadata;
use crate::config::ConstraintLoweringMode;
use crate::config::DebugLocationStyle;
use crate::config::LlzkStructLayout;
use crate::config::OptLevel;
use crate::config::UnusedVariablePolicy;
use crate::output_format::OutputFormat;
use crate::witness::WitnessComputation;

use llzk::targets::pcl::translate_module;

mod builder;
mod codegen;
pub mod config;
mod constraints;
mod field;
mod lookups;
pub mod output_format;
#[cfg(test)]
mod test_helpers;
mod witness;

type SynthesisFn = fn(&mut BasicAssembly<Mersenne31Field>);
type WitnessSsaFn = fn(&[u32]) -> Vec<Vec<RawExpression<Mersenne31Field>>>;

/// Static recipe for generating one circuit family.
#[derive(Clone, Copy)]
struct CircuitRecipe {
    name: &'static str,
    bytecode_size: usize,
    trace_len_log2: usize,
    synthesis_fn: SynthesisFn,
    witness_ssa_fn: WitnessSsaFn,
}

/// Shared generation options for emitting one or more circuit families.
#[derive(Clone, Debug)]
pub struct CircuitGenerationConfig {
    pub output: String,
    pub format: OutputFormat,
    pub opt_level: OptLevel,
    pub layout: LlzkStructLayout,
    pub debug_location_style: DebugLocationStyle,
    pub constraint_lowering_mode: ConstraintLoweringMode,
    pub unused_variable_policy: UnusedVariablePolicy,
    pub emit_suspicious_unused: bool,
}

impl CircuitGenerationConfig {
    /// Generate the `add_sub_lui_auipc_mop` circuit.
    pub fn gen_add_sub_lui_auipc_mop(&self) -> Result<()> {
        use add_sub_lui_auipc_mop::dump_ssa_form;
        use add_sub_lui_auipc_mop::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use add_sub_lui_auipc_mop::TRACE_LEN_LOG2;
        use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "add_sub_lui_auipc_mop",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                add_sub_lui_auipc_mop_table_addition_fn(cs);
                add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the `jump_branch_slt` circuit with `SUPPORT_SIGNED=true`.
    pub fn gen_jump_branch_slt(&self) -> Result<()> {
        use jump_branch_slt::dump_ssa_form;
        use jump_branch_slt::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use jump_branch_slt::TRACE_LEN_LOG2;
        use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::jump_branch_slt::jump_branch_slt_table_addition_fn;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "jump_branch_slt",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                jump_branch_slt_table_addition_fn(cs);
                jump_branch_slt_circuit_with_preprocessed_bytecode::<_, _, true>(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the `load_store_subword_only` circuit.
    pub fn gen_load_store_subword_only(&self) -> Result<()> {
        use load_store_subword_only::dump_ssa_form;
        use load_store_subword_only::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use load_store_subword_only::TRACE_LEN_LOG2;
        use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_table_addition_fn;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "load_store_subword_only",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                subword_only_load_store_table_addition_fn(cs);
                subword_only_load_store_circuit_with_preprocessed_bytecode::<
                    _,
                    _,
                    { common_constants::ROM_SECOND_WORD_BITS },
                >(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the `load_store_word_only` circuit.
    pub fn gen_load_store_word_only(&self) -> Result<()> {
        use load_store_word_only::dump_ssa_form;
        use load_store_word_only::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use load_store_word_only::TRACE_LEN_LOG2;
        use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_table_addition_fn;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "load_store_word_only",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                word_only_load_store_table_addition_fn(cs);
                // TODO: `RomRead` / `RomAddressSpaceSeparator` table contents depend on the
                // concrete bytecode image. Do not synthesize those tables from the mock bytecode.
                word_only_load_store_circuit_with_preprocessed_bytecode::<
                    _,
                    _,
                    { common_constants::ROM_SECOND_WORD_BITS },
                >(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the signed `mul_div` circuit.
    pub fn gen_mul_div(&self) -> Result<()> {
        use mul_div::dump_ssa_form;
        use mul_div::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use mul_div::TRACE_LEN_LOG2;
        use prover::cs::machine::ops::unrolled::mul_div::mul_div_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::mul_div::mul_div_table_addition_fn;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "mul_div",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                mul_div_table_addition_fn(cs);
                // SUPPORT_SIGNED = true is the usage in the frontend.
                mul_div_circuit_with_preprocessed_bytecode::<_, _, true>(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the `shift_binary_csr` circuit.
    pub fn gen_shift_binary_csr(&self) -> Result<()> {
        use prover::cs::machine::machine_configurations::create_csr_table_for_delegation;
        use prover::cs::machine::ops::unrolled::shift_binary_csr::shift_binop_csrrw_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::shift_binary_csr::shift_binop_csrrw_table_addition_fn;
        use prover::cs::tables::LookupWrapper;
        use prover::cs::tables::TableType;
        use shift_binary_csr::dump_ssa_form;
        use shift_binary_csr::ALLOWED_DELEGATION_CSRS;
        use shift_binary_csr::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use shift_binary_csr::TRACE_LEN_LOG2;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "shift_binary_csr",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                let csr_table = create_csr_table_for_delegation::<Mersenne31Field>(
                    true,
                    ALLOWED_DELEGATION_CSRS,
                    TableType::SpecialCSRProperties.to_table_id(),
                );
                shift_binop_csrrw_table_addition_fn(cs);
                cs.add_table_with_content(
                    TableType::SpecialCSRProperties,
                    LookupWrapper::Dimensional3(csr_table),
                );
                shift_binop_csrrw_circuit_with_preprocessed_bytecode(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }

    /// Generate the `unified_reduced_machine` circuit.
    pub fn gen_unified_reduced_machine(&self) -> Result<()> {
        use prover::cs::machine::machine_configurations::create_csr_table_for_delegation;
        use prover::cs::machine::ops::unrolled::reduced_machine_ops::reduced_machine_circuit_with_preprocessed_bytecode;
        use prover::cs::machine::ops::unrolled::reduced_machine_ops::reduced_machine_table_addition_fn;
        use prover::cs::tables::LookupWrapper;
        use prover::cs::tables::TableType;
        use unified_reduced_machine::dump_ssa_form;
        use unified_reduced_machine::ALLOWED_DELEGATION_CSRS;
        use unified_reduced_machine::ROM_ADDRESS_SPACE_SECOND_WORD_BITS;
        use unified_reduced_machine::TRACE_LEN_LOG2;
        let bytecode_size = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;

        self.generate_circuit(CircuitRecipe {
            name: "unified_reduced_machine",
            bytecode_size,
            trace_len_log2: TRACE_LEN_LOG2 as usize,
            synthesis_fn: |cs| {
                let csr_table = create_csr_table_for_delegation::<Mersenne31Field>(
                    true,
                    ALLOWED_DELEGATION_CSRS,
                    TableType::SpecialCSRProperties.to_table_id(),
                );
                reduced_machine_table_addition_fn(cs);
                cs.add_table_with_content(
                    TableType::SpecialCSRProperties,
                    LookupWrapper::Dimensional3(csr_table),
                );
                // TODO: the reduced-machine ROM-backed setup tables depend on the actual bytecode
                // image. Keep the family generic for now instead of materializing them from the
                // mock bytecode used for LLZK extraction.
                reduced_machine_circuit_with_preprocessed_bytecode::<
                    _,
                    _,
                    { common_constants::ROM_SECOND_WORD_BITS },
                >(cs);
            },
            witness_ssa_fn: dump_ssa_form,
        })
    }
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
            GenCircuitResult::Mlir(module) => {
                // pretty_form is not parsable by llzk-opt
                let flags = OperationPrintingFlags::new().enable_debug_info(true, false);
                write!(
                    file,
                    "{}",
                    module.as_operation().to_string_with_flags(flags)?
                )?
            }
            GenCircuitResult::Pcl(picus_program) => write!(file, "{}", picus_program)?,
        }
        Ok(())
    }
}

impl CircuitGenerationConfig {
    /// Build, lower, and serialize one LLZK circuit family from the given synthesis and witness
    /// SSA functions.
    fn generate_circuit(&self, recipe: CircuitRecipe) -> Result<()> {
        let mut cs = BasicAssembly::<Mersenne31Field>::new();
        // Placeholder ROM image used during LLZK extraction.
        //
        // The LLZK backend currently emits circuit-family IR rather than program-specific IR, so
        // it does not receive a concrete bytecode image from the CLI. The zero-filled slice here
        // is only for APIs that require a ROM-sized input to finish circuit construction or SSA
        // extraction.
        let bytecode = vec![0u32; recipe.bytecode_size];

        (recipe.synthesis_fn)(&mut cs);

        let (circuit_output, _maybe_wit_placer) = cs.finalize();
        let substitutions = merge_llzk_placeholder_aliases(&circuit_output);
        let special_csr_properties = SpecialCsrPropertiesMetadata::new(&circuit_output);

        // From this point we intentionally build two different artifacts from the same circuit:
        // - `compiled_artifact` is the column-layout view with constraint expressions over logical
        //   variables used to emit LLZK constraints (i.e., by `@constrain`).
        // - `witness_ssa_fn(&bytecode)` is the witness-evaluation program used by `@compute`. It is
        //   a sequence of typed [`RawExpression`] blocks that describes how to derive witness
        //   values and write them back into logical variables.
        //
        // We also preserve the circuit's placeholder substitution map and augment it with
        // additional aliases. Several shuffle-RAM witness placeholders are already represented by
        // explicit LLZK inputs/outputs via `shuffle_ram_queries`, but the core circuit code does
        // not record them in `substitutions`.
        let compiler = OneRowCompiler::<Mersenne31Field>::default();
        // The compilation process here also adds constraints.
        let compiled_artifact = compiler.compile_executor_circuit_assuming_preprocessed_bytecode(
            circuit_output.clone(),
            recipe.bytecode_size,
            recipe.trace_len_log2,
        );
        // Keep witness SSA available even for `ConstrainOnly` layouts so usage-based extraction
        // and signal classification can reason about the same logical variable universe.
        let witness = WitnessComputation::new(
            compiled_artifact.clone(),
            (recipe.witness_ssa_fn)(&bytecode),
            substitutions,
            special_csr_properties,
        );

        // Generate an empty LLZK module.
        let ctx = LlzkContext::new();
        let module_location = format!("llzk://layout/module/{}", recipe.name);
        let mut module = llzk_module(Location::new(&ctx, &module_location, 0, 0));
        let env: ModuleEnv<'_, Mersenne31Field> =
            ModuleEnv::new(&ctx, &module, self.debug_location_style);

        let circuit_bundle = CircuitBundle::new(
            recipe.name,
            self.layout,
            self.constraint_lowering_mode,
            self.unused_variable_policy,
            self.emit_suspicious_unused,
            circuit_output,
            compiled_artifact,
            witness,
        )?;
        circuit_bundle.emit_llzk(&env)?;

        verify_operation_with_diags(&module.as_operation())?;
        run_optimizer_pipeline(&ctx, &mut module, self.format, self.opt_level)?;
        verify_operation_with_diags(&module.as_operation())?;

        let res = GenCircuitResult::new(self.format, &module)?;
        write_result(&res, self.format, &self.output, recipe.name)?;

        Ok(())
    }
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
/// - Those same paths also allocate `ShuffleRamAddress(1)` and `ShuffleRamAddress(2)` first and
///   then store the resulting registers directly in `ShuffleRamQueryType::RegisterOrRam.address`.
///   LLZK now exposes those query addresses as ordinary inputs, so witness lowering can read the
///   existing boundary value instead of issuing a second oracle call.
/// - `ShuffleRamQueryType::RegisterOrRam` also stores a separate `is_register` discriminator. When
///   that flag is a real circuit variable, it is exposed as an LLZK input and can safely alias
///   `ShuffleRamIsRegisterAccess(i)`. We intentionally do not synthesize aliases for constant
///   discriminators because there is no boundary variable to map them to.
/// - In the unrolled load/store families, `WriteRegMemReadWitness` is assigned into the same
///   `rd_or_store_ram_access_query_read_value` limbs that are later added as shuffle-RAM query 2,
///   so aliasing it to query 2's `read_value` preserves the existing witness flow.
/// - Those same families also route `WriteRegMemWriteValue` and `ShuffleRamWriteValue(2)` into
///   query 2's `write_value`. `unified_reduced_machine` still consumes that slot as a pre-existing
///   witness input, so LLZK keeps it input-backed for that circuit only. The alias is recorded here
///   so witness lowering can canonicalize those legacy placeholders onto the same boundary value
///   when that mixed policy is active.
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
        insert_query_address_alias(&mut aliases, Placeholder::ShuffleRamAddress(1), query);
        insert_query_is_register_alias(
            &mut aliases,
            Placeholder::ShuffleRamIsRegisterAccess(1),
            query,
        );
    }
    // query 2 is the destination prior-value slot (`WriteRdReadSetWitness`,
    //   `WriteRegMemReadWitness`, `ShuffleRamReadValue(2)`, `ShuffleRamAddress(2)`, and
    //   `ShuffleRamIsRegisterAccess(2)`)
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
        insert_register_alias(
            &mut aliases,
            Placeholder::WriteRegMemWriteValue,
            query.write_value,
        );
        insert_register_alias(
            &mut aliases,
            Placeholder::ShuffleRamWriteValue(2),
            query.write_value,
        );
        insert_query_address_alias(&mut aliases, Placeholder::ShuffleRamAddress(2), query);
        insert_query_is_register_alias(
            &mut aliases,
            Placeholder::ShuffleRamIsRegisterAccess(2),
            query,
        );
    }

    aliases
}

/// Insert both limbs of a `RegisterOrRam` query address if the query carries one.
fn insert_query_address_alias(
    aliases: &mut HashMap<(Placeholder, usize), Variable>,
    placeholder: Placeholder,
    query: &ShuffleRamMemQuery,
) {
    if let ShuffleRamQueryType::RegisterOrRam { address, .. } = query.query_type {
        insert_register_alias(aliases, placeholder, address);
    }
}

/// Insert the discriminator variable for a `RegisterOrRam` query when the source circuit stores
/// it as a real boolean variable rather than a constant.
fn insert_query_is_register_alias(
    aliases: &mut HashMap<(Placeholder, usize), Variable>,
    placeholder: Placeholder,
    query: &ShuffleRamMemQuery,
) {
    if let ShuffleRamQueryType::RegisterOrRam { is_register, .. } = query.query_type {
        if let Some(variable) = is_register.get_variable() {
            insert_scalar_alias(aliases, placeholder, variable);
        }
    }
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

/// Insert the single variable backing a scalar-valued placeholder alias.
fn insert_scalar_alias(
    aliases: &mut HashMap<(Placeholder, usize), Variable>,
    placeholder: Placeholder,
    variable: Variable,
) {
    aliases.entry((placeholder, 0)).or_insert(variable);
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

    fn register_or_ram_query(
        local_timestamp_in_cycle: usize,
        read_value: [u64; 2],
        write_value: [u64; 2],
        address: [u64; 2],
        is_register: prover::cs::types::Boolean,
    ) -> ShuffleRamMemQuery {
        ShuffleRamMemQuery {
            query_type: ShuffleRamQueryType::RegisterOrRam {
                is_register,
                address: [Variable(address[0]), Variable(address[1])],
            },
            local_timestamp_in_cycle,
            read_value: [Variable(read_value[0]), Variable(read_value[1])],
            write_value: [Variable(write_value[0]), Variable(write_value[1])],
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

    #[test]
    fn shuffle_placeholder_aliases_cover_supported_shuffle_writes() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_query(1, [20, 21]),
            ShuffleRamMemQuery {
                query_type: ShuffleRamQueryType::RegisterOnly {
                    register_index: Variable(102),
                },
                local_timestamp_in_cycle: 2,
                read_value: [Variable(30), Variable(31)],
                write_value: [Variable(40), Variable(41)],
            },
        ]);

        assert_eq!(
            aliases[&(Placeholder::ShuffleRamWriteValue(2), 0)],
            Variable(40)
        );
        assert_eq!(
            aliases[&(Placeholder::WriteRegMemWriteValue, 1)],
            Variable(41)
        );
    }

    #[test]
    fn shuffle_placeholder_aliases_cover_supported_shuffle_addresses() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_or_ram_query(
                1,
                [20, 21],
                [20, 21],
                [50, 51],
                prover::cs::types::Boolean::Constant(true),
            ),
            register_or_ram_query(
                2,
                [30, 31],
                [40, 41],
                [60, 61],
                prover::cs::types::Boolean::Constant(true),
            ),
        ]);

        assert_eq!(
            aliases[&(Placeholder::ShuffleRamAddress(1), 0)],
            Variable(50)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamAddress(1), 1)],
            Variable(51)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamAddress(2), 0)],
            Variable(60)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamAddress(2), 1)],
            Variable(61)
        );
    }

    #[test]
    fn shuffle_placeholder_aliases_cover_variable_is_register_discriminators() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_or_ram_query(
                1,
                [20, 21],
                [20, 21],
                [50, 51],
                prover::cs::types::Boolean::Is(Variable(70)),
            ),
            register_or_ram_query(
                2,
                [30, 31],
                [40, 41],
                [60, 61],
                prover::cs::types::Boolean::Is(Variable(71)),
            ),
        ]);

        assert_eq!(
            aliases[&(Placeholder::ShuffleRamIsRegisterAccess(1), 0)],
            Variable(70)
        );
        assert_eq!(
            aliases[&(Placeholder::ShuffleRamIsRegisterAccess(2), 0)],
            Variable(71)
        );
    }

    #[test]
    fn shuffle_placeholder_aliases_skip_constant_is_register_discriminators() {
        let aliases = derive_shuffle_ram_placeholder_aliases(&[
            register_query(0, [10, 11]),
            register_or_ram_query(
                1,
                [20, 21],
                [20, 21],
                [50, 51],
                prover::cs::types::Boolean::Constant(true),
            ),
            register_or_ram_query(
                2,
                [30, 31],
                [40, 41],
                [60, 61],
                prover::cs::types::Boolean::Constant(false),
            ),
        ]);

        assert!(!aliases.contains_key(&(Placeholder::ShuffleRamIsRegisterAccess(1), 0)));
        assert!(!aliases.contains_key(&(Placeholder::ShuffleRamIsRegisterAccess(2), 0)));
    }
}
