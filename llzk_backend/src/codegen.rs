//! LLZK circuit emission coordination and variable extraction helpers.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;

use anyhow::anyhow;
use anyhow::Result;
use llzk::prelude::*;
use prover::cs::cs::circuit::CircuitOutput;
use prover::cs::definitions::OpcodeFamilyCircuitState;
use prover::cs::definitions::Variable;
use prover::field::PrimeField;

use crate::builder::*;
use crate::config::LlzkStructLayout;
use crate::constraints::AddConstraints;
use crate::constraints::EmitLLZKInConstrain;
use crate::field::FieldInfo;
use crate::witness::WitnessComputation;

/// Trait implemented by types that can emit LLZK IR within the module scope.
pub(crate) trait EmitLLZKInModule<'ctx, F: FieldInfo> {
    type Output;

    fn emit_llzk(&self, env: &ModuleEnv<'ctx, F>) -> Result<Self::Output>;
}

/// Extension trait for [`StructDefOpLike`] that adds a method for filling the `@compute`
/// function.
pub trait AddCompute<'ctx: 'op, 'op, F: FieldInfo>: StructDefOpLike<'ctx, 'op> {
    /// Invokes the callback scoped in `@compute`. The `struct.new` and `function.return %self`
    /// operations are added automatically and do not need to be inserted by the provided callback.
    fn add_compute(
        &'op self,
        env: &'ctx ModuleEnv<'ctx, F>,
        f: impl FnOnce(&mut OpsBuilder<'ctx, 'op, F>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let compute_fn = self.get_compute_func().ok_or_else(|| {
            anyhow!(
                "struct {} is missing its @compute function",
                StructDefOpLike::name(self)
            )
        })?;
        let mut ops_builder = OpsBuilder::new(env, compute_fn);
        f(&mut ops_builder)
    }
}

impl<'ctx: 'op, 'op, F: FieldInfo, T: StructDefOpMutLike<'ctx, 'op>> AddCompute<'ctx, 'op, F>
    for T
{
}

/// This enum holds information about extracted variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtractedVariable {
    /// A register value represented by a low and high limb.
    Register { low: Variable, high: Variable },
    /// A scalar field element.
    Scalar(Variable),
}

impl ExtractedVariable {
    /// Create a new register.
    pub fn register(reg: [Variable; 2]) -> Self {
        Self::Register {
            low: reg[0],
            high: reg[1],
        }
    }

    /// Create a new felt.
    pub fn scalar(v: Variable) -> Self {
        Self::Scalar(v)
    }

    /// Checks if the given `variable` is contained in the extraction.
    pub fn contains(&self, v: &Variable) -> bool {
        match self {
            ExtractedVariable::Register { low, high } => v == low || v == high,
            ExtractedVariable::Scalar(variable) => v == variable,
        }
    }

    /// Number of contained vars.
    pub fn num_vars(&self) -> usize {
        match self {
            ExtractedVariable::Register { .. } => 2,
            ExtractedVariable::Scalar(_) => 1,
        }
    }
}

/// Trait for extracting inputs, outputs, and intermediate variables from the implementing circuit
/// representation.
pub trait VariableExtractor {
    /// Extract all variables that need to be passed as inputs.
    fn get_inputs(&self) -> Result<Vec<ExtractedVariable>>;
    /// Extract all variables that need to be produced as outputs.
    fn get_outputs(&self) -> Result<Vec<ExtractedVariable>>;
    /// Extract all variables that are only internal.
    fn get_intermediates(&self) -> Result<Vec<ExtractedVariable>>;
}

impl<F: PrimeField> VariableExtractor for OpcodeFamilyCircuitState<F> {
    fn get_inputs(&self) -> Result<Vec<ExtractedVariable>> {
        // These inputs are the canonical LLZK boundary view of executor machine state. The source
        // circuit also tracks many of them through placeholder substitutions for the legacy
        // witness/oracle path, and `@compute` lowers those placeholder reads back to these inputs
        // so the same logical value is not derived from two unrelated sources downstream.
        let mut inputs = vec![
            ExtractedVariable::scalar(self.execute),
            ExtractedVariable::register(self.cycle_start_state.pc),
            ExtractedVariable::register(self.cycle_start_state.timestamp),
            ExtractedVariable::scalar(self.decoder_data.rs1_index),
            ExtractedVariable::scalar(self.decoder_data.rs2_index),
            ExtractedVariable::scalar(self.decoder_data.rd_index),
            ExtractedVariable::scalar(self.decoder_data.rd_is_zero),
            ExtractedVariable::register(self.decoder_data.imm),
            ExtractedVariable::scalar(self.decoder_data.funct3),
            ExtractedVariable::scalar(self.decoder_data.circuit_family_extra_mask),
        ];
        if let Some(v) = self.decoder_data.funct7 {
            inputs.push(ExtractedVariable::scalar(v));
        }
        inputs.sort();
        Ok(inputs)
    }

    fn get_outputs(&self) -> Result<Vec<ExtractedVariable>> {
        let mut outputs = vec![
            ExtractedVariable::register(self.cycle_end_state.pc),
            ExtractedVariable::register(self.cycle_end_state.timestamp),
        ];
        outputs.sort();
        Ok(outputs)
    }

    fn get_intermediates(&self) -> Result<Vec<ExtractedVariable>> {
        Ok(vec![])
    }
}

impl<F: PrimeField> VariableExtractor for CircuitOutput<F> {
    fn get_inputs(&self) -> Result<Vec<ExtractedVariable>> {
        // Inputs are:
        // - RAM read query values
        // - RAM write query prior values (these will show up in constraints sometimes)
        // - Inputs from the executor machine state
        let exec_state = &self
            .executor_machine_state
            .ok_or_else(|| anyhow!("executor_machine_state not initialized"))?;
        let mut inputs = exec_state.get_inputs()?;

        for query in &self.shuffle_ram_queries {
            inputs.push(ExtractedVariable::register(query.read_value));
        }
        inputs.sort();
        assert!(
            inputs.windows(2).all(|w| w[0] != w[1]),
            "found duplicate inputs"
        );
        Ok(inputs)
    }

    fn get_outputs(&self) -> Result<Vec<ExtractedVariable>> {
        // Outputs are:
        // - RAM write queries
        // - end state from the executor_machine_state
        let exec_state = &self
            .executor_machine_state
            .ok_or_else(|| anyhow!("executor_machine_state not initialized"))?;
        let mut outputs = exec_state.get_outputs()?;
        for query in &self.shuffle_ram_queries {
            if !query.is_readonly() {
                outputs.push(ExtractedVariable::register(query.write_value));
            }
        }
        outputs.sort();
        Ok(outputs)
    }

    fn get_intermediates(&self) -> Result<Vec<ExtractedVariable>> {
        // Intermediates are:
        // - everything else that isn't an input or output
        let io = [self.get_inputs()?, self.get_outputs()?].concat();
        // TODO: the prior values for RAM writes are technically separate variables,
        // but they don't cleanly fall into the inputs or outputs for now. So we just
        // ignore them for now, but they will need to be constrained by the shuffle
        // ram constraints.
        let mut intermediates = (0u64..u64::try_from(self.num_of_variables)?)
            .map(Variable)
            .filter(|v| {
                // TODO: We check the ram queries explicitly to ignore the prior write values
                let in_ram_reads = self
                    .shuffle_ram_queries
                    .iter()
                    .any(|&q| q.read_value[0] == *v || q.read_value[1] == *v);
                let in_io = io.iter().any(|x| x.contains(v));
                !in_io && !in_ram_reads
            })
            .map(ExtractedVariable::Scalar)
            .collect::<Vec<_>>();

        intermediates.sort();
        Ok(intermediates)
    }
}

fn num_vars(vars: impl IntoIterator<Item = ExtractedVariable>) -> usize {
    vars.into_iter().map(|v| v.num_vars()).sum()
}

/// Holds the circuit artifacts required to emit one LLZK circuit struct.
pub struct CircuitBundle<F: FieldInfo> {
    /// Name to give the emitted LLZK struct.
    name: String,
    /// The option for how to generate the `@compute`/`@constraint` or `@product`
    /// methods of the emitted LLZK struct.
    layout: LlzkStructLayout,
    /// The output of the airbender circuit, used for constraint and witness generation
    circuit_output: CircuitOutput<F>,
    /// The output of the witness SSA generation, used for generating witness computation in LLZK,
    /// if needed
    witness: Option<WitnessComputation<F>>,
}

impl<F: FieldInfo> CircuitBundle<F> {
    /// Create a new emission bundle for a single circuit.
    ///
    /// The `witness` is optional since not all layouts require it, but the generation of `witness`
    /// requires `circuit_output`, so `circuit_output` is always required.
    /// Will return an error if the witness is omitted for any layout other than
    /// [`LlzkStructLayout::ComputeOnly`]
    pub fn new(
        name: &str,
        layout: LlzkStructLayout,
        circuit_output: CircuitOutput<F>,
        witness: Option<WitnessComputation<F>>,
    ) -> Result<Self> {
        if matches!((layout, &witness), (LlzkStructLayout::ComputeOnly, None)) {
            anyhow::bail!("must provide witness for {}", layout);
        }
        Ok(Self {
            name: name.to_string(),
            layout,
            circuit_output,
            witness,
        })
    }

    /// Return a reference to the circuit's emitted struct name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<F: FieldInfo> Deref for CircuitBundle<F> {
    type Target = CircuitOutput<F>;

    fn deref(&self) -> &Self::Target {
        &self.circuit_output
    }
}

impl<'ctx, F: FieldInfo> EmitLLZKInModule<'ctx, F> for CircuitBundle<F> {
    type Output = ();

    fn emit_llzk(&self, env: &ModuleEnv<'ctx, F>) -> Result<Self::Output> {
        if !F::is_built_in() {
            panic!("non-built-in fields are not yet supported in LLZK module emission")
        }

        // TODO: Support product program
        if matches!(self.layout, LlzkStructLayout::Product) {
            anyhow::bail!("@product program generation is currently unsupported");
        }

        let mut struct_builder = StructBuilder::new(env, self.name());

        // Sanity check: all variables should be an input, output, or intermediate.
        let num_input_vars = num_vars(self.get_inputs()?);
        let num_output_vars = num_vars(self.get_outputs()?);
        let num_intermediate_vars = num_vars(self.get_intermediates()?);
        let extracted = num_input_vars + num_output_vars + num_intermediate_vars;
        assert_eq!(self.num_of_variables, extracted);

        let vars = StructVars::new(self, &mut struct_builder)?;
        let struct_op = struct_builder.build_in_module()?;

        if !matches!(&self.layout, LlzkStructLayout::ComputeOnly) {
            struct_op.add_constraints(
                env,
                |builder: &mut OpsBuilder<'_, '_, F>| -> Result<()> {
                    // Add some constants to reuse at the beginning here.
                    builder.insert_constant_at_start(builder.index_type(), 1)?;
                    builder.insert_constant_at_start(builder.index_type(), 0)?;
                    builder.insert_constant_at_start(builder.felt_type(), 1)?;
                    builder.insert_constant_at_start(builder.felt_type(), 0)?;
                    // Add boolean constraints.
                    for bool_var in self.boolean_vars.iter() {
                        let val = vars.get_constrain_val(builder, bool_var)?;
                        let _ = builder.felt_type();
                        builder.append_boolean_constraint(val)?;
                    }
                    // Add range constraints.
                    self.range_check_expressions
                        .emit_constrain(builder, &vars)?;
                    // Add lookup constraints.
                    self.lookups.emit_constrain(builder, &vars)?;
                    // Add all other constraints.
                    self.constraints.emit_constrain(builder, &vars)
                },
            )?;
        }

        if !matches!(&self.layout, LlzkStructLayout::ConstrainOnly) {
            let wit = self
                .witness
                .as_ref()
                .ok_or_else(|| anyhow!("must have witness specified"))?;
            struct_op.add_compute(env, |builder: &mut OpsBuilder<'_, '_, F>| {
                wit.emit_compute(builder, &vars)
            })?;

            wit.declare_runtime_externs(env)?;
        }
        Ok(())
    }
}

/// Holds the information about the variables and their representation in the LLZK struct.
pub struct StructVars<F: FieldInfo> {
    /// Maps internal and output Variables to a tuple `(member name, optional index if the member
    /// is an array type)`. All members are assumed to be either felts or "registers", which are
    /// flat, two-element felt arrays.
    member_map: HashMap<Variable, (String, Option<u64>)>,
    /// Maps input variables to a tuple `(input ordinal, optional limb index)`.
    ///
    /// The stored ordinal is zero-based with respect to the logical circuit inputs. Constraint
    /// lowering adds one when reading from `@constrain` because argument 0 is the struct `self`
    /// value, while witness lowering uses the ordinal directly in `@compute`.
    arg_map: HashMap<Variable, (usize, Option<u64>)>,
    /// Ties [`StructVars`] to a specific field. This is prefered to having every member
    /// take the [`FieldInfo`] struct as a parameter, because mixed-field operations are
    /// currently not supported.
    _field: PhantomData<F>,
}

impl<F: FieldInfo> StructVars<F> {
    /// Creates a new [`StructVars`] instance by:
    /// - Extracting struct inputs/outputs/intermediate variables (into [`ExtractedVariable`]s) from
    ///   the provided [`CircuitOutput`] instance,
    /// - Adding new struct arguments and members based on the [`ExtractedVariable`]s
    fn new<'ctx>(
        co: &CircuitOutput<F>,
        struct_builder: &mut StructBuilder<'ctx, '_, F>,
    ) -> Result<Self> {
        // Add inputs to struct.
        let mut arg_map: HashMap<Variable, (usize, Option<u64>)> = HashMap::new();
        for (input_num, input) in co.get_inputs()?.iter().enumerate() {
            match input {
                ExtractedVariable::Register { low, high } => {
                    arg_map.insert(*low, (input_num, Some(0)));
                    arg_map.insert(*high, (input_num, Some(1)));
                    let register_type = struct_builder.register_type();
                    struct_builder.with_input(register_type);
                }
                ExtractedVariable::Scalar(variable) => {
                    arg_map.insert(*variable, (input_num, None));
                    let felt_type = struct_builder.felt_type();
                    struct_builder.with_input(felt_type);
                }
            };
        }

        // Add outputs to struct.
        let mut member_map: HashMap<Variable, (String, Option<u64>)> = HashMap::new();
        for output in co.get_outputs()?.iter() {
            // TODO: better naming scheme
            match output {
                ExtractedVariable::Register { low, high } => {
                    let name = format!("out_reg_{}_{}", low.0, high.0);
                    member_map.insert(*low, (name.clone(), Some(0)));
                    member_map.insert(*high, (name.clone(), Some(1)));
                    let register_type = struct_builder.register_type();
                    struct_builder.with_member(name, register_type, true);
                }
                ExtractedVariable::Scalar(variable) => {
                    let name = format!("out_var_{}", variable.0);
                    member_map.insert(*variable, (name.clone(), None));
                    let felt_type = struct_builder.felt_type();
                    struct_builder.with_member(name, felt_type, true);
                }
            }
        }

        // Add intermediates to struct.
        for output in co.get_intermediates()?.iter() {
            match output {
                ExtractedVariable::Register { low, high } => {
                    let name = format!("internal_reg_{}_{}", low.0, high.0);
                    member_map.insert(*low, (name.clone(), Some(0)));
                    member_map.insert(*high, (name.clone(), Some(1)));
                    let register_type = struct_builder.register_type();
                    struct_builder.with_member(name, register_type, false);
                }
                ExtractedVariable::Scalar(variable) => {
                    let name = format!("internal_var_{}", variable.0);
                    member_map.insert(*variable, (name.clone(), None));
                    let felt_type = struct_builder.felt_type();
                    struct_builder.with_member(name, felt_type, false);
                }
            }
        }

        Ok(Self {
            member_map,
            arg_map,
            _field: PhantomData,
        })
    }

    /// Try to read a variable from the `@constrain` view of the struct.
    ///
    /// `@constrain` receives the struct instance as argument 0, so public inputs begin at
    /// argument 1.
    pub fn try_get_constrain_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        if let Some(val) = self.get_input_val_at_offset(builder, 1, var)? {
            Ok(Some(val))
        } else {
            self.get_constrain_member_val(builder, var)
        }
    }

    /// Read a variable from the `@constrain` view of the struct and error if it is unavailable.
    pub fn get_constrain_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        var: &Variable,
    ) -> Result<Value<'ctx, 'sco>> {
        self.try_get_constrain_val(builder, var)?
            .ok_or_else(|| anyhow!("Could not find {var:?} in constrain inputs or members"))
    }

    /// Try to read a variable from the `@compute` view of the struct.
    ///
    /// `@compute` does not receive a `self` argument. Its public inputs begin at argument 0 and
    /// the partially constructed witness struct is the result of the leading `struct.new`.
    ///
    /// This is the canonical path for circuit boundary values. Witness lowering prefers these
    /// inputs over runtime oracle hooks whenever an SSA placeholder is just another name for an
    /// already-exposed `@compute` argument.
    pub fn try_get_compute_input_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        self.get_input_val_at_offset(builder, 0, var)
    }

    /// Try to read a variable from the full `@compute` view of the struct.
    ///
    /// This checks the explicit function arguments first and then falls back to the partially
    /// constructed struct members.
    pub fn try_get_compute_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        self_value: Value<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        if let Some(val) = self.try_get_compute_input_val(builder, var)? {
            Ok(Some(val))
        } else {
            self.get_member_val_from(builder, self_value, var)
        }
    }

    /// Read a variable from the `@compute` view of the struct and error if it is unavailable.
    pub fn get_compute_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        self_value: Value<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Value<'ctx, 'sco>> {
        self.try_get_compute_val(builder, self_value, var)?
            .ok_or_else(|| anyhow!("Could not find {var:?} in compute inputs or members"))
    }

    /// Return `true` when `var` is one of the explicit `@compute` inputs.
    pub fn has_compute_input(&self, var: &Variable) -> bool {
        self.arg_map.contains_key(var)
    }

    /// Return `true` when `var` is stored as a struct member in the LLZK boundary.
    ///
    /// This is the key distinction for witness lowering when the one-row compiler maps a logical
    /// variable into the `MemorySubtree`: if the same variable is also exposed as an LLZK output or
    /// intermediate member, `@compute` should update the struct member rather than routing that
    /// write through the generic memory runtime hook.
    pub fn has_compute_member(&self, var: &Variable) -> bool {
        self.member_map.contains_key(var)
    }

    /// Return `true` when `var` is visible through either the `@compute` inputs or the returned
    /// struct.
    pub fn is_compute_exposed(&self, var: &Variable) -> bool {
        self.has_compute_input(var) || self.has_compute_member(var)
    }

    /// Update `var` with `value` by creating a `struct.writem` operation in `@compute` targeting
    /// the struct member that corresponds to `var`.
    ///
    /// The function handles both scalar members and register-valued members, where a single
    /// logical variable corresponds to one limb of a two-element array.
    pub fn assign_compute_member<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        self_value: Value<'ctx, 'sco>,
        var: &Variable,
        value: Value<'ctx, 'sco>,
    ) -> Result<()> {
        let (member_name, index) = self
            .member_map
            .get(var)
            .ok_or_else(|| anyhow!("Variable {var:?} is not stored as a struct member"))?;
        let location = builder.unknown_location();
        match index {
            None => builder.append_member_write(location, self_value, member_name, value),
            Some(index) => {
                let register = builder.append_member_read(
                    location,
                    self_value,
                    builder.register_type(),
                    member_name,
                )?;
                let indices = &[builder.get_constant_from_start(builder.index_type(), *index)?];
                builder.append_array_write(location, register, indices, value)?;
                builder.append_member_write(location, self_value, member_name, register)
            }
        }
    }

    fn get_input_val_at_offset<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        arg_offset: usize,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        match self.arg_map.get(var) {
            None => Ok(None),
            Some((arg_no, index)) => {
                let arg_val = builder.get_arg_value(*arg_no + arg_offset)?;
                let val = match index {
                    None => arg_val,
                    Some(index) => {
                        let indices =
                            &[builder.get_constant_from_start(builder.index_type(), *index)?];
                        builder.append_array_read(builder.unknown_location(), arg_val, indices)?
                    }
                };
                Ok(Some(val))
            }
        }
    }

    fn get_constrain_member_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        let self_val = builder.get_arg_value(0)?;
        self.get_member_val_from(builder, self_val, var)
    }

    /// Get the value from the struct member corresponding to `var` from the struct instance
    /// represented by `self_val`.
    fn get_member_val_from<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        self_val: Value<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        match self.member_map.get(var) {
            None => Ok(None),
            Some((member_name, index)) => {
                let location = builder.unknown_location();
                match index {
                    None => {
                        let member_ty = builder.felt_type();
                        let member_val = builder.append_member_read(
                            location,
                            self_val,
                            member_ty,
                            member_name,
                        )?;
                        Ok(Some(member_val))
                    }
                    Some(index) => {
                        let member_ty = builder.register_type();
                        let member_val = builder.append_member_read(
                            location,
                            self_val,
                            member_ty,
                            member_name,
                        )?;
                        let indices =
                            &[builder.get_constant_from_start(builder.index_type(), *index)?];
                        let read_val = builder.append_array_read(
                            builder.unknown_location(),
                            member_val,
                            indices,
                        )?;
                        Ok(Some(read_val))
                    }
                }
            }
        }
    }

    /// A test-only function that allows direct construction of [`StructVars`]
    /// using synthetic struct members and arguments rather than parsing them
    /// from the [`CircuitOutput`] object.
    #[cfg(test)]
    pub(crate) fn from_test_maps(
        member_map: HashMap<Variable, (String, Option<u64>)>,
        arg_map: HashMap<Variable, (usize, Option<u64>)>,
    ) -> Self {
        Self {
            member_map,
            arg_map,
            _field: PhantomData,
        }
    }
}
