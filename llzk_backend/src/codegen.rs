use std::collections::HashMap;

use anyhow::anyhow;
use anyhow::Result;
use llzk::dialect::{constrain, felt};
use llzk::prelude::*;
use prover::cs::constraint::Constraint;
use prover::cs::constraint::Term;
use prover::cs::cs::circuit::RangeCheckQuery;
use prover::cs::definitions::LookupInput;
use prover::cs::definitions::OpcodeFamilyCircuitState;
use prover::{
    cs::{cs::circuit::CircuitOutput, definitions::Variable},
    field::PrimeField,
};

use crate::builder::*;

/// This enum holds the possible representations for SSA values
pub enum SsaAddress<'ctx, 'val> {
    /// Represents a single variable that is neither an input or an output.
    /// It's encoded as a struct member of [`FeltType`].
    Intermediate(Value<'ctx, 'val>),
}

/// Trait implemented by types that can emit LLZK IR.
trait EmitLLZK<'ctx: 'sco, 'sco> {
    type Output;

    fn emit_llzk(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        vars: &StructVars,
    ) -> Result<Self::Output>;
}

impl<'ctx: 'sco, 'sco, T: EmitLLZK<'ctx, 'sco, Output = ()>> EmitLLZK<'ctx, 'sco> for Vec<T> {
    type Output = ();

    fn emit_llzk(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        vars: &StructVars,
    ) -> Result<Self::Output> {
        self.iter().try_for_each(|t| t.emit_llzk(builder, vars))
    }
}

/// Extension trait for [`StructDefOpLike`] that adds a method for filling the `@constrain` function.
pub trait AddConstraints<'ctx: 'op, 'op>: StructDefOpLike<'ctx, 'op> {
    /// Invokes the callback scoped in `@constrain`.
    ///
    /// All ops added with the [`OpsBuilder`] are automatically added to that function.
    fn add_constraints(
        &'op self,
        f: impl FnOnce(&mut OpsBuilder<'ctx, 'op>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let constrain_fn = self.get_constrain_func().ok_or_else(|| {
            anyhow!(
                "struct {} is missing its @constrain function",
                StructDefOpLike::name(self)
            )
        })?;
        let mut builder = OpsBuilder::new(unsafe { self.context().to_ref() }, constrain_fn);
        f(&mut builder)
    }
}

impl<'ctx: 'op, 'op, T: StructDefOpMutLike<'ctx, 'op>> AddConstraints<'ctx, 'op> for T {}

/// This enum holds information about extracted variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtractedVariable {
    /// A register value represented by a low and high limb.
    Register { low: Variable, high: Variable },
    /// A scalar field element
    Scalar(Variable),
}

impl ExtractedVariable {
    /// Create a new register
    pub fn register(reg: [Variable; 2]) -> Self {
        Self::Register {
            low: reg[0],
            high: reg[1],
        }
    }
    /// Create a new felt
    pub fn scalar(v: Variable) -> Self {
        Self::Scalar(v)
    }
    /// Checks if the given `variable` is contained in the extraction
    pub fn contains(&self, v: &Variable) -> bool {
        match self {
            ExtractedVariable::Register { low, high } => v == low || v == high,
            ExtractedVariable::Scalar(variable) => v == variable,
        }
    }
    /// Number of contained vars
    pub fn num_vars(&self) -> usize {
        match self {
            ExtractedVariable::Register { .. } => 2,
            ExtractedVariable::Scalar(_) => 1,
        }
    }
}

/// Trait for extracting inputs, outputs, and intermediate variables from the
/// implementing circuit representation.
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
        // - RAM read queries
        // - Inputs from the executor machine state
        let exec_state = &self
            .executor_machine_state
            .ok_or_else(|| anyhow!("executor_machine_state not initialized"))?;
        let mut inputs = exec_state.get_inputs()?;

        for query in &self.shuffle_ram_queries {
            if query.is_readonly() {
                inputs.push(ExtractedVariable::register(query.read_value));
            }
        }
        inputs.sort();
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
            .map(|i| Variable(i))
            .filter(|v| {
                // TODO: We check the ram queries explicitly to ignore the prior write values
                let in_ram_reads = self
                    .shuffle_ram_queries
                    .iter()
                    .any(|&q| q.read_value[0] == *v || q.read_value[1] == *v);
                let in_io = io.iter().any(|x| x.contains(v));
                !in_io && !in_ram_reads
            })
            .map(|v| ExtractedVariable::Scalar(v))
            .collect::<Vec<_>>();

        intermediates.sort();
        Ok(intermediates)
    }
}

/// Trait for generating LLZK within a given module.
pub trait GenerateLlzk {
    fn generate_in_module<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        struct_name: &str,
    ) -> Result<()>;
}

fn num_vars(vars: impl IntoIterator<Item = ExtractedVariable>) -> usize {
    vars.into_iter().map(|v| v.num_vars()).sum()
}

impl<F: PrimeField> GenerateLlzk for CircuitOutput<F> {
    // TODO: break down the monolith
    fn generate_in_module<'ctx>(
        &self,
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        struct_name: &str,
    ) -> Result<()> {
        let llzk_builder = Builder::new(&ctx);
        let mut struct_builder = StructBuilder::new(&ctx, struct_name);

        // Sanity check: all variables should be an input, output, or intermediate,
        // with the exception of the 2 variables used to encode the RAM write's
        // prior value (i.e., the read_value of one RAM query).
        let num_input_vars = num_vars(self.get_inputs()?);
        let num_output_vars = num_vars(self.get_outputs()?);
        let num_intermediate_vars = num_vars(self.get_intermediates()?);
        let extracted = num_input_vars + num_output_vars + num_intermediate_vars;
        let expected = self.num_of_variables - 2;
        assert_eq!(extracted, expected);

        let vars = StructVars::new(self, &mut struct_builder, &llzk_builder)?;
        let struct_op = struct_builder.build_in_module(&module)?;

        struct_op.add_constraints(|builder: &mut OpsBuilder<'_, '_>| -> Result<()> {
            // Add some constants to reuse at the beginning here.
            builder.insert_constant_at_start(builder.index_type(), 1)?;
            builder.insert_constant_at_start(builder.index_type(), 0)?;
            builder.insert_constant_at_start(builder.felt_type(), 1)?;
            builder.insert_constant_at_start(builder.felt_type(), 0)?;
            // Add boolean constraints
            for bool_var in self.boolean_vars.iter() {
                let val = vars.get_val(builder, bool_var)?;
                let _ = builder.felt_type();
                builder.append_boolean_constraint(val)?;
            }
            // Add range constraints
            self.range_check_expressions.emit_llzk(builder, &vars)?;
            // Add all other constraints
            self.constraints.emit_llzk(builder, &vars)
        })
    }
}

impl<'ctx: 'sco, 'sco, F: PrimeField> EmitLLZK<'ctx, 'sco> for RangeCheckQuery<F> {
    type Output = ();

    fn emit_llzk(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        vars: &StructVars,
    ) -> Result<Self::Output> {
        match &self.input {
            LookupInput::Variable(variable) => {
                let val = vars.get_val(builder, variable)?;
                builder.append_range_constraint(val, self.width)?;
            }
            LookupInput::Expression { .. } => todo!("expression range check"),
        }
        Ok(())
    }
}

impl<'ctx: 'sco, 'sco, F: PrimeField> EmitLLZK<'ctx, 'sco> for (Constraint<F>, bool) {
    type Output = ();

    fn emit_llzk(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        vars: &StructVars,
    ) -> Result<Self::Output> {
        let (constraint, _prevent_optimization) = self;

        let zero = builder.get_constant_from_start(builder.felt_type(), 0)?;
        let sum = constraint
            .terms
            .iter()
            .map(|term| term.emit_llzk(builder, vars))
            .try_fold(zero, |sum, term_val| {
                builder.append_op_with_result(felt::add(
                    builder.unknown_location(),
                    sum,
                    term_val?,
                )?)
            })?;
        builder.append_op_with_no_results(constrain::eq(builder.unknown_location(), sum, zero))
    }
}

impl<'ctx: 'sco, 'sco, F: PrimeField> EmitLLZK<'ctx, 'sco> for Term<F> {
    type Output = Value<'ctx, 'sco>;

    fn emit_llzk(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        vars: &StructVars,
    ) -> Result<Self::Output> {
        match self {
            Term::Constant(c) => {
                let coeff = c.as_u64_reduced();
                let coeff_opp = F::CHARACTERISTICS - coeff;
                let coeff_val = builder.get_constant_from_start(builder.felt_type(), coeff)?;
                Ok(if coeff < coeff_opp {
                    coeff_val
                } else {
                    builder
                        .append_op_with_result(felt::neg(builder.unknown_location(), coeff_val)?)?
                })
            }
            Term::Expression {
                coeff,
                inner,
                degree,
            } => {
                let coeff = coeff.as_u64_reduced();

                let coeff_opp = F::CHARACTERISTICS - coeff;
                let mut monomial = builder.get_constant_from_start(builder.felt_type(), 1)?;
                for var in inner.iter().take(*degree) {
                    let var_val = vars.get_val(builder, var)?;
                    let mul = felt::mul(builder.unknown_location(), monomial, var_val)?;
                    monomial = builder.append_op_with_result(mul)?;
                }

                Ok(if coeff < coeff_opp {
                    if coeff == 1 {
                        monomial
                    } else {
                        let coeff_val =
                            builder.get_constant_from_start(builder.felt_type(), coeff)?;
                        let mul = felt::mul(builder.unknown_location(), coeff_val, monomial)?;
                        builder.append_op_with_result(mul)?
                    }
                } else if coeff_opp == 1 {
                    builder
                        .append_op_with_result(felt::neg(builder.unknown_location(), monomial)?)?
                } else {
                    let coeff_opp_val =
                        builder.get_constant_from_start(builder.felt_type(), coeff_opp)?;
                    let mul = builder.append_op_with_result(felt::mul(
                        builder.unknown_location(),
                        coeff_opp_val,
                        monomial,
                    )?)?;
                    builder.append_op_with_result(felt::neg(builder.unknown_location(), mul)?)?
                })
            }
        }
    }
}

/// Holds the information about the variables and their representation in the LLZK struct.
struct StructVars {
    field_map: HashMap<Variable, (String, Option<u64>)>,
    arg_map: HashMap<Variable, (usize, Option<u64>)>,
}

impl StructVars {
    fn new<'ctx, F: PrimeField>(
        co: &CircuitOutput<F>,
        struct_builder: &mut StructBuilder<'ctx, '_>,
        llzk_builder: &Builder<'ctx>,
    ) -> Result<Self> {
        // Add inputs to struct
        // maps Variable to (input argument, optional index if array)
        let mut arg_map: HashMap<Variable, (usize, Option<u64>)> = HashMap::new();
        for (input_num, input) in co.get_inputs()?.iter().enumerate() {
            let arg_no = input_num + 1; // because of the self arg
            match input {
                ExtractedVariable::Register { low, high } => {
                    arg_map.insert(*low, (arg_no, Some(0)));
                    arg_map.insert(*high, (arg_no, Some(1)));
                    struct_builder.with_input(llzk_builder.register_type());
                }
                ExtractedVariable::Scalar(variable) => {
                    arg_map.insert(*variable, (arg_no, None));
                    struct_builder.with_input(llzk_builder.felt_type());
                }
            };
        }
        // Add outputs to struct
        // Maps CircuitOutput variable to (field name, index)
        let mut field_map: HashMap<Variable, (String, Option<u64>)> = HashMap::new();
        for output in co.get_outputs()?.iter() {
            // TODO: better naming scheme
            match &output {
                ExtractedVariable::Register { low, high } => {
                    let name = format!("out_reg_{}_{}", low.0, high.0);
                    field_map.insert(*low, (name.clone(), Some(0)));
                    field_map.insert(*high, (name.clone(), Some(1)));
                    struct_builder.with_member(name, llzk_builder.register_type(), true);
                }
                ExtractedVariable::Scalar(variable) => {
                    let name = format!("out_var_{}", variable.0);
                    field_map.insert(*variable, (name.clone(), None));
                    struct_builder.with_member(name, llzk_builder.felt_type(), true);
                }
            }
        }
        // Add intermediates to struct
        for output in co.get_intermediates()?.iter() {
            match &output {
                ExtractedVariable::Register { low, high } => {
                    let name = format!("internal_reg_{}_{}", low.0, high.0);
                    field_map.insert(*low, (name.clone(), Some(0)));
                    field_map.insert(*high, (name.clone(), Some(1)));
                    struct_builder.with_member(name, llzk_builder.register_type(), false);
                }
                ExtractedVariable::Scalar(variable) => {
                    let name = format!("internal_var_{}", variable.0);
                    field_map.insert(*variable, (name.clone(), None));
                    struct_builder.with_member(name, llzk_builder.felt_type(), false);
                }
            }
        }

        Ok(Self { field_map, arg_map })
    }

    fn get_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Value<'ctx, 'sco>> {
        if let Some(val) = self.get_input_val(builder, var)? {
            Ok(val)
        } else if let Some(val) = self.get_member_val(builder, var)? {
            Ok(val)
        } else {
            Err(anyhow!(
                "Could not find {var:?} in args or member definitions"
            ))
        }
    }

    fn get_input_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        match self.arg_map.get(var) {
            None => Ok(None),
            Some((arg_no, index)) => {
                let arg_val = builder.get_arg_value(*arg_no)?;
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

    fn get_member_val<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco>,
        var: &Variable,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        match self.field_map.get(var) {
            None => Ok(None),
            Some((member_name, index)) => {
                let self_val = builder.get_arg_value(0)?;
                let location = builder.unknown_location();
                match index {
                    None => {
                        // TODO: specify field?
                        let member_ty = builder.felt_type();
                        let member_val = builder.append_member_read(
                            location,
                            self_val,
                            member_ty,
                            &member_name,
                        )?;
                        Ok(Some(member_val))
                    }
                    Some(index) => {
                        let member_ty = builder.register_type();
                        let member_val = builder.append_member_read(
                            location,
                            self_val,
                            member_ty,
                            &member_name,
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
}
