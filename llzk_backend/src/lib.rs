use anyhow::anyhow;
use anyhow::Result;
use llzk::prelude::*;
use prover::cs::definitions::OpcodeFamilyCircuitState;
use prover::{cs::{cs::circuit::CircuitOutput, definitions::Variable}, field::{Field, PrimeField}};

use crate::builder::OpsBuilder;

pub mod builder;

/// This enum holds the possible representations for SSA values
pub enum SsaAddress<'ctx, 'val> {
    /// Represents a single variable that is neither an input or an output.
    /// It's encoded as a struct member of [`FeltType`].
    Intermediate(Value<'ctx, 'val>),
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
        let mut builder = OpsBuilder::new(constrain_fn);
        f(&mut builder)
    }
}

/// This enum holds information about extracted variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtractedVariable {
    /// A register value represented by a low and high limb.
    Register {low: Variable, high: Variable},
    /// A scalar field element
    Scalar(Variable)
}

impl ExtractedVariable {
    /// Create a new register
    pub fn register(reg: [Variable; 2]) -> Self {
        Self::Register { low: reg[0], high: reg[1] }
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
            ExtractedVariable::register(self.cycle_end_state.timestamp)
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
        let exec_state = &self.executor_machine_state.ok_or_else(|| anyhow!("executor_machine_state not initialized"))?;
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
        let exec_state = &self.executor_machine_state.ok_or_else(|| anyhow!("executor_machine_state not initialized"))?;
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
                let in_ram_reads = self.shuffle_ram_queries.iter().any(|&q| {
                    q.read_value[0] == *v || q.read_value[1] == *v
                });
                let in_io = io.iter().any(|x| x.contains(v));
                !in_io && !in_ram_reads
            })
            .map(|v| ExtractedVariable::Scalar(v))
            .collect::<Vec<_>>();

        intermediates.sort();
        Ok(intermediates)
    }
}
