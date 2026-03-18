//! Witness SSA extraction and lowering for LLZK `@compute`.
//!
//! The existing constraint lowering path only needs the finalized [`CircuitOutput`], but witness
//! lowering needs one extra layer of compiler metadata:
//! - the one-row compiler's variable-to-column mapping, so we know which SSA writes belong in the
//!   returned struct and which only touch transient memory columns; and
//! - the witness placer's SSA blocks, so we can replay the same evaluation order inside LLZK.
//!
//! The extraction flow implemented here mirrors the Rust witness generator closely:
//! 1. walk each SSA block in order;
//! 2. cache every subexpression by its SSA index;
//! 3. lower writes that target witness or scratch columns into struct member updates;
//! 4. canonicalize placeholder-based witness reads back to explicit LLZK inputs whenever the same
//!    logical machine-state value is already part of the `@compute` boundary; and
//! 5. lower remaing ROM and memory-subtree accesses as explicit runtime hooks.

use std::collections::BTreeMap;
use std::collections::HashMap;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use llzk::dialect::bool;
use llzk::dialect::felt;
use llzk::prelude::melior_dialects::arith;
use llzk::prelude::*;
use prover::common_constants;
use prover::cs::cs::placeholder::Placeholder;
use prover::cs::cs::witness_placer::graph_description::BoolNodeExpression;
use prover::cs::cs::witness_placer::graph_description::Expression;
use prover::cs::cs::witness_placer::graph_description::FieldNodeExpression;
use prover::cs::cs::witness_placer::graph_description::FixedWidthIntegerNodeExpression;
use prover::cs::cs::witness_placer::graph_description::RawExpression;
use prover::cs::definitions::ColumnAddress;
use prover::cs::definitions::LookupSetDescription;
use prover::cs::definitions::TableIndex;
use prover::cs::definitions::Variable;
use prover::cs::definitions::COMMON_TABLE_WIDTH;
use prover::cs::one_row_compiler::CompiledCircuitArtifact;
use prover::cs::tables::TableType;
use prover::field::PrimeField;

use crate::builder::ModuleEnv;
use crate::builder::OpsBuilder;
use crate::codegen::StructVars;
use crate::field::FieldInfo;

const U8_MODULUS: u64 = 1 << 8;
const U16_MODULUS: u64 = 1 << 16;

const READ_FROM_ROM_EXTERN: &str = "read_from_rom";
const READ_FROM_MEMORY_SUBTREE_EXTERN: &str = "read_from_memory_subtree";
const WRITE_TO_MEMORY_SUBTREE_EXTERN: &str = "write_to_memory_subtree";
const READ_ORACLE_FIELD_EXTERN: &str = "read_oracle_field";
const READ_ORACLE_BOOL_EXTERN: &str = "read_oracle_bool";
const READ_ORACLE_U8_EXTERN: &str = "read_oracle_u8";
const READ_ORACLE_U16_EXTERN: &str = "read_oracle_u16";
const READ_ORACLE_U32_EXTERN: &str = "read_oracle_u32";

/// An encoding for a witness placeholder (from [`Placeholder`]), but encoded for easy
/// emission to LLZK in a fixed format that is consistent across all placeholder types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodedOraclePlaceholder {
    kind: u64,
    arg0: u64,
    arg1: u64,
}

/// Convert a witness placeholder into the fixed-width format used by LLZK oracle hooks.
fn encode_oracle_placeholder(placeholder: Placeholder) -> EncodedOraclePlaceholder {
    use Placeholder::*;

    let kind = match placeholder {
        XregsInit => 0,
        XregsFin => 1,
        XregInit(_) => 2,
        XregFin(_) => 3,
        Instruction => 4,
        MemSlot => 5,
        PcInit => 6,
        PcFin => 7,
        StatusInit => 8,
        StatusFin => 9,
        IeInit => 10,
        IeFin => 11,
        IpInit => 12,
        IpFin => 13,
        TvecInit => 14,
        TvecFin => 15,
        ScratchInit => 16,
        ScratchFin => 17,
        EpcInit => 18,
        EpcFin => 19,
        CauseInit => 20,
        CauseFin => 21,
        TvalInit => 22,
        TvalFin => 23,
        ModeInit => 24,
        ModeFin => 25,
        MemorySaptInit => 26,
        MemorySaptFin => 27,
        ContinueExecutionInit => 28,
        ContinueExecutionFin => 29,
        ExternalOracle => 30,
        Trapped => 31,
        InvalidEncoding => 32,
        FirstRegMem => 33,
        SecondRegMem => 34,
        WriteRegMemReadWitness => 35,
        WriteRegMemWriteValue => 36,
        MemoryLoadOp => 37,
        WriteRdReadSetWitness => 38,
        ShuffleRamLazyInitAddressThis => 39,
        ShuffleRamLazyInitAddressNext => 40,
        ShuffleRamAddress(_) => 41,
        ShuffleRamReadTimestamp(_) => 42,
        ShuffleRamReadValue(_) => 43,
        ShuffleRamIsRegisterAccess(_) => 44,
        ShuffleRamWriteValue(_) => 45,
        ExecuteDelegation => 46,
        DelegationType => 47,
        DelegationABIOffset => 48,
        DelegationWriteTimestamp => 49,
        DelegationMemoryReadValue(_) => 50,
        DelegationMemoryReadTimestamp(_) => 51,
        DelegationMemoryWriteValue(_) => 52,
        DelegationRegisterReadValue(_) => 53,
        DelegationRegisterReadTimestamp(_) => 54,
        DelegationRegisterWriteValue(_) => 55,
        DelegationIndirectReadValue { .. } => 56,
        DelegationIndirectReadTimestamp { .. } => 57,
        DelegationIndirectWriteValue { .. } => 58,
        DelegationNondeterminismAccess(_) => 59,
        DelegationNondeterminismAccessNoSplits(_) => 60,
        ExecuteOpcodeFamilyCycle => 61,
        OpcodeFamilyCycleInitialTimestamp => 62,
        OpcodeFamilyCycleFinalTimestamp => 63,
        RS1Index => 64,
        RS2Index => 65,
        MemLoadAddress => 66,
        RDIndex => 67,
        RDIsZero => 68,
        DecodedImm => 69,
        DecodedFunct3 => 70,
        DecodedFunct7 => 71,
        DecodedExecutorFamilyMask => 72,
        LoadStoreRamValue => 73,
        MemStoreAddress => 74,
        DelegationIndirectAccessVariableOffset { .. } => 75,
    };

    let (arg0, arg1) = match placeholder {
        XregInit(idx) | XregFin(idx) => (idx as u64, 0),
        ShuffleRamAddress(access_idx)
        | ShuffleRamReadTimestamp(access_idx)
        | ShuffleRamReadValue(access_idx)
        | ShuffleRamIsRegisterAccess(access_idx)
        | ShuffleRamWriteValue(access_idx)
        | DelegationMemoryReadValue(access_idx)
        | DelegationMemoryReadTimestamp(access_idx)
        | DelegationMemoryWriteValue(access_idx)
        | DelegationNondeterminismAccess(access_idx)
        | DelegationNondeterminismAccessNoSplits(access_idx) => (access_idx as u64, 0),
        DelegationRegisterReadValue(register_index)
        | DelegationRegisterReadTimestamp(register_index)
        | DelegationRegisterWriteValue(register_index) => (register_index as u64, 0),
        DelegationIndirectReadValue {
            register_index,
            word_index,
        }
        | DelegationIndirectReadTimestamp {
            register_index,
            word_index,
        }
        | DelegationIndirectWriteValue {
            register_index,
            word_index,
        } => (register_index as u64, word_index as u64),
        DelegationIndirectAccessVariableOffset { variable_index } => (variable_index as u64, 0),
        _ => (0, 0),
    };

    EncodedOraclePlaceholder { kind, arg0, arg1 }
}

/// Bundles the metadata required to lower witness generation into LLZK `@compute`.
///
/// The compiled artifact tells us where every logical variable lives in the witness layout, the
/// SSA blocks preserve the witness placer's evaluation order and conditional write structure, and
/// the placeholder substitution map lets us recognize when legacy oracle-style SSA inputs are
/// actually aliases of explicit LLZK `@compute` arguments.
pub(crate) struct WitnessComputation<F: FieldInfo> {
    compiled: CompiledCircuitArtifact<F>,
    ssa: Vec<Vec<RawExpression<F>>>,
    substitutions: HashMap<(Placeholder, usize), Variable>,
}

impl<F: FieldInfo> WitnessComputation<F> {
    /// Create a new witness computation plan from the one-row compiler output, witness SSA, and
    /// placeholder substitution map.
    ///
    /// The substitution map is what lets LLZK avoid deriving the same machine-state input from two
    /// different sources. When a placeholder such as `PcInit` already names an explicit
    /// `@compute` argument, witness lowering reads the argument and only falls back to an oracle
    /// hook for placeholders that remain true runtime-only data.
    pub fn new(
        compiled: CompiledCircuitArtifact<F>,
        ssa: Vec<Vec<RawExpression<F>>>,
        substitutions: HashMap<(Placeholder, usize), Variable>,
    ) -> Self {
        Self {
            compiled,
            ssa,
            substitutions,
        }
    }

    /// Declare runtime hooks if they are needed by the emitted LLZK module.
    pub fn declare_runtime_externs<'ctx>(&self, env: &ModuleEnv<'ctx, F>) -> Result<()> {
        let maybe_declare =
            |name: &str, inputs: &[Type<'ctx>], results: &[Type<'ctx>]| -> Result<()> {
                if env.module_contains_call_to(name)? {
                    env.declare_private_extern_function(name, inputs, results)?;
                }
                Ok(())
            };

        maybe_declare(
            READ_FROM_ROM_EXTERN,
            &[env.felt_type()],
            &[env.felt_type(), env.felt_type()],
        )?;
        maybe_declare(
            READ_FROM_MEMORY_SUBTREE_EXTERN,
            &[env.index_type()],
            &[env.felt_type()],
        )?;
        maybe_declare(
            WRITE_TO_MEMORY_SUBTREE_EXTERN,
            &[env.index_type(), env.felt_type()],
            &[],
        )?;
        maybe_declare(
            READ_ORACLE_FIELD_EXTERN,
            &[
                env.felt_type(),
                env.felt_type(),
                env.felt_type(),
                env.felt_type(),
            ],
            &[env.felt_type()],
        )?;
        maybe_declare(
            READ_ORACLE_BOOL_EXTERN,
            &[env.felt_type(), env.felt_type(), env.felt_type()],
            &[env.bool_type()],
        )?;
        maybe_declare(
            READ_ORACLE_U8_EXTERN,
            &[env.felt_type(), env.felt_type(), env.felt_type()],
            &[env.felt_type()],
        )?;
        maybe_declare(
            READ_ORACLE_U16_EXTERN,
            &[env.felt_type(), env.felt_type(), env.felt_type()],
            &[env.felt_type()],
        )?;
        maybe_declare(
            READ_ORACLE_U32_EXTERN,
            &[env.felt_type(), env.felt_type(), env.felt_type()],
            &[env.felt_type(), env.felt_type()],
        )
    }

    /// Emit LLZK operations that reconstruct witness columns inside a struct `@compute` function.
    ///
    /// The lowering intentionally follows the SSA block structure produced by the witness placer so
    /// that every new helper can be reviewed against the existing Rust witness evaluator one block
    /// at a time.
    pub fn emit_compute<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<()> {
        let has_runtime_memory_reads = self.has_runtime_memory_reads(vars);
        let self_value = builder.get_compute_self_value()?;
        for block in &self.ssa {
            let mut lowering = ComputeLowering::new(
                builder,
                vars,
                self_value,
                &self.compiled.variable_mapping,
                &self.compiled.witness_layout.width_3_lookups,
                &self.substitutions,
                has_runtime_memory_reads,
                block,
            );
            block.emit_compute(&mut lowering)?;
        }
        Ok(())
    }

    /// Conservatively detect whether any emitted SSA path will read a compiled memory-subtree
    /// column through the LLZK runtime hook.
    ///
    /// This powers the strict seeding-write omission rule: we only drop boundary-to-memory mirror
    /// writes when the entire `@compute` body is otherwise independent of runtime memory state.
    /// The scan intentionally over-approximates. If it is unsure, it reports `true` and keeps the
    /// write.
    fn has_runtime_memory_reads(&self, vars: &StructVars<F>) -> bool {
        self.ssa.iter().flatten().any(|expr| {
            raw_expression_reads_runtime_memory(expr, &self.compiled.variable_mapping, vars)
        })
    }
}

#[derive(Clone, Copy)]
struct U32Parts<'ctx, 'sco> {
    low: Value<'ctx, 'sco>,
    high: Value<'ctx, 'sco>,
}

#[derive(Clone, Copy)]
enum IntegerValue<'ctx, 'sco> {
    U8(Value<'ctx, 'sco>),
    U16(Value<'ctx, 'sco>),
    U32(U32Parts<'ctx, 'sco>),
}

impl<'ctx, 'sco> IntegerValue<'ctx, 'sco> {
    fn bit_width(&self) -> u32 {
        match self {
            Self::U8(_) => 8,
            Self::U16(_) => 16,
            Self::U32(_) => 32,
        }
    }
}

#[derive(Clone, Copy)]
enum ComputedValue<'ctx, 'sco> {
    Field(Value<'ctx, 'sco>),
    Bool(Value<'ctx, 'sco>),
    Integer(IntegerValue<'ctx, 'sco>),
}

enum SsaSlot<'ctx, 'sco> {
    Value(ComputedValue<'ctx, 'sco>),
    Lookup(Vec<Value<'ctx, 'sco>>),
    Unit,
}

/// Trait implemented by SSA witness nodes that can emit LLZK IR inside a struct `@compute`
/// function.
trait EmitLlzkInCompute<'a, 'ctx: 'sco, 'sco, F: FieldInfo> {
    type Output;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output>;
}

impl<'a, 'ctx: 'sco, 'sco, F, T> EmitLlzkInCompute<'a, 'ctx, 'sco, F> for Vec<T>
where
    F: FieldInfo,
    T: EmitLlzkInCompute<'a, 'ctx, 'sco, F, Output = ()>,
{
    type Output = ();

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        self.iter().try_for_each(|expr| expr.emit_compute(lowering))
    }
}

/// Borrowed view of the SSA lookup forms that need dedicated lowering.
///
/// Keeping these variants separate from the rest of [`RawExpression`] mirrors the constraint-side
/// organization more closely: generic expression dispatch stays in `RawExpression`, while lookup
/// semantics live behind their own lowering implementation.
enum LookupInvocation {
    Perform {
        input_subexpr_idxes: Box<[usize]>,
        table_id_subexpr_idx: usize,
        num_outputs: usize,
        lookup_mapping_idx: usize,
    },
    MaybePerform {
        input_subexpr_idxes: Box<[usize]>,
        table_id_subexpr_idx: usize,
        mask_id_subexpr_idx: usize,
        num_outputs: usize,
    },
}

impl LookupInvocation {
    fn from_raw<F: PrimeField>(expr: &RawExpression<F>) -> Option<Self> {
        match expr {
            RawExpression::PerformLookup {
                input_subexpr_idxes,
                table_id_subexpr_idx,
                num_outputs,
                lookup_mapping_idx,
            } => Some(Self::Perform {
                input_subexpr_idxes: input_subexpr_idxes.clone(),
                table_id_subexpr_idx: *table_id_subexpr_idx,
                num_outputs: *num_outputs,
                lookup_mapping_idx: *lookup_mapping_idx,
            }),
            RawExpression::MaybePerformLookup {
                input_subexpr_idxes,
                table_id_subexpr_idx,
                mask_id_subexpr_idx,
                num_outputs,
            } => Some(Self::MaybePerform {
                input_subexpr_idxes: input_subexpr_idxes.clone(),
                table_id_subexpr_idx: *table_id_subexpr_idx,
                mask_id_subexpr_idx: *mask_id_subexpr_idx,
                num_outputs: *num_outputs,
            }),
            _ => None,
        }
    }

    fn input_subexpr_idxes(&self) -> &[usize] {
        match self {
            Self::Perform {
                input_subexpr_idxes,
                ..
            }
            | Self::MaybePerform {
                input_subexpr_idxes,
                ..
            } => input_subexpr_idxes,
        }
    }

    fn table_id_subexpr_idx(&self) -> usize {
        match self {
            Self::Perform {
                table_id_subexpr_idx,
                ..
            }
            | Self::MaybePerform {
                table_id_subexpr_idx,
                ..
            } => *table_id_subexpr_idx,
        }
    }

    fn num_outputs(&self) -> usize {
        match self {
            Self::Perform { num_outputs, .. } | Self::MaybePerform { num_outputs, .. } => {
                *num_outputs
            }
        }
    }

    fn lookup_mapping_idx(&self) -> Option<usize> {
        match self {
            Self::Perform {
                lookup_mapping_idx, ..
            } => Some(*lookup_mapping_idx),
            Self::MaybePerform { .. } => None,
        }
    }

    fn mask_id_subexpr_idx(&self) -> Option<usize> {
        match self {
            Self::Perform { .. } => None,
            Self::MaybePerform {
                mask_id_subexpr_idx,
                ..
            } => Some(*mask_id_subexpr_idx),
        }
    }
}

fn variable_uses_runtime_memory<F: FieldInfo>(
    variable: Variable,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    matches!(
        variable_mapping.get(&variable),
        Some(ColumnAddress::MemorySubtree(_))
    ) && !vars.is_compute_exposed(&variable)
}

fn expression_reads_runtime_memory<F: FieldInfo>(
    expr: &Expression<F>,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    match expr {
        Expression::Bool(expr) => {
            bool_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        Expression::Field(expr) => {
            field_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        Expression::U8(expr) | Expression::U16(expr) | Expression::U32(expr) => {
            integer_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
    }
}

fn field_expression_reads_runtime_memory<F: FieldInfo>(
    expr: &FieldNodeExpression<F>,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    match expr {
        FieldNodeExpression::Place(variable) => {
            variable_uses_runtime_memory(*variable, variable_mapping, vars)
        }
        FieldNodeExpression::SubExpression(..)
        | FieldNodeExpression::Constant(..)
        | FieldNodeExpression::OracleValue { .. }
        | FieldNodeExpression::LookupOutput { .. }
        | FieldNodeExpression::MaybeLookupOutput { .. } => false,
        FieldNodeExpression::FromInteger(expr) => {
            integer_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        FieldNodeExpression::FromMask(expr) => {
            bool_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        FieldNodeExpression::Add { lhs, rhs }
        | FieldNodeExpression::Sub { lhs, rhs }
        | FieldNodeExpression::Mul { lhs, rhs } => {
            field_expression_reads_runtime_memory(lhs, variable_mapping, vars)
                || field_expression_reads_runtime_memory(rhs, variable_mapping, vars)
        }
        FieldNodeExpression::AddProduct {
            additive_term,
            mul_0,
            mul_1,
        } => {
            field_expression_reads_runtime_memory(additive_term, variable_mapping, vars)
                || field_expression_reads_runtime_memory(mul_0, variable_mapping, vars)
                || field_expression_reads_runtime_memory(mul_1, variable_mapping, vars)
        }
        FieldNodeExpression::Select {
            selector,
            if_true,
            if_false,
        } => {
            bool_expression_reads_runtime_memory(selector, variable_mapping, vars)
                || field_expression_reads_runtime_memory(if_true, variable_mapping, vars)
                || field_expression_reads_runtime_memory(if_false, variable_mapping, vars)
        }
        FieldNodeExpression::InverseUnchecked(expr) | FieldNodeExpression::InverseOrZero(expr) => {
            field_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
    }
}

fn bool_expression_reads_runtime_memory<F: FieldInfo>(
    expr: &BoolNodeExpression<F>,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    match expr {
        BoolNodeExpression::Place(variable) => {
            variable_uses_runtime_memory(*variable, variable_mapping, vars)
        }
        BoolNodeExpression::SubExpression(..)
        | BoolNodeExpression::Constant(..)
        | BoolNodeExpression::OracleValue { .. } => false,
        BoolNodeExpression::FromGenericInteger(expr) => {
            integer_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        BoolNodeExpression::FromGenericIntegerEquality { lhs, rhs }
        | BoolNodeExpression::FromGenericIntegerCarry { lhs, rhs }
        | BoolNodeExpression::FromGenericIntegerBorrow { lhs, rhs } => {
            integer_expression_reads_runtime_memory(lhs, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(rhs, variable_mapping, vars)
        }
        BoolNodeExpression::FromField(expr) => {
            field_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        BoolNodeExpression::FromFieldEquality { lhs, rhs } => {
            field_expression_reads_runtime_memory(lhs, variable_mapping, vars)
                || field_expression_reads_runtime_memory(rhs, variable_mapping, vars)
        }
        BoolNodeExpression::And { lhs, rhs } | BoolNodeExpression::Or { lhs, rhs } => {
            bool_expression_reads_runtime_memory(lhs, variable_mapping, vars)
                || bool_expression_reads_runtime_memory(rhs, variable_mapping, vars)
        }
        BoolNodeExpression::Select {
            selector,
            if_true,
            if_false,
        } => {
            bool_expression_reads_runtime_memory(selector, variable_mapping, vars)
                || bool_expression_reads_runtime_memory(if_true, variable_mapping, vars)
                || bool_expression_reads_runtime_memory(if_false, variable_mapping, vars)
        }
        BoolNodeExpression::Negate(expr) => {
            bool_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
    }
}

fn integer_expression_reads_runtime_memory<F: FieldInfo>(
    expr: &FixedWidthIntegerNodeExpression<F>,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    match expr {
        FixedWidthIntegerNodeExpression::U8Place(variable)
        | FixedWidthIntegerNodeExpression::U16Place(variable) => {
            variable_uses_runtime_memory(*variable, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::U8SubExpression(..)
        | FixedWidthIntegerNodeExpression::U16SubExpression(..)
        | FixedWidthIntegerNodeExpression::U32SubExpression(..)
        | FixedWidthIntegerNodeExpression::U32OracleValue { .. }
        | FixedWidthIntegerNodeExpression::U16OracleValue { .. }
        | FixedWidthIntegerNodeExpression::U8OracleValue { .. }
        | FixedWidthIntegerNodeExpression::ConstantU8(..)
        | FixedWidthIntegerNodeExpression::ConstantU16(..)
        | FixedWidthIntegerNodeExpression::ConstantU32(..) => false,
        FixedWidthIntegerNodeExpression::U32FromMask(expr) => {
            bool_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::U32FromField(expr) => {
            field_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::WidenFromU8(expr)
        | FixedWidthIntegerNodeExpression::WidenFromU16(expr)
        | FixedWidthIntegerNodeExpression::TruncateFromU16(expr)
        | FixedWidthIntegerNodeExpression::TruncateFromU32(expr)
        | FixedWidthIntegerNodeExpression::I32FromU32(expr)
        | FixedWidthIntegerNodeExpression::U32FromI32(expr)
        | FixedWidthIntegerNodeExpression::BinaryNot(expr) => {
            integer_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::Select {
            selector,
            if_true,
            if_false,
        } => {
            bool_expression_reads_runtime_memory(selector, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(if_true, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(if_false, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::WrappingAdd { lhs, rhs }
        | FixedWidthIntegerNodeExpression::WrappingSub { lhs, rhs }
        | FixedWidthIntegerNodeExpression::BinaryAnd { lhs, rhs }
        | FixedWidthIntegerNodeExpression::BinaryOr { lhs, rhs }
        | FixedWidthIntegerNodeExpression::BinaryXor { lhs, rhs }
        | FixedWidthIntegerNodeExpression::MulLow { lhs, rhs }
        | FixedWidthIntegerNodeExpression::MulHigh { lhs, rhs }
        | FixedWidthIntegerNodeExpression::DivAssumeNonzero { lhs, rhs }
        | FixedWidthIntegerNodeExpression::RemAssumeNonzero { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedDivAssumeNonzeroNoOverflowBits { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedRemAssumeNonzeroNoOverflowBits { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedMulLowBits { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedMulHighBits { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedByUnsignedMulLowBits { lhs, rhs }
        | FixedWidthIntegerNodeExpression::SignedByUnsignedMulHighBits { lhs, rhs } => {
            integer_expression_reads_runtime_memory(lhs, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(rhs, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::WrappingShl { lhs, .. }
        | FixedWidthIntegerNodeExpression::WrappingShr { lhs, .. } => {
            integer_expression_reads_runtime_memory(lhs, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::LowestBits { value, .. } => {
            integer_expression_reads_runtime_memory(value, variable_mapping, vars)
        }
        FixedWidthIntegerNodeExpression::AddProduct {
            additive_term,
            mul_0,
            mul_1,
        } => {
            integer_expression_reads_runtime_memory(additive_term, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(mul_0, variable_mapping, vars)
                || integer_expression_reads_runtime_memory(mul_1, variable_mapping, vars)
        }
    }
}

fn raw_expression_reads_runtime_memory<F: FieldInfo>(
    expr: &RawExpression<F>,
    variable_mapping: &BTreeMap<Variable, ColumnAddress>,
    vars: &StructVars<F>,
) -> bool {
    match expr {
        RawExpression::Bool(expr) => {
            bool_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        RawExpression::Field(expr) => {
            field_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        RawExpression::Integer(expr) => {
            integer_expression_reads_runtime_memory(expr, variable_mapping, vars)
        }
        RawExpression::AccessLookup { .. }
        | RawExpression::PerformLookup { .. }
        | RawExpression::MaybePerformLookup { .. } => false,
        RawExpression::WriteVariable {
            into_variable,
            source_subexpr,
            condition_subexpr_idx,
        } => {
            expression_reads_runtime_memory(source_subexpr, variable_mapping, vars)
                || (condition_subexpr_idx.is_some()
                    && matches!(
                        variable_mapping.get(into_variable),
                        Some(ColumnAddress::MemorySubtree(_))
                    )
                    && !vars.has_member(into_variable))
        }
    }
}

/// Lowers one SSA block into LLZK ops while keeping a slot-per-subexpression cache.
///
/// This mirrors the existing Rust witness generator's "one raw expression, one SSA slot" model so
/// that indices coming from `SubExpression(..)` nodes continue to line up exactly.
struct ComputeLowering<'a, 'ctx: 'sco, 'sco, F: FieldInfo> {
    builder: &'a OpsBuilder<'ctx, 'sco, F>,
    vars: &'a StructVars<F>,
    self_value: Value<'ctx, 'sco>,
    variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
    lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
    substitutions: &'a HashMap<(Placeholder, usize), Variable>,
    has_runtime_memory_reads: bool,
    block: &'a [RawExpression<F>],
    slots: Vec<SsaSlot<'ctx, 'sco>>,
    slot_input_origins: Vec<Option<Variable>>,
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> ComputeLowering<'a, 'ctx, 'sco, F> {
    /// Create a fresh lowering state for one SSA block.
    fn new(
        builder: &'a OpsBuilder<'ctx, 'sco, F>,
        vars: &'a StructVars<F>,
        self_value: Value<'ctx, 'sco>,
        variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
        lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
        substitutions: &'a HashMap<(Placeholder, usize), Variable>,
        has_runtime_memory_reads: bool,
        block: &'a [RawExpression<F>],
    ) -> Self {
        Self {
            builder,
            vars,
            self_value,
            variable_mapping,
            lookup_sets,
            substitutions,
            has_runtime_memory_reads,
            block,
            slots: Vec::new(),
            slot_input_origins: Vec::new(),
        }
    }

    /// Append one SSA slot to the block-local cache.
    fn push_slot(&mut self, slot: SsaSlot<'ctx, 'sco>, input_origin: Option<Variable>) {
        self.slots.push(slot);
        self.slot_input_origins.push(input_origin);
    }

    /// Materialize a write-back either into the returned witness struct or into the external
    /// memory runtime.
    ///
    /// The one-row compiler may place some boundary variables in the compiled `MemorySubtree`
    /// even though LLZK also exposes them as public outputs or internal struct members. When that
    /// happens, `@compute` should update the struct member first so the returned LLZK value agrees
    /// with what `@constrain` sees. Only variables that have no struct representation are routed
    /// through the generic memory runtime hook.
    fn lower_write(
        &mut self,
        into_variable: &Variable,
        source_subexpr: &Expression<F>,
        condition_subexpr_idx: Option<usize>,
    ) -> Result<()> {
        if self.vars.has_member(into_variable) {
            let mut value = self.expression_to_store_value(source_subexpr)?;
            if let Some(condition_idx) = condition_subexpr_idx {
                let condition = self.slot_as_bool(condition_idx)?;
                let existing =
                    self.vars
                        .get_compute_val(self.builder, self.self_value, into_variable)?;
                value = self.select_value(condition, value, existing)?;
            }

            return self.vars.assign_compute_member(
                self.builder,
                self.self_value,
                into_variable,
                value,
            );
        }

        if self.should_omit_seed_memory_write(
            into_variable,
            source_subexpr,
            condition_subexpr_idx,
        )? {
            return Ok(());
        }

        match self.variable_mapping[into_variable] {
            ColumnAddress::SetupSubtree(..) => {
                bail!("setup columns are read-only during witness lowering")
            }
            ColumnAddress::MemorySubtree(offset) => {
                let mut value = self.expression_to_store_value(source_subexpr)?;
                if let Some(condition_idx) = condition_subexpr_idx {
                    let condition = self.slot_as_bool(condition_idx)?;
                    let existing = self.read_memory_subtree(offset)?;
                    value = self.select_value(condition, value, existing)?;
                }
                return self.write_memory_subtree(offset, value);
            }
            ColumnAddress::WitnessSubtree(..) | ColumnAddress::OptimizedOut(..) => {
                // These placements do not need a separate runtime hook. Witness-subtree values
                // are materialized through the returned LLZK struct, and `OptimizedOut` means the
                // one-row compiler eliminated any distinct backing column to update. So we fall
                // through to the struct-member write below instead of bailing here; if the
                // variable is not actually exposed on the LLZK boundary, that write will error and
                // surface a real mapping bug.
            }
        }

        let mut value = self.expression_to_store_value(source_subexpr)?;
        if let Some(condition_idx) = condition_subexpr_idx {
            let condition = self.slot_as_bool(condition_idx)?;
            let existing =
                self.vars
                    .get_compute_val(self.builder, self.self_value, into_variable)?;
            value = self.select_value(condition, value, existing)?;
        }

        self.vars
            .assign_compute_member(self.builder, self.self_value, into_variable, value)
    }

    /// Return `true` when this write is a pure boundary-to-memory seeding copy that can be omitted
    /// without changing the returned LLZK value.
    ///
    /// The rule is intentionally strict:
    /// - the target must live in the compiled `MemorySubtree`,
    /// - the write must be unconditional,
    /// - the source expression must be provably just an explicit `@compute` input value, and
    /// - the whole emitted `@compute` must never read memory-subtree state back.
    ///
    /// If any of those checks fail, we keep the runtime write.
    fn should_omit_seed_memory_write(
        &self,
        into_variable: &Variable,
        source_subexpr: &Expression<F>,
        condition_subexpr_idx: Option<usize>,
    ) -> Result<bool> {
        if self.has_runtime_memory_reads || condition_subexpr_idx.is_some() {
            return Ok(false);
        }
        let Some(ColumnAddress::MemorySubtree(_)) =
            self.variable_mapping.get(into_variable).copied()
        else {
            return Ok(false);
        };

        Ok(self
            .strict_input_origin_for_expression(source_subexpr)
            .is_some())
    }

    /// Convert a lowered expression into the felt encoding stored in the LLZK witness struct.
    fn computed_value_to_store(
        &self,
        value: ComputedValue<'ctx, 'sco>,
    ) -> Result<Value<'ctx, 'sco>> {
        match value {
            ComputedValue::Bool(value) => self.bool_to_field(value),
            ComputedValue::Field(value) => Ok(value),
            ComputedValue::Integer(value) => self.integer_to_field(value),
        }
    }

    /// Convert a typed SSA reference into the felt-encoded value that the witness struct stores.
    fn expression_to_store_value(&mut self, expr: &Expression<F>) -> Result<Value<'ctx, 'sco>> {
        let value = expr.emit_compute(self)?;
        self.computed_value_to_store(value)
    }

    /// Read a logical circuit variable as the felt value currently visible to `@compute`.
    ///
    /// Variables mapped into the witness struct are read from inputs or members. Variables placed
    /// in the compiled `MemorySubtree` are read through the external runtime hook because they are
    /// part of mutable execution-memory state rather than the struct returned by `@compute`.
    fn read_variable(&self, variable: Variable) -> Result<Value<'ctx, 'sco>> {
        if let Some(value) =
            self.vars
                .try_get_compute_val(self.builder, self.self_value, &variable)?
        {
            return Ok(value);
        }

        match self.variable_mapping.get(&variable).copied() {
            Some(ColumnAddress::MemorySubtree(offset)) => self.read_memory_subtree(offset),
            other => Err(anyhow!(
                "variable {variable:?} is not exposed to @compute (column {other:?})"
            )),
        }
    }

    /// Read a compiled memory-subtree column through the LLZK runtime hook.
    fn read_memory_subtree(&self, offset: usize) -> Result<Value<'ctx, 'sco>> {
        let location = self.builder.unknown_location();
        let offset = self
            .builder
            .get_constant_from_start(self.builder.index_type(), offset as u64)?;
        self.builder.append_call_with_result(
            location,
            READ_FROM_MEMORY_SUBTREE_EXTERN,
            &[offset],
            self.builder.felt_type(),
        )
    }

    /// Write a compiled memory-subtree column through the LLZK runtime hook.
    fn write_memory_subtree(&self, offset: usize, value: Value<'ctx, 'sco>) -> Result<()> {
        let location = self.builder.unknown_location();
        let offset = self
            .builder
            .get_constant_from_start(self.builder.index_type(), offset as u64)?;
        self.builder.append_call_no_results(
            location,
            WRITE_TO_MEMORY_SUBTREE_EXTERN,
            &[offset, value],
        )
    }

    /// Try to resolve one placeholder limb through the explicit `@compute` inputs.
    ///
    /// Many executor-state placeholders in the legacy witness path are just alternate names for
    /// the same logical values that LLZK already models as `@compute` arguments. Reading those
    /// arguments directly keeps the dataflow consistent between `@compute` and `@constrain`.
    ///
    /// We only redirect placeholders to inputs here, not to struct members. That distinction is
    /// important for output placeholders such as `PcFin`: they may be exposed as LLZK members, but
    /// reading the member before it is assigned would be incorrect. Those cases still fall back to
    /// the runtime oracle path until the SSA itself computes the output value.
    ///
    /// In the current circuit set this means placeholders like `PcInit`, decoded instruction
    /// fields, `ExecuteOpcodeFamilyCycle`, and the legacy read-side shuffle aliases such as
    /// `FirstRegMem` come from `%argN`, while non-boundary values such as shuffle timestamps still
    /// come from `@read_oracle_*`.
    fn try_read_placeholder_input_limb(
        &self,
        placeholder: Placeholder,
        subindex: usize,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        let Some(variable) = self.substitutions.get(&(placeholder, subindex)).copied() else {
            return Ok(None);
        };

        self.vars.try_get_compute_input_val(self.builder, &variable)
    }

    /// Try to read a field placeholder from the existing LLZK inputs before using an oracle hook.
    fn try_read_field_placeholder_input(
        &self,
        placeholder: Placeholder,
        subindex: usize,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        self.try_read_placeholder_input_limb(placeholder, subindex)
    }

    /// Try to read a boolean placeholder from the existing LLZK inputs before using an oracle
    /// hook.
    fn try_read_bool_placeholder_input(
        &self,
        placeholder: Placeholder,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        self.try_read_placeholder_input_limb(placeholder, 0)?
            .map(|value| self.field_is_nonzero(value))
            .transpose()
    }

    /// Try to read an 8-bit placeholder from the existing LLZK inputs before using an oracle
    /// hook.
    fn try_read_u8_placeholder_input(
        &self,
        placeholder: Placeholder,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        self.try_read_placeholder_input_limb(placeholder, 0)
    }

    /// Try to read a 16-bit placeholder from the existing LLZK inputs before using an oracle
    /// hook.
    fn try_read_u16_placeholder_input(
        &self,
        placeholder: Placeholder,
    ) -> Result<Option<Value<'ctx, 'sco>>> {
        self.try_read_placeholder_input_limb(placeholder, 0)
    }

    /// Try to read a 32-bit placeholder from the existing LLZK inputs before using an oracle
    /// hook.
    fn try_read_u32_placeholder_input(
        &self,
        placeholder: Placeholder,
    ) -> Result<Option<U32Parts<'ctx, 'sco>>> {
        let Some(low_var) = self.substitutions.get(&(placeholder, 0)).copied() else {
            return Ok(None);
        };
        let Some(high_var) = self.substitutions.get(&(placeholder, 1)).copied() else {
            return Err(anyhow!(
                "placeholder {placeholder:?} is missing its high limb substitution"
            ));
        };

        let low = self
            .vars
            .try_get_compute_input_val(self.builder, &low_var)?;
        let high = self
            .vars
            .try_get_compute_input_val(self.builder, &high_var)?;
        match (low, high) {
            (Some(low), Some(high)) => Ok(Some(U32Parts { low, high })),
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => Err(anyhow!(
                "placeholder {placeholder:?} is only partially exposed as a @compute input"
            )),
        }
    }

    /// Return the LLZK type used for oracle hook metadata arguments.
    ///
    /// The backend keeps the placeholder encoding stable at the Rust level, but the emitted ABI
    /// still needs to use LLZK value types. `function.def` rejects plain `i64` arguments, so we
    /// materialize the `(kind, arg0, arg1, subindex)` metadata as felt constants instead.
    fn oracle_abi_type(&self) -> Type<'ctx> {
        self.builder.felt_type()
    }

    /// Materialize one metadata constant for an oracle hook call.
    fn oracle_abi_constant(&self, value: u64) -> Result<Value<'ctx, 'sco>> {
        self.builder
            .get_constant_from_start(self.oracle_abi_type(), value)
    }

    /// Convert a [`Placeholder`] into the three metadata arguments passed to oracle hooks.
    fn oracle_placeholder_args(&self, placeholder: Placeholder) -> Result<[Value<'ctx, 'sco>; 3]> {
        let encoded = encode_oracle_placeholder(placeholder);
        Ok([
            self.oracle_abi_constant(encoded.kind)?,
            self.oracle_abi_constant(encoded.arg0)?,
            self.oracle_abi_constant(encoded.arg1)?,
        ])
    }

    /// Read a field-valued oracle placeholder through the LLZK runtime hook.
    fn read_field_oracle(
        &self,
        placeholder: Placeholder,
        subindex: usize,
    ) -> Result<Value<'ctx, 'sco>> {
        if let Some(value) = self.try_read_field_placeholder_input(placeholder, subindex)? {
            return Ok(value);
        }

        let location = self.builder.unknown_location();
        let [kind, arg0, arg1] = self.oracle_placeholder_args(placeholder)?;
        let subindex = self.oracle_abi_constant(subindex as u64)?;
        self.builder.append_call_with_result(
            location,
            READ_ORACLE_FIELD_EXTERN,
            &[kind, arg0, arg1, subindex],
            self.builder.felt_type(),
        )
    }

    /// Read a boolean oracle placeholder through the LLZK runtime hook.
    fn read_bool_oracle(&self, placeholder: Placeholder) -> Result<Value<'ctx, 'sco>> {
        if let Some(value) = self.try_read_bool_placeholder_input(placeholder)? {
            return Ok(value);
        }

        let location = self.builder.unknown_location();
        let [kind, arg0, arg1] = self.oracle_placeholder_args(placeholder)?;
        self.builder.append_call_with_result(
            location,
            READ_ORACLE_BOOL_EXTERN,
            &[kind, arg0, arg1],
            self.builder.bool_type(),
        )
    }

    /// Read an 8-bit oracle placeholder through the LLZK runtime hook.
    fn read_u8_oracle(&self, placeholder: Placeholder) -> Result<Value<'ctx, 'sco>> {
        if let Some(value) = self.try_read_u8_placeholder_input(placeholder)? {
            return Ok(value);
        }

        let location = self.builder.unknown_location();
        let [kind, arg0, arg1] = self.oracle_placeholder_args(placeholder)?;
        self.builder.append_call_with_result(
            location,
            READ_ORACLE_U8_EXTERN,
            &[kind, arg0, arg1],
            self.builder.felt_type(),
        )
    }

    /// Read a 16-bit oracle placeholder through the LLZK runtime hook.
    fn read_u16_oracle(&self, placeholder: Placeholder) -> Result<Value<'ctx, 'sco>> {
        if let Some(value) = self.try_read_u16_placeholder_input(placeholder)? {
            return Ok(value);
        }

        let location = self.builder.unknown_location();
        let [kind, arg0, arg1] = self.oracle_placeholder_args(placeholder)?;
        self.builder.append_call_with_result(
            location,
            READ_ORACLE_U16_EXTERN,
            &[kind, arg0, arg1],
            self.builder.felt_type(),
        )
    }

    /// Read a 32-bit oracle placeholder through the LLZK runtime hook.
    fn read_u32_oracle(&self, placeholder: Placeholder) -> Result<U32Parts<'ctx, 'sco>> {
        if let Some(value) = self.try_read_u32_placeholder_input(placeholder)? {
            return Ok(value);
        }

        let location = self.builder.unknown_location();
        let [low, high] = {
            let [kind, arg0, arg1] = self.oracle_placeholder_args(placeholder)?;
            self.builder.append_call::<2>(
                location,
                READ_ORACLE_U32_EXTERN,
                &[kind, arg0, arg1],
                &[self.builder.felt_type(), self.builder.felt_type()],
            )?
        };
        Ok(U32Parts { low, high })
    }

    /// Convert an `i1` condition into the felt encoding used by witness columns.
    fn bool_to_field(&self, value: Value<'ctx, 'sco>) -> Result<Value<'ctx, 'sco>> {
        self.select_value(
            value,
            self.builder.get_felt_constant_from_start(1)?,
            self.builder.get_felt_constant_from_start(0)?,
        )
    }

    /// Return an `i1` indicating whether the felt value is non-zero.
    fn field_is_nonzero(&self, value: Value<'ctx, 'sco>) -> Result<Value<'ctx, 'sco>> {
        self.builder.append_op_with_result(bool::ne(
            self.builder.unknown_location(),
            value,
            self.builder.get_felt_constant_from_start(0)?,
        )?)
    }

    /// Emit a generic `arith.select`, which works for both LLZK felt values and builtin integer
    /// values such as `i1`.
    fn select_value(
        &self,
        condition: Value<'ctx, 'sco>,
        if_true: Value<'ctx, 'sco>,
        if_false: Value<'ctx, 'sco>,
    ) -> Result<Value<'ctx, 'sco>> {
        self.builder.append_op_with_result(arith::select(
            condition,
            if_true,
            if_false,
            self.builder.unknown_location(),
        ))
    }

    /// Convert a 32-bit limb pair back into the field encoding used by witness columns.
    fn u32_to_field(&self, value: U32Parts<'ctx, 'sco>) -> Result<Value<'ctx, 'sco>> {
        let high_scaled = self.builder.append_op_with_result(felt::mul(
            self.builder.unknown_location(),
            value.high,
            self.builder.get_felt_constant_from_start(U16_MODULUS)?,
        )?)?;
        self.builder.append_op_with_result(felt::add(
            self.builder.unknown_location(),
            value.low,
            high_scaled,
        )?)
    }

    /// Convert an integer witness value into the felt encoding stored in struct members.
    fn integer_to_field(&self, value: IntegerValue<'ctx, 'sco>) -> Result<Value<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) | IntegerValue::U16(value) => Ok(value),
            IntegerValue::U32(value) => self.u32_to_field(value),
        }
    }

    /// Decompose a felt value into a 32-bit pair of 16-bit limbs.
    fn field_to_u32(&self, value: Value<'ctx, 'sco>) -> Result<U32Parts<'ctx, 'sco>> {
        let low = self.lowest_bits_felt(value, 16)?;
        let high = self.builder.append_op_with_result(felt::uintdiv(
            self.builder.unknown_location(),
            value,
            self.builder.get_felt_constant_from_start(U16_MODULUS)?,
        )?)?;
        Ok(U32Parts { low, high })
    }

    /// Build a `u32` constant as two 16-bit limbs.
    fn u32_constant(&self, value: u32) -> Result<U32Parts<'ctx, 'sco>> {
        Ok(U32Parts {
            low: self
                .builder
                .get_felt_constant_from_start(u64::from(value & 0xffff))?,
            high: self
                .builder
                .get_felt_constant_from_start(u64::from(value >> 16))?,
        })
    }

    /// Reduce a felt value modulo `2^bits`.
    fn lowest_bits_felt(&self, value: Value<'ctx, 'sco>, bits: u32) -> Result<Value<'ctx, 'sco>> {
        if bits == 0 {
            return self.builder.get_felt_constant_from_start(0);
        }
        let modulus = 1u64 << bits;
        self.builder.append_op_with_result(felt::umod(
            self.builder.unknown_location(),
            value,
            self.builder.get_felt_constant_from_start(modulus)?,
        )?)
    }

    /// Compute `lhs == rhs` over witness integers.
    fn integer_equal(
        &self,
        lhs: IntegerValue<'ctx, 'sco>,
        rhs: IntegerValue<'ctx, 'sco>,
    ) -> Result<Value<'ctx, 'sco>> {
        match (lhs, rhs) {
            (IntegerValue::U8(lhs), IntegerValue::U8(rhs))
            | (IntegerValue::U16(lhs), IntegerValue::U16(rhs)) => self
                .builder
                .append_op_with_result(bool::eq(self.builder.unknown_location(), lhs, rhs)?),
            (IntegerValue::U32(lhs), IntegerValue::U32(rhs)) => {
                let low_eq = self.builder.append_op_with_result(bool::eq(
                    self.builder.unknown_location(),
                    lhs.low,
                    rhs.low,
                )?)?;
                let high_eq = self.builder.append_op_with_result(bool::eq(
                    self.builder.unknown_location(),
                    lhs.high,
                    rhs.high,
                )?)?;
                self.builder.append_op_with_result(bool::and(
                    self.builder.unknown_location(),
                    low_eq,
                    high_eq,
                )?)
            }
            _ => bail!("integer equality requires operands of the same width"),
        }
    }

    /// Compute whether an integer witness value is non-zero.
    fn integer_is_nonzero(&self, value: IntegerValue<'ctx, 'sco>) -> Result<Value<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) | IntegerValue::U16(value) => self.field_is_nonzero(value),
            IntegerValue::U32(value) => {
                let low_nonzero = self.field_is_nonzero(value.low)?;
                let high_nonzero = self.field_is_nonzero(value.high)?;
                self.builder.append_op_with_result(bool::or(
                    self.builder.unknown_location(),
                    low_nonzero,
                    high_nonzero,
                )?)
            }
        }
    }

    /// Compute a wrapping add together with its carry/overflow bit.
    fn overflowing_add(
        &self,
        lhs: IntegerValue<'ctx, 'sco>,
        rhs: IntegerValue<'ctx, 'sco>,
    ) -> Result<(IntegerValue<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        match (lhs, rhs) {
            (IntegerValue::U8(lhs), IntegerValue::U8(rhs)) => {
                let (sum, carry) = self.add_small(lhs, rhs, 8)?;
                Ok((IntegerValue::U8(sum), carry))
            }
            (IntegerValue::U16(lhs), IntegerValue::U16(rhs)) => {
                let (sum, carry) = self.add_small(lhs, rhs, 16)?;
                Ok((IntegerValue::U16(sum), carry))
            }
            (IntegerValue::U32(lhs), IntegerValue::U32(rhs)) => {
                let (sum, carry) = self.add_u32(lhs, rhs)?;
                Ok((IntegerValue::U32(sum), carry))
            }
            _ => bail!("wrapping add requires operands of the same width"),
        }
    }

    /// Compute a wrapping subtraction together with its borrow bit.
    fn overflowing_sub(
        &self,
        lhs: IntegerValue<'ctx, 'sco>,
        rhs: IntegerValue<'ctx, 'sco>,
    ) -> Result<(IntegerValue<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        match (lhs, rhs) {
            (IntegerValue::U8(lhs), IntegerValue::U8(rhs)) => {
                let (diff, borrow) = self.sub_small(lhs, rhs, 8)?;
                Ok((IntegerValue::U8(diff), borrow))
            }
            (IntegerValue::U16(lhs), IntegerValue::U16(rhs)) => {
                let (diff, borrow) = self.sub_small(lhs, rhs, 16)?;
                Ok((IntegerValue::U16(diff), borrow))
            }
            (IntegerValue::U32(lhs), IntegerValue::U32(rhs)) => {
                let (diff, borrow) = self.sub_u32(lhs, rhs)?;
                Ok((IntegerValue::U32(diff), borrow))
            }
            _ => bail!("wrapping sub requires operands of the same width"),
        }
    }

    /// Add two `u8`/`u16` values in the field and recover the carry with a bound comparison.
    fn add_small(
        &self,
        lhs: Value<'ctx, 'sco>,
        rhs: Value<'ctx, 'sco>,
        width: u32,
    ) -> Result<(Value<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        let sum = self.builder.append_op_with_result(felt::add(
            self.builder.unknown_location(),
            lhs,
            rhs,
        )?)?;
        let modulus = 1u64 << width;
        let carry = self.builder.append_op_with_result(bool::ge(
            self.builder.unknown_location(),
            sum,
            self.builder.get_felt_constant_from_start(modulus)?,
        )?)?;
        let wrapped = self.lowest_bits_felt(sum, width)?;
        Ok((wrapped, carry))
    }

    /// Subtract two `u8`/`u16` values while preserving the borrow bit.
    fn sub_small(
        &self,
        lhs: Value<'ctx, 'sco>,
        rhs: Value<'ctx, 'sco>,
        width: u32,
    ) -> Result<(Value<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        let borrow = self.builder.append_op_with_result(bool::lt(
            self.builder.unknown_location(),
            lhs,
            rhs,
        )?)?;
        let borrow_case = self.builder.append_op_with_result(felt::sub(
            self.builder.unknown_location(),
            self.builder.append_op_with_result(felt::add(
                self.builder.unknown_location(),
                lhs,
                self.builder.get_felt_constant_from_start(1u64 << width)?,
            )?)?,
            rhs,
        )?)?;
        let direct_case = self.builder.append_op_with_result(felt::sub(
            self.builder.unknown_location(),
            lhs,
            rhs,
        )?)?;
        Ok((self.select_value(borrow, borrow_case, direct_case)?, borrow))
    }

    /// Add two 32-bit limb pairs and recover the carry bit from the high limb.
    fn add_u32(
        &self,
        lhs: U32Parts<'ctx, 'sco>,
        rhs: U32Parts<'ctx, 'sco>,
    ) -> Result<(U32Parts<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        let (low, low_carry) = self.add_small(lhs.low, rhs.low, 16)?;
        let carry_felt = self.bool_to_field(low_carry)?;
        let high_rhs = self.builder.append_op_with_result(felt::add(
            self.builder.unknown_location(),
            rhs.high,
            carry_felt,
        )?)?;
        let (high, high_carry) = self.add_small(lhs.high, high_rhs, 16)?;
        Ok((U32Parts { low, high }, high_carry))
    }

    /// Subtract two 32-bit limb pairs and recover the final borrow bit.
    fn sub_u32(
        &self,
        lhs: U32Parts<'ctx, 'sco>,
        rhs: U32Parts<'ctx, 'sco>,
    ) -> Result<(U32Parts<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        let (low, low_borrow) = self.sub_small(lhs.low, rhs.low, 16)?;
        let borrow_felt = self.bool_to_field(low_borrow)?;
        let high_rhs = self.builder.append_op_with_result(felt::add(
            self.builder.unknown_location(),
            rhs.high,
            borrow_felt,
        )?)?;
        let (high, high_borrow) = self.sub_small(lhs.high, high_rhs, 16)?;
        Ok((U32Parts { low, high }, high_borrow))
    }

    /// Apply a logical right shift to an integer witness value.
    fn shift_right(
        &self,
        value: IntegerValue<'ctx, 'sco>,
        magnitude: u32,
    ) -> Result<IntegerValue<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) => Ok(IntegerValue::U8(
                self.builder.append_op_with_result(felt::shr(
                    self.builder.unknown_location(),
                    value,
                    self.builder
                        .get_felt_constant_from_start(u64::from(magnitude))?,
                )?)?,
            )),
            IntegerValue::U16(value) => Ok(IntegerValue::U16(
                self.builder.append_op_with_result(felt::shr(
                    self.builder.unknown_location(),
                    value,
                    self.builder
                        .get_felt_constant_from_start(u64::from(magnitude))?,
                )?)?,
            )),
            IntegerValue::U32(value) => {
                Ok(IntegerValue::U32(self.shift_right_u32(value, magnitude)?))
            }
        }
    }

    /// Apply a logical left shift to an integer witness value.
    fn shift_left(
        &self,
        value: IntegerValue<'ctx, 'sco>,
        magnitude: u32,
    ) -> Result<IntegerValue<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) => {
                let shifted = self.builder.append_op_with_result(felt::shl(
                    self.builder.unknown_location(),
                    value,
                    self.builder
                        .get_felt_constant_from_start(u64::from(magnitude))?,
                )?)?;
                Ok(IntegerValue::U8(self.lowest_bits_felt(shifted, 8)?))
            }
            IntegerValue::U16(value) => {
                let shifted = self.builder.append_op_with_result(felt::shl(
                    self.builder.unknown_location(),
                    value,
                    self.builder
                        .get_felt_constant_from_start(u64::from(magnitude))?,
                )?)?;
                Ok(IntegerValue::U16(self.lowest_bits_felt(shifted, 16)?))
            }
            IntegerValue::U32(value) => {
                Ok(IntegerValue::U32(self.shift_left_u32(value, magnitude)?))
            }
        }
    }

    /// Keep only the lowest `num_bits` of an integer witness value.
    fn lowest_bits(
        &self,
        value: IntegerValue<'ctx, 'sco>,
        num_bits: u32,
    ) -> Result<IntegerValue<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) => Ok(IntegerValue::U8(
                self.lowest_bits_felt(value, num_bits.min(8))?,
            )),
            IntegerValue::U16(value) => Ok(IntegerValue::U16(
                self.lowest_bits_felt(value, num_bits.min(16))?,
            )),
            IntegerValue::U32(value) => {
                if num_bits >= 32 {
                    Ok(IntegerValue::U32(value))
                } else if num_bits <= 16 {
                    Ok(IntegerValue::U32(U32Parts {
                        low: self.lowest_bits_felt(value.low, num_bits)?,
                        high: self.builder.get_felt_constant_from_start(0)?,
                    }))
                } else {
                    Ok(IntegerValue::U32(U32Parts {
                        low: value.low,
                        high: self.lowest_bits_felt(value.high, num_bits - 16)?,
                    }))
                }
            }
        }
    }

    /// Select between two integer values of the same width.
    fn select_integer(
        &self,
        condition: Value<'ctx, 'sco>,
        if_true: IntegerValue<'ctx, 'sco>,
        if_false: IntegerValue<'ctx, 'sco>,
    ) -> Result<IntegerValue<'ctx, 'sco>> {
        match (if_true, if_false) {
            (IntegerValue::U8(lhs), IntegerValue::U8(rhs)) => {
                Ok(IntegerValue::U8(self.select_value(condition, lhs, rhs)?))
            }
            (IntegerValue::U16(lhs), IntegerValue::U16(rhs)) => {
                Ok(IntegerValue::U16(self.select_value(condition, lhs, rhs)?))
            }
            (IntegerValue::U32(lhs), IntegerValue::U32(rhs)) => Ok(IntegerValue::U32(U32Parts {
                low: self.select_value(condition, lhs.low, rhs.low)?,
                high: self.select_value(condition, lhs.high, rhs.high)?,
            })),
            _ => bail!("integer select requires both branches to have the same width"),
        }
    }

    /// Apply bitwise `not` while respecting the integer width.
    fn bitwise_not(&self, value: IntegerValue<'ctx, 'sco>) -> Result<IntegerValue<'ctx, 'sco>> {
        match value {
            IntegerValue::U8(value) => {
                let mask = self.builder.get_felt_constant_from_start(U8_MODULUS - 1)?;
                Ok(IntegerValue::U8(self.builder.append_op_with_result(
                    felt::bit_xor(self.builder.unknown_location(), value, mask)?,
                )?))
            }
            IntegerValue::U16(value) => {
                let mask = self.builder.get_felt_constant_from_start(U16_MODULUS - 1)?;
                Ok(IntegerValue::U16(self.builder.append_op_with_result(
                    felt::bit_xor(self.builder.unknown_location(), value, mask)?,
                )?))
            }
            IntegerValue::U32(value) => Ok(IntegerValue::U32(U32Parts {
                low: match self.bitwise_not(IntegerValue::U16(value.low))? {
                    IntegerValue::U16(value) => value,
                    _ => unreachable!(),
                },
                high: match self.bitwise_not(IntegerValue::U16(value.high))? {
                    IntegerValue::U16(value) => value,
                    _ => unreachable!(),
                },
            })),
        }
    }

    /// Apply a width-preserving bitwise binary operation.
    fn bitwise_binop(
        &self,
        opname: &str,
        lhs: IntegerValue<'ctx, 'sco>,
        rhs: IntegerValue<'ctx, 'sco>,
    ) -> Result<IntegerValue<'ctx, 'sco>> {
        let op = |lhs, rhs| -> Result<Value<'ctx, 'sco>> {
            match opname {
                "and" => self.builder.append_op_with_result(felt::bit_and(
                    self.builder.unknown_location(),
                    lhs,
                    rhs,
                )?),
                "or" => self.builder.append_op_with_result(felt::bit_or(
                    self.builder.unknown_location(),
                    lhs,
                    rhs,
                )?),
                "xor" => self.builder.append_op_with_result(felt::bit_xor(
                    self.builder.unknown_location(),
                    lhs,
                    rhs,
                )?),
                _ => bail!("unsupported bitwise operation {opname}"),
            }
        };

        match (lhs, rhs) {
            (IntegerValue::U8(lhs), IntegerValue::U8(rhs)) => Ok(IntegerValue::U8(op(lhs, rhs)?)),
            (IntegerValue::U16(lhs), IntegerValue::U16(rhs)) => {
                Ok(IntegerValue::U16(op(lhs, rhs)?))
            }
            (IntegerValue::U32(lhs), IntegerValue::U32(rhs)) => Ok(IntegerValue::U32(U32Parts {
                low: op(lhs.low, rhs.low)?,
                high: op(lhs.high, rhs.high)?,
            })),
            _ => bail!("bitwise operation requires operands of the same width"),
        }
    }

    /// Shift a 32-bit value right by manipulating its two 16-bit limbs directly.
    fn shift_right_u32(
        &self,
        value: U32Parts<'ctx, 'sco>,
        magnitude: u32,
    ) -> Result<U32Parts<'ctx, 'sco>> {
        if magnitude == 0 {
            return Ok(value);
        }
        if magnitude >= 32 {
            return Ok(U32Parts {
                low: self.builder.get_felt_constant_from_start(0)?,
                high: self.builder.get_felt_constant_from_start(0)?,
            });
        }
        if magnitude >= 16 {
            let shift = magnitude - 16;
            return Ok(U32Parts {
                low: self.builder.append_op_with_result(felt::shr(
                    self.builder.unknown_location(),
                    value.high,
                    self.builder
                        .get_felt_constant_from_start(u64::from(shift))?,
                )?)?,
                high: self.builder.get_felt_constant_from_start(0)?,
            });
        }

        let low_base = self.builder.append_op_with_result(felt::shr(
            self.builder.unknown_location(),
            value.low,
            self.builder
                .get_felt_constant_from_start(u64::from(magnitude))?,
        )?)?;
        let high_base = self.builder.append_op_with_result(felt::shr(
            self.builder.unknown_location(),
            value.high,
            self.builder
                .get_felt_constant_from_start(u64::from(magnitude))?,
        )?)?;
        let carry_mask = self
            .builder
            .get_felt_constant_from_start((1u64 << magnitude) - 1)?;
        let carry_bits = self.builder.append_op_with_result(felt::bit_and(
            self.builder.unknown_location(),
            value.high,
            carry_mask,
        )?)?;
        let carry = self.builder.append_op_with_result(felt::shl(
            self.builder.unknown_location(),
            carry_bits,
            self.builder
                .get_felt_constant_from_start(u64::from(16 - magnitude))?,
        )?)?;
        Ok(U32Parts {
            low: self.builder.append_op_with_result(felt::bit_or(
                self.builder.unknown_location(),
                low_base,
                carry,
            )?)?,
            high: high_base,
        })
    }

    /// Shift a 32-bit value left by manipulating its two 16-bit limbs directly.
    fn shift_left_u32(
        &self,
        value: U32Parts<'ctx, 'sco>,
        magnitude: u32,
    ) -> Result<U32Parts<'ctx, 'sco>> {
        if magnitude == 0 {
            return Ok(value);
        }
        if magnitude >= 32 {
            return Ok(U32Parts {
                low: self.builder.get_felt_constant_from_start(0)?,
                high: self.builder.get_felt_constant_from_start(0)?,
            });
        }
        if magnitude >= 16 {
            let shift = magnitude - 16;
            let shifted_high = self.builder.append_op_with_result(felt::shl(
                self.builder.unknown_location(),
                value.low,
                self.builder
                    .get_felt_constant_from_start(u64::from(shift))?,
            )?)?;
            return Ok(U32Parts {
                low: self.builder.get_felt_constant_from_start(0)?,
                high: self.lowest_bits_felt(shifted_high, 16)?,
            });
        }

        let low_shifted = self.builder.append_op_with_result(felt::shl(
            self.builder.unknown_location(),
            value.low,
            self.builder
                .get_felt_constant_from_start(u64::from(magnitude))?,
        )?)?;
        let high_shifted = self.builder.append_op_with_result(felt::shl(
            self.builder.unknown_location(),
            value.high,
            self.builder
                .get_felt_constant_from_start(u64::from(magnitude))?,
        )?)?;
        let carry = self.builder.append_op_with_result(felt::shr(
            self.builder.unknown_location(),
            value.low,
            self.builder
                .get_felt_constant_from_start(u64::from(16 - magnitude))?,
        )?)?;
        Ok(U32Parts {
            low: self.lowest_bits_felt(low_shifted, 16)?,
            high: self.builder.append_op_with_result(felt::bit_or(
                self.builder.unknown_location(),
                self.lowest_bits_felt(high_shifted, 16)?,
                carry,
            )?)?,
        })
    }

    /// Read the felt-encoded inputs for one SSA lookup invocation.
    fn lookup_inputs_as_fields(
        &self,
        input_subexpr_idxes: &[usize],
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        input_subexpr_idxes
            .iter()
            .map(|idx| self.slot_as_field(*idx))
            .collect()
    }

    /// Read the felt-encoded table id currently referenced by a lookup SSA node.
    ///
    /// Some circuits feed table ids through ordinary witness expressions before issuing the
    /// lookup, so the emitted `@compute` code needs access to the already-lowered SSA slot even
    /// when the table cannot be identified purely from the original raw expression.
    fn lookup_table_id_value(&self, table_id_subexpr_idx: usize) -> Result<Value<'ctx, 'sco>> {
        self.slot_as_field(table_id_subexpr_idx)
    }

    /// Resolve the table id referenced by a lookup SSA node when it is statically identifiable.
    ///
    /// The witness placer currently materializes table ids as constant subexpressions. We inspect
    /// the original block first, then fall back to the compiled lookup metadata when the SSA node
    /// carries a `lookup_mapping_idx`. If neither source is constant we return `None`, and the
    /// caller can emit a runtime dispatch across the supported deterministic table families.
    fn resolve_lookup_table(
        &self,
        table_id_subexpr_idx: usize,
        lookup_mapping_idx: Option<usize>,
    ) -> Result<Option<TableType>> {
        let table_expr = self
            .block
            .get(table_id_subexpr_idx)
            .ok_or_else(|| anyhow!("SSA slot {table_id_subexpr_idx} is out of bounds"))?;
        let table = match table_expr {
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU8(value)) => {
                Some(TableType::get_table_from_id(u32::from(*value)))
            }
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU16(value)) => {
                Some(TableType::get_table_from_id(u32::from(*value)))
            }
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU32(value)) => {
                Some(TableType::get_table_from_id(*value))
            }
            RawExpression::Field(FieldNodeExpression::Constant(value)) => {
                let table_id: u64 = value.as_u64_reduced();
                Some(TableType::get_table_from_id(u32::try_from(table_id)?))
            }
            _ => None,
        };

        if let Some(lookup_mapping_idx) = lookup_mapping_idx {
            let expected = self
                .lookup_sets
                .get(lookup_mapping_idx)
                .ok_or_else(|| anyhow!("lookup mapping {lookup_mapping_idx} is out of bounds"))?;
            match expected.table_index {
                TableIndex::Constant(expected_table) => {
                    if let Some(table) = table {
                        if expected_table != table {
                            bail!(
                                "SSA lookup mapping {lookup_mapping_idx} expects table {:?}, found {:?}",
                                expected_table,
                                table
                            );
                        }
                    }
                    return Ok(Some(expected_table));
                }
                TableIndex::Variable(column) => {
                    // TODO(LLZK compute): specialize dynamic lookup columns once a circuit needs
                    // table families that cannot be handled by the runtime dispatch below.
                    let _ = column;
                }
            }
        }

        Ok(table)
    }

    /// Lower one supported lookup table family into LLZK `@compute`.
    ///
    /// Keeping the table-to-helper dispatch in one place lets the static and dynamic resolution
    /// paths share the same deterministic implementations.
    fn compute_lookup_for_table(
        &self,
        table: TableType,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        match table {
            TableType::ConditionalJmpBranchSlt => {
                self.compute_conditional_jmp_branch_slt_lookup(inputs, num_outputs)
            }
            TableType::JumpCleanupOffset => {
                self.compute_jump_cleanup_offset_lookup(inputs, num_outputs)
            }
            TableType::MemoryGetOffsetAndMaskWithTrap => {
                self.compute_memory_get_offset_and_mask_with_trap_lookup(inputs, num_outputs)
            }
            TableType::RomAddressSpaceSeparator => {
                self.compute_rom_address_space_separator_lookup(inputs, num_outputs)
            }
            TableType::MemoryLoadHalfwordOrByte => {
                self.compute_memory_load_halfword_or_byte_lookup(inputs, num_outputs)
            }
            TableType::MemStoreClearOriginalRamValueLimb => {
                self.compute_mem_store_clear_original_ram_value_limb_lookup(inputs, num_outputs)
            }
            TableType::MemStoreClearWrittenValueLimb => {
                self.compute_mem_store_clear_written_value_limb_lookup(inputs, num_outputs)
            }
            TableType::AlignedRomRead => self.compute_aligned_rom_read_lookup(inputs, num_outputs),
            _ => {
                // TODO(LLZK compute): add deterministic lowering for the remaining lookup tables
                // used by the supported circuits.
                self.new_lookup_outputs(num_outputs)
            }
        }
    }

    /// Runtime-dispatch a lookup whose table id is chosen by witness expressions.
    ///
    /// `load_store_subword_only` builds several table ids by selecting among constant table
    /// numbers inside SSA blocks. For those cases we compute the supported candidate tables up
    /// front and select the matching output tuple by comparing the runtime table id.
    fn compute_dynamic_lookup(
        &self,
        table_id: Value<'ctx, 'sco>,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        let supported_tables: &[TableType] = match inputs.len() {
            1 => &[
                TableType::JumpCleanupOffset,
                TableType::MemoryGetOffsetAndMaskWithTrap,
                TableType::RomAddressSpaceSeparator,
                TableType::MemoryLoadHalfwordOrByte,
                TableType::MemStoreClearOriginalRamValueLimb,
                TableType::MemStoreClearWrittenValueLimb,
            ],
            2 => &[TableType::ConditionalJmpBranchSlt],
            _ => &[],
        };

        let mut outputs = self.new_lookup_outputs(num_outputs)?;
        for table in supported_tables.iter().copied() {
            let candidate_outputs = self.compute_lookup_for_table(table, inputs, num_outputs)?;
            let is_selected = self.field_eq_constant(table_id, u64::from(table.to_table_id()))?;
            outputs = candidate_outputs
                .into_iter()
                .zip(outputs.into_iter())
                .map(|(candidate, fallback)| self.select_value(is_selected, candidate, fallback))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(outputs)
    }

    /// Materialize the all-zero lookup row used by `maybe_lookup` and by padded table outputs.
    fn zero_lookup_outputs(&self, num_outputs: usize) -> Result<Vec<Value<'ctx, 'sco>>> {
        let zero = self.builder.get_felt_constant_from_start(0)?;
        Ok((0..num_outputs).map(|_| zero).collect())
    }

    /// Apply the `maybe_lookup` mask to already-lowered lookup outputs.
    ///
    /// This preserves the witness runtime contract from `simple_proxy::maybe_lookup`: inactive
    /// lookups return an all-zero tuple even if the active-path lowering is still nondeterministic.
    fn mask_lookup_outputs(
        &self,
        mask: Value<'ctx, 'sco>,
        outputs: Vec<Value<'ctx, 'sco>>,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        let zero = self.builder.get_felt_constant_from_start(0)?;
        outputs
            .into_iter()
            .map(|value| self.select_value(mask, value, zero))
            .collect()
    }

    /// Resize a concrete lookup result tuple to the arity requested by the SSA node.
    ///
    /// Some tables conceptually produce fewer than the full width-3 row, with trailing zeros
    /// reserved for unused columns. The SSA only asks for the outputs it later reads, so this
    /// helper trims or zero-pads accordingly.
    fn finalize_lookup_outputs(
        &self,
        mut outputs: Vec<Value<'ctx, 'sco>>,
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if outputs.len() > num_outputs {
            outputs.truncate(num_outputs);
            return Ok(outputs);
        }

        if outputs.len() < num_outputs {
            outputs.extend(self.zero_lookup_outputs(num_outputs - outputs.len())?);
        }

        Ok(outputs)
    }

    /// Build a boolean literal for control-flow within a lowered lookup.
    fn bool_constant(&self, value: bool) -> Result<Value<'ctx, 'sco>> {
        self.builder
            .get_constant_from_start(self.builder.bool_type(), value as u64)
    }

    /// Compare a felt-encoded small integer against a literal used by a lookup decoder.
    ///
    /// The witness SSA records lookup inputs as field elements, so deterministic lowering needs a
    /// compact way to recover table cases such as `funct3 == 0b101` without first building a wider
    /// integer type.
    fn field_eq_constant(
        &self,
        value: Value<'ctx, 'sco>,
        constant: u64,
    ) -> Result<Value<'ctx, 'sco>> {
        self.builder.append_op_with_result(bool::eq(
            self.builder.unknown_location(),
            value,
            self.builder.get_felt_constant_from_start(constant)?,
        )?)
    }

    /// Extract a small bit-slice from a felt-encoded lookup input.
    ///
    /// Many witness tables pack several control fields into one felt key. This helper mirrors the
    /// table-generation code by shifting right by a fixed amount and then reducing modulo `2^bits`.
    fn shifted_low_bits(
        &self,
        value: Value<'ctx, 'sco>,
        shift: u64,
        bits: u32,
    ) -> Result<Value<'ctx, 'sco>> {
        let shifted = if shift == 0 {
            value
        } else {
            self.builder.append_op_with_result(felt::shr(
                self.builder.unknown_location(),
                value,
                self.builder.get_felt_constant_from_start(shift)?,
            )?)?
        };
        self.lowest_bits_felt(shifted, bits)
    }

    /// Deterministically lower the branch/jump condition lookup used by `jump_branch_slt`.
    ///
    /// This matches `create_conditional_jmp_branch_slt_family_resolution_table` directly instead
    /// of routing through a materialized table: unpack the four comparison bits, derive the signed
    /// and unsigned comparison predicates, and then select the requested condition by `funct3`.
    fn compute_conditional_jmp_branch_slt_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 2 {
            bail!(
                "ConditionalJmpBranchSlt expects 2 inputs, found {}",
                inputs.len()
            );
        }

        let shift = |value, amount| {
            self.builder.append_op_with_result(felt::shr(
                self.builder.unknown_location(),
                value,
                self.builder.get_felt_constant_from_start(amount)?,
            )?)
        };

        let a = inputs[0];
        let funct3 = inputs[1];

        let uf = self.lowest_bits_felt(a, 1)?;
        let out_is_zero = self.lowest_bits_felt(shift(a, 1)?, 1)?;
        let sign1_felt = self.lowest_bits_felt(shift(a, 2)?, 1)?;
        let sign2_felt = self.lowest_bits_felt(shift(a, 3)?, 1)?;

        let eq = self.field_is_nonzero(out_is_zero)?;
        let unsigned_lt = self.field_is_nonzero(uf)?;
        let sign1 = self.field_is_nonzero(sign1_felt)?;
        let sign2 = self.field_is_nonzero(sign2_felt)?;
        let not_sign1 = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), sign1)?)?;
        let not_sign2 = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), sign2)?)?;
        let sign1_xor_sign2 = self.builder.append_op_with_result(bool::or(
            self.builder.unknown_location(),
            self.builder.append_op_with_result(bool::and(
                self.builder.unknown_location(),
                sign1,
                not_sign2,
            )?)?,
            self.builder.append_op_with_result(bool::and(
                self.builder.unknown_location(),
                not_sign1,
                sign2,
            )?)?,
        )?)?;
        let signed_lt = self.select_value(sign1_xor_sign2, sign1, unsigned_lt)?;
        let not_eq = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), eq)?)?;
        let not_signed_lt = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), signed_lt)?)?;
        let not_unsigned_lt = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), unsigned_lt)?)?;

        let match_funct3 = |value| {
            self.builder.append_op_with_result(bool::eq(
                self.builder.unknown_location(),
                funct3,
                self.builder.get_felt_constant_from_start(value)?,
            )?)
        };

        let flag = self.select_value(
            match_funct3(0)?,
            eq,
            self.select_value(
                match_funct3(1)?,
                not_eq,
                self.select_value(
                    match_funct3(2)?,
                    signed_lt,
                    self.select_value(
                        match_funct3(3)?,
                        unsigned_lt,
                        self.select_value(
                            match_funct3(4)?,
                            signed_lt,
                            self.select_value(
                                match_funct3(5)?,
                                not_signed_lt,
                                self.select_value(match_funct3(6)?, unsigned_lt, not_unsigned_lt)?,
                            )?,
                        )?,
                    )?,
                )?,
            )?,
        )?;

        self.finalize_lookup_outputs(vec![self.bool_to_field(flag)?], num_outputs)
    }

    /// Deterministically lower the low-PC cleanup lookup used by jumps and taken branches.
    ///
    /// The table returns `(bit_1, cleaned_low)` where `cleaned_low` is the 4-byte-aligned version
    /// of the low PC limb. Recomputing it directly keeps `@compute` faithful without embedding the
    /// full generated table.
    fn compute_jump_cleanup_offset_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!("JumpCleanupOffset expects 1 input, found {}", inputs.len());
        }

        let input = inputs[0];
        let check_bit = self.lowest_bits_felt(
            self.builder.append_op_with_result(felt::shr(
                self.builder.unknown_location(),
                input,
                self.builder.get_felt_constant_from_start(1)?,
            )?)?,
            1,
        )?;
        let cleaned = self.builder.append_op_with_result(felt::sub(
            self.builder.unknown_location(),
            input,
            self.lowest_bits_felt(input, 2)?,
        )?)?;

        self.finalize_lookup_outputs(vec![check_bit, cleaned], num_outputs)
    }

    /// Deterministically lower the packed offset/mask lookup used by subword memory ops.
    ///
    /// This mirrors `create_memory_offset_mask_with_trap_table`: unpack the low address bits and
    /// control flags from the composite key, derive the alignment trap condition, and then rebuild
    /// the compact bitmask consumed by the later witness SSA writes.
    fn compute_memory_get_offset_and_mask_with_trap_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!(
                "MemoryGetOffsetAndMaskWithTrap expects 1 input, found {}",
                inputs.len()
            );
        }

        let input = inputs[0];
        let offset = self.lowest_bits_felt(input, 2)?;
        let funct3 = self.shifted_low_bits(input, 16, 3)?;
        let is_store = self.field_is_nonzero(self.shifted_low_bits(input, 17, 1)?)?;
        let rd_is_x0 = self.field_is_nonzero(self.shifted_low_bits(input, 18, 1)?)?;

        let offset_is_nonzero = self.field_is_nonzero(offset)?;
        let offset_is_odd = self.field_is_nonzero(self.lowest_bits_felt(offset, 1)?)?;
        let false_bool = self.bool_constant(false)?;
        let match_funct3 = |value| self.field_eq_constant(funct3, value);

        let is_word = match_funct3(0b010)?;
        let is_halfword = self.builder.append_op_with_result(bool::or(
            self.builder.unknown_location(),
            match_funct3(0b001)?,
            match_funct3(0b101)?,
        )?)?;
        let is_byte = self.builder.append_op_with_result(bool::or(
            self.builder.unknown_location(),
            match_funct3(0b000)?,
            match_funct3(0b100)?,
        )?)?;

        let less_than_word = self.builder.append_op_with_result(bool::or(
            self.builder.unknown_location(),
            is_halfword,
            is_byte,
        )?)?;
        let base_trap = self.select_value(
            is_word,
            offset_is_nonzero,
            self.select_value(
                is_halfword,
                offset_is_odd,
                self.select_value(is_byte, false_bool, self.bool_constant(true)?)?,
            )?,
        )?;

        let valid_funct3_for_load = self.builder.append_op_with_result(bool::or(
            self.builder.unknown_location(),
            is_word,
            less_than_word,
        )?)?;
        let is_load = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), is_store)?)?;
        let allow_x0_unaligned_load = self.builder.append_op_with_result(bool::and(
            self.builder.unknown_location(),
            valid_funct3_for_load,
            self.builder.append_op_with_result(bool::and(
                self.builder.unknown_location(),
                is_load,
                rd_is_x0,
            )?)?,
        )?)?;
        let is_trap = self.select_value(allow_x0_unaligned_load, false_bool, base_trap)?;

        let use_high_limb = self.builder.append_op_with_result(bool::ge(
            self.builder.unknown_location(),
            offset,
            self.builder.get_felt_constant_from_start(2)?,
        )?)?;
        let bitmask = self.builder.append_op_with_result(felt::add(
            self.builder.unknown_location(),
            self.bool_to_field(less_than_word)?,
            self.builder.append_op_with_result(felt::add(
                self.builder.unknown_location(),
                self.builder.append_op_with_result(felt::mul(
                    self.builder.unknown_location(),
                    self.bool_to_field(use_high_limb)?,
                    self.builder.get_felt_constant_from_start(2)?,
                )?)?,
                self.builder.append_op_with_result(felt::mul(
                    self.builder.unknown_location(),
                    self.bool_to_field(is_trap)?,
                    self.builder.get_felt_constant_from_start(4)?,
                )?)?,
            )?)?,
        )?)?;

        self.finalize_lookup_outputs(vec![offset, bitmask], num_outputs)
    }

    /// Deterministically lower the ROM/RAM separator lookup used by subword loads.
    ///
    /// The generated table depends only on the fixed ROM boundary for the machine configuration,
    /// so `@compute` can reconstruct the same `(is_ram_range, rom_chunk)` tuple directly from the
    /// address high limb without consulting the materialized lookup table.
    fn compute_rom_address_space_separator_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!(
                "RomAddressSpaceSeparator expects 1 input, found {}",
                inputs.len()
            );
        }

        let input = inputs[0];
        let rom_bound = 1u64 << common_constants::ROM_SECOND_WORD_BITS;
        let is_ram_range = self.builder.append_op_with_result(bool::ge(
            self.builder.unknown_location(),
            input,
            self.builder.get_felt_constant_from_start(rom_bound)?,
        )?)?;
        let rom_chunk =
            self.lowest_bits_felt(input, common_constants::ROM_SECOND_WORD_BITS as u32)?;

        self.finalize_lookup_outputs(
            vec![self.bool_to_field(is_ram_range)?, rom_chunk],
            num_outputs,
        )
    }

    /// Deterministically lower the byte/halfword load extension table used by subword loads.
    ///
    /// This follows `create_memory_load_halfword_or_byte_table`: decode the selected 16-bit limb,
    /// the byte offset inside that limb, and the `funct3` mode, then rebuild the two 16-bit output
    /// limbs that represent the loaded 32-bit value.
    fn compute_memory_load_halfword_or_byte_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!(
                "MemoryLoadHalfwordOrByte expects 1 input, found {}",
                inputs.len()
            );
        }

        let input = inputs[0];
        let limb_value = self.lowest_bits_felt(input, 16)?;
        let offset = self.shifted_low_bits(input, 16, 2)?;
        let funct3 = self.shifted_low_bits(input, 18, 3)?;
        let offset_is_odd = self.field_is_nonzero(self.lowest_bits_felt(offset, 1)?)?;
        let use_low_byte = self
            .builder
            .append_op_with_result(bool::not(self.builder.unknown_location(), offset_is_odd)?)?;
        let low_byte = self.lowest_bits_felt(limb_value, 8)?;
        let high_byte = self.shifted_low_bits(limb_value, 8, 8)?;
        let selected_byte = self.select_value(use_low_byte, low_byte, high_byte)?;
        let byte_sign = self.field_is_nonzero(self.shifted_low_bits(selected_byte, 7, 1)?)?;
        let limb_sign = self.field_is_nonzero(self.shifted_low_bits(limb_value, 15, 1)?)?;
        let zero = self.builder.get_felt_constant_from_start(0)?;
        let byte_signed_low = self.select_value(
            byte_sign,
            self.builder.append_op_with_result(felt::add(
                self.builder.unknown_location(),
                selected_byte,
                self.builder.get_felt_constant_from_start(0xff00)?,
            )?)?,
            selected_byte,
        )?;
        let byte_signed_high = self.select_value(
            byte_sign,
            self.builder.get_felt_constant_from_start(0xffff)?,
            zero,
        )?;
        let halfword_signed_high = self.select_value(
            limb_sign,
            self.builder.get_felt_constant_from_start(0xffff)?,
            zero,
        )?;
        let halfword_low = self.select_value(offset_is_odd, zero, limb_value)?;
        let halfword_signed_high = self.select_value(offset_is_odd, zero, halfword_signed_high)?;

        let match_funct3 = |value| self.field_eq_constant(funct3, value);
        let low = self.select_value(
            match_funct3(0b010)?,
            zero,
            self.select_value(
                match_funct3(0b001)?,
                halfword_low,
                self.select_value(
                    match_funct3(0b101)?,
                    halfword_low,
                    self.select_value(
                        match_funct3(0b000)?,
                        byte_signed_low,
                        self.select_value(match_funct3(0b100)?, selected_byte, zero)?,
                    )?,
                )?,
            )?,
        )?;
        let high = self.select_value(
            match_funct3(0b010)?,
            zero,
            self.select_value(
                match_funct3(0b001)?,
                halfword_signed_high,
                self.select_value(
                    match_funct3(0b101)?,
                    zero,
                    self.select_value(match_funct3(0b000)?, byte_signed_high, zero)?,
                )?,
            )?,
        )?;

        self.finalize_lookup_outputs(vec![low, high], num_outputs)
    }

    /// Deterministically lower the table that clears bytes from the original RAM limb on stores.
    ///
    /// This mirrors `create_memory_store_halfword_or_byte_clear_source_limb_table`, which keeps
    /// only the untouched bytes of the original RAM limb before the cleaned write contribution is
    /// added back in.
    fn compute_mem_store_clear_original_ram_value_limb_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!(
                "MemStoreClearOriginalRamValueLimb expects 1 input, found {}",
                inputs.len()
            );
        }

        let input = inputs[0];
        let limb_value = self.lowest_bits_felt(input, 16)?;
        let offset = self.shifted_low_bits(input, 16, 2)?;
        let funct3 = self.shifted_low_bits(input, 18, 3)?;
        let offset_is_odd = self.field_is_nonzero(self.lowest_bits_felt(offset, 1)?)?;
        let cleaned_byte = self.select_value(
            offset_is_odd,
            self.lowest_bits_felt(limb_value, 8)?,
            self.builder.append_op_with_result(felt::bit_and(
                self.builder.unknown_location(),
                limb_value,
                self.builder.get_felt_constant_from_start(0xff00)?,
            )?)?,
        )?;
        let cleaned = self.select_value(
            self.field_eq_constant(funct3, 0b000)?,
            cleaned_byte,
            self.builder.get_felt_constant_from_start(0)?,
        )?;

        self.finalize_lookup_outputs(vec![cleaned], num_outputs)
    }

    /// Deterministically lower the table that positions the written byte/halfword contribution.
    ///
    /// This matches `create_memory_store_halfword_or_byte_clear_written_limb_table`: depending on
    /// the store width and byte offset, keep either the full halfword or the selected byte shifted
    /// into its destination position.
    fn compute_mem_store_clear_written_value_limb_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!(
                "MemStoreClearWrittenValueLimb expects 1 input, found {}",
                inputs.len()
            );
        }

        let input = inputs[0];
        let limb_value = self.lowest_bits_felt(input, 16)?;
        let offset = self.shifted_low_bits(input, 16, 2)?;
        let funct3 = self.shifted_low_bits(input, 18, 3)?;
        let offset_is_odd = self.field_is_nonzero(self.lowest_bits_felt(offset, 1)?)?;
        let value_to_store = self.lowest_bits_felt(limb_value, 8)?;
        let shifted_byte = self.select_value(
            offset_is_odd,
            self.builder.append_op_with_result(felt::shl(
                self.builder.unknown_location(),
                value_to_store,
                self.builder.get_felt_constant_from_start(8)?,
            )?)?,
            value_to_store,
        )?;
        let cleaned = self.select_value(
            self.field_eq_constant(funct3, 0b001)?,
            limb_value,
            self.select_value(
                self.field_eq_constant(funct3, 0b000)?,
                shifted_byte,
                self.builder.get_felt_constant_from_start(0)?,
            )?,
        )?;

        self.finalize_lookup_outputs(vec![cleaned], num_outputs)
    }

    /// Lower the ROM word lookup used by subword ROM loads through the external ROM hook.
    ///
    /// The aligned ROM table is keyed by word index and returns the 32-bit opcode split into two
    /// 16-bit limbs. We intentionally defer this lookup to the downstream LLZK executor instead of
    /// baking a placeholder bytecode image into the emitted IR.
    fn compute_aligned_rom_read_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!("AlignedRomRead expects 1 input, found {}", inputs.len());
        }

        let [low, high] = self.builder.append_call::<2>(
            self.builder.unknown_location(),
            READ_FROM_ROM_EXTERN,
            inputs,
            &[self.builder.felt_type(), self.builder.felt_type()],
        )?;
        self.finalize_lookup_outputs(vec![low, high], num_outputs)
    }

    /// Allocate witness holes for lookup outputs that are not yet lowered deterministically.
    fn new_lookup_outputs(&self, num_outputs: usize) -> Result<Vec<Value<'ctx, 'sco>>> {
        // TODO(LLZK compute): replace these witness holes with deterministic lookup lowering once
        // the remaining table families and hard witness-only helpers are modeled in `@compute`
        // without depending on external oracle state.
        (0..num_outputs)
            .map(|_| self.builder.new_nondet_felt())
            .collect()
    }

    /// Read a field-valued SSA slot by index.
    fn slot_as_field(&self, idx: usize) -> Result<Value<'ctx, 'sco>> {
        match self
            .slots
            .get(idx)
            .ok_or_else(|| anyhow!("SSA slot {idx} is out of bounds"))?
        {
            SsaSlot::Value(ComputedValue::Field(value)) => Ok(*value),
            SsaSlot::Value(ComputedValue::Integer(value)) => self.integer_to_field(*value),
            SsaSlot::Value(ComputedValue::Bool(value)) => self.bool_to_field(*value),
            SsaSlot::Lookup(_) => bail!("SSA slot {idx} is a lookup tuple, not a scalar value"),
            SsaSlot::Unit => bail!("SSA slot {idx} has no scalar value"),
        }
    }

    /// Read a boolean `i1` SSA slot by index.
    fn slot_as_bool(&self, idx: usize) -> Result<Value<'ctx, 'sco>> {
        match self
            .slots
            .get(idx)
            .ok_or_else(|| anyhow!("SSA slot {idx} is out of bounds"))?
        {
            SsaSlot::Value(ComputedValue::Bool(value)) => Ok(*value),
            _ => bail!("SSA slot {idx} does not contain a boolean value"),
        }
    }

    /// Read an integer SSA slot by index.
    fn slot_as_integer(&self, idx: usize) -> Result<IntegerValue<'ctx, 'sco>> {
        match self
            .slots
            .get(idx)
            .ok_or_else(|| anyhow!("SSA slot {idx} is out of bounds"))?
        {
            SsaSlot::Value(ComputedValue::Integer(value)) => Ok(*value),
            _ => bail!("SSA slot {idx} does not contain an integer value"),
        }
    }

    /// Read a lookup tuple SSA slot by index.
    fn slot_as_lookup(&self, idx: usize) -> Result<&[Value<'ctx, 'sco>]> {
        match self
            .slots
            .get(idx)
            .ok_or_else(|| anyhow!("SSA slot {idx} is out of bounds"))?
        {
            SsaSlot::Lookup(values) => Ok(values),
            _ => bail!("SSA slot {idx} does not contain lookup outputs"),
        }
    }

    fn strict_input_origin_for_expression(&self, expr: &Expression<F>) -> Option<Variable> {
        match expr {
            Expression::Bool(expr) => self.strict_input_origin_for_bool_expr(expr),
            Expression::Field(expr) => self.strict_input_origin_for_field_expr(expr),
            Expression::U8(expr) | Expression::U16(expr) | Expression::U32(expr) => {
                self.strict_input_origin_for_integer_expr(expr)
            }
        }
    }

    fn strict_input_origin_for_field_expr(
        &self,
        expr: &FieldNodeExpression<F>,
    ) -> Option<Variable> {
        match expr {
            FieldNodeExpression::Place(var) => self.vars.has_compute_input(var).then_some(*var),
            FieldNodeExpression::SubExpression(idx) => {
                self.slot_input_origins.get(*idx).copied().flatten()
            }
            FieldNodeExpression::FromInteger(inner) => {
                self.strict_input_origin_for_integer_expr(inner)
            }
            FieldNodeExpression::FromMask(inner) => self.strict_input_origin_for_bool_expr(inner),
            FieldNodeExpression::OracleValue { placeholder, .. } => {
                self.strict_placeholder_input_origin(*placeholder)
            }
            _ => None,
        }
    }

    fn strict_input_origin_for_bool_expr(&self, expr: &BoolNodeExpression<F>) -> Option<Variable> {
        match expr {
            BoolNodeExpression::Place(var) => self.vars.has_compute_input(var).then_some(*var),
            BoolNodeExpression::SubExpression(idx) => {
                self.slot_input_origins.get(*idx).copied().flatten()
            }
            BoolNodeExpression::OracleValue { placeholder } => {
                self.strict_placeholder_input_origin(*placeholder)
            }
            _ => None,
        }
    }

    fn strict_placeholder_input_origin(&self, placeholder: Placeholder) -> Option<Variable> {
        self.substitutions
            .get(&(placeholder, 0))
            .copied()
            .filter(|var| self.vars.has_compute_input(var))
    }

    fn strict_input_origin_for_integer_expr(
        &self,
        expr: &FixedWidthIntegerNodeExpression<F>,
    ) -> Option<Variable> {
        match expr {
            FixedWidthIntegerNodeExpression::U8Place(var)
            | FixedWidthIntegerNodeExpression::U16Place(var) => {
                self.vars.has_compute_input(var).then_some(*var)
            }
            FixedWidthIntegerNodeExpression::U8SubExpression(idx)
            | FixedWidthIntegerNodeExpression::U16SubExpression(idx)
            | FixedWidthIntegerNodeExpression::U32SubExpression(idx) => {
                self.slot_input_origins.get(*idx).copied().flatten()
            }
            FixedWidthIntegerNodeExpression::U32OracleValue { placeholder }
            | FixedWidthIntegerNodeExpression::U16OracleValue { placeholder }
            | FixedWidthIntegerNodeExpression::U8OracleValue { placeholder } => {
                self.strict_placeholder_input_origin(*placeholder)
            }
            FixedWidthIntegerNodeExpression::WidenFromU8(inner)
            | FixedWidthIntegerNodeExpression::WidenFromU16(inner)
            | FixedWidthIntegerNodeExpression::TruncateFromU16(inner)
            | FixedWidthIntegerNodeExpression::TruncateFromU32(inner)
            | FixedWidthIntegerNodeExpression::I32FromU32(inner)
            | FixedWidthIntegerNodeExpression::U32FromI32(inner) => {
                self.strict_input_origin_for_integer_expr(inner)
            }
            FixedWidthIntegerNodeExpression::WrappingShr { lhs, magnitude }
                if *magnitude == 0 || *magnitude == 16 =>
            {
                self.strict_input_origin_for_integer_expr(lhs)
            }
            _ => None,
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F> for RawExpression<F> {
    type Output = ();

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        let (slot, input_origin) = if let Some(lookup) = LookupInvocation::from_raw(self) {
            (SsaSlot::Lookup(lookup.emit_compute(lowering)?), None)
        } else {
            match self {
                RawExpression::Bool(expr) => (
                    SsaSlot::Value(ComputedValue::Bool(expr.emit_compute(lowering)?)),
                    lowering.strict_input_origin_for_bool_expr(expr),
                ),
                RawExpression::Field(expr) => (
                    SsaSlot::Value(ComputedValue::Field(expr.emit_compute(lowering)?)),
                    lowering.strict_input_origin_for_field_expr(expr),
                ),
                RawExpression::Integer(expr) => (
                    SsaSlot::Value(ComputedValue::Integer(expr.emit_compute(lowering)?)),
                    lowering.strict_input_origin_for_integer_expr(expr),
                ),
                RawExpression::AccessLookup {
                    subindex,
                    output_index,
                } => {
                    let lookup_value = *lowering
                        .slot_as_lookup(*subindex)?
                        .get(*output_index)
                        .ok_or_else(|| anyhow!("lookup output {output_index} is out of bounds"))?;
                    (SsaSlot::Value(ComputedValue::Field(lookup_value)), None)
                }
                RawExpression::WriteVariable {
                    into_variable,
                    source_subexpr,
                    condition_subexpr_idx,
                } => {
                    lowering.lower_write(into_variable, source_subexpr, *condition_subexpr_idx)?;
                    (SsaSlot::Unit, None)
                }
                RawExpression::PerformLookup { .. } | RawExpression::MaybePerformLookup { .. } => {
                    unreachable!("lookup raw expressions are handled by LookupInvocation")
                }
            }
        };

        lowering.push_slot(slot, input_origin);
        Ok(())
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F> for LookupInvocation {
    type Output = Vec<Value<'ctx, 'sco>>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        let inputs = lowering.lookup_inputs_as_fields(self.input_subexpr_idxes())?;
        let outputs = if let Some(table) =
            lowering.resolve_lookup_table(self.table_id_subexpr_idx(), self.lookup_mapping_idx())?
        {
            lowering.compute_lookup_for_table(table, &inputs, self.num_outputs())?
        } else {
            lowering.compute_dynamic_lookup(
                lowering.lookup_table_id_value(self.table_id_subexpr_idx())?,
                &inputs,
                self.num_outputs(),
            )?
        };

        if let Some(mask_id_subexpr_idx) = self.mask_id_subexpr_idx() {
            let mask = lowering.slot_as_bool(mask_id_subexpr_idx)?;
            lowering.mask_lookup_outputs(mask, outputs)
        } else {
            Ok(outputs)
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F> for Expression<F> {
    type Output = ComputedValue<'ctx, 'sco>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        match self {
            Expression::Bool(expr) => Ok(ComputedValue::Bool(expr.emit_compute(lowering)?)),
            Expression::Field(expr) => Ok(ComputedValue::Field(expr.emit_compute(lowering)?)),
            Expression::U8(expr) | Expression::U16(expr) | Expression::U32(expr) => {
                Ok(ComputedValue::Integer(expr.emit_compute(lowering)?))
            }
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F>
    for FieldNodeExpression<F>
{
    type Output = Value<'ctx, 'sco>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        match self {
            FieldNodeExpression::Place(variable) => lowering.read_variable(*variable),
            FieldNodeExpression::SubExpression(idx) => lowering.slot_as_field(*idx),
            FieldNodeExpression::Constant(constant) => lowering
                .builder
                .get_felt_constant_from_start(constant.as_u64_reduced()),
            FieldNodeExpression::FromInteger(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering.integer_to_field(value)
            }
            FieldNodeExpression::FromMask(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering.bool_to_field(value)
            }
            FieldNodeExpression::OracleValue {
                placeholder,
                subindex,
            } => lowering.read_field_oracle(*placeholder, *subindex),
            FieldNodeExpression::Add { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(felt::add(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            FieldNodeExpression::Sub { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(felt::sub(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            FieldNodeExpression::Mul { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(felt::mul(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            FieldNodeExpression::AddProduct {
                additive_term,
                mul_0,
                mul_1,
            } => {
                let mul_0 = mul_0.emit_compute(lowering)?;
                let mul_1 = mul_1.emit_compute(lowering)?;
                let product = lowering.builder.append_op_with_result(felt::mul(
                    lowering.builder.unknown_location(),
                    mul_0,
                    mul_1,
                )?)?;
                let additive_term = additive_term.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(felt::add(
                    lowering.builder.unknown_location(),
                    additive_term,
                    product,
                )?)
            }
            FieldNodeExpression::Select {
                selector,
                if_true,
                if_false,
            } => {
                let selector = selector.emit_compute(lowering)?;
                let if_true = if_true.emit_compute(lowering)?;
                let if_false = if_false.emit_compute(lowering)?;
                lowering.select_value(selector, if_true, if_false)
            }
            FieldNodeExpression::InverseUnchecked(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering
                    .builder
                    .append_op_with_result(felt::inv(lowering.builder.unknown_location(), value)?)
            }
            // TODO(LLZK compute): lower inverse-or-zero deterministically instead of using a
            // witness hole.
            FieldNodeExpression::InverseOrZero(_expr) => lowering.builder.new_nondet_felt(),
            FieldNodeExpression::LookupOutput { .. }
            | FieldNodeExpression::MaybeLookupOutput { .. } => {
                bail!("lookup outputs must be rewritten into SSA access nodes before lowering")
            }
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F>
    for BoolNodeExpression<F>
{
    type Output = Value<'ctx, 'sco>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        match self {
            BoolNodeExpression::Place(variable) => {
                lowering.field_is_nonzero(lowering.read_variable(*variable)?)
            }
            BoolNodeExpression::SubExpression(idx) => lowering.slot_as_bool(*idx),
            BoolNodeExpression::Constant(constant) => lowering
                .builder
                .get_constant_from_start(lowering.builder.bool_type(), *constant as u64),
            BoolNodeExpression::OracleValue { placeholder } => {
                lowering.read_bool_oracle(*placeholder)
            }
            BoolNodeExpression::FromGenericInteger(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering.integer_is_nonzero(value)
            }
            BoolNodeExpression::FromGenericIntegerEquality { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.integer_equal(lhs, rhs)
            }
            BoolNodeExpression::FromGenericIntegerCarry { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.overflowing_add(lhs, rhs).map(|(_, carry)| carry)
            }
            BoolNodeExpression::FromGenericIntegerBorrow { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.overflowing_sub(lhs, rhs).map(|(_, borrow)| borrow)
            }
            BoolNodeExpression::FromField(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering.field_is_nonzero(value)
            }
            BoolNodeExpression::FromFieldEquality { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(bool::eq(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            BoolNodeExpression::And { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(bool::and(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            BoolNodeExpression::Or { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.builder.append_op_with_result(bool::or(
                    lowering.builder.unknown_location(),
                    lhs,
                    rhs,
                )?)
            }
            BoolNodeExpression::Select {
                selector,
                if_true,
                if_false,
            } => {
                let selector = selector.emit_compute(lowering)?;
                let if_true = if_true.emit_compute(lowering)?;
                let if_false = if_false.emit_compute(lowering)?;
                lowering.select_value(selector, if_true, if_false)
            }
            BoolNodeExpression::Negate(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering
                    .builder
                    .append_op_with_result(bool::not(lowering.builder.unknown_location(), value)?)
            }
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLlzkInCompute<'a, 'ctx, 'sco, F>
    for FixedWidthIntegerNodeExpression<F>
{
    type Output = IntegerValue<'ctx, 'sco>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        match self {
            FixedWidthIntegerNodeExpression::U8Place(variable) => {
                Ok(IntegerValue::U8(lowering.read_variable(*variable)?))
            }
            FixedWidthIntegerNodeExpression::U16Place(variable) => {
                Ok(IntegerValue::U16(lowering.read_variable(*variable)?))
            }
            FixedWidthIntegerNodeExpression::U8SubExpression(idx)
            | FixedWidthIntegerNodeExpression::U16SubExpression(idx)
            | FixedWidthIntegerNodeExpression::U32SubExpression(idx) => {
                lowering.slot_as_integer(*idx)
            }
            FixedWidthIntegerNodeExpression::U32OracleValue { placeholder } => {
                Ok(IntegerValue::U32(lowering.read_u32_oracle(*placeholder)?))
            }
            FixedWidthIntegerNodeExpression::U16OracleValue { placeholder } => {
                Ok(IntegerValue::U16(lowering.read_u16_oracle(*placeholder)?))
            }
            FixedWidthIntegerNodeExpression::U8OracleValue { placeholder } => {
                Ok(IntegerValue::U8(lowering.read_u8_oracle(*placeholder)?))
            }
            FixedWidthIntegerNodeExpression::ConstantU8(constant) => Ok(IntegerValue::U8(
                lowering
                    .builder
                    .get_felt_constant_from_start(u64::from(*constant))?,
            )),
            FixedWidthIntegerNodeExpression::ConstantU16(constant) => Ok(IntegerValue::U16(
                lowering
                    .builder
                    .get_felt_constant_from_start(u64::from(*constant))?,
            )),
            FixedWidthIntegerNodeExpression::ConstantU32(constant) => {
                Ok(IntegerValue::U32(lowering.u32_constant(*constant)?))
            }
            FixedWidthIntegerNodeExpression::U32FromMask(expr) => {
                let value = expr.emit_compute(lowering)?;
                let low = lowering.bool_to_field(value)?;
                Ok(IntegerValue::U32(U32Parts {
                    low,
                    high: lowering.builder.get_felt_constant_from_start(0)?,
                }))
            }
            FixedWidthIntegerNodeExpression::U32FromField(expr) => {
                let value = expr.emit_compute(lowering)?;
                Ok(IntegerValue::U32(lowering.field_to_u32(value)?))
            }
            FixedWidthIntegerNodeExpression::WidenFromU8(expr) => {
                match expr.emit_compute(lowering)? {
                    IntegerValue::U8(value) => Ok(IntegerValue::U16(value)),
                    other => bail!(
                        "expected u8 input when widening to u16, found {}-bit value",
                        other.bit_width()
                    ),
                }
            }
            FixedWidthIntegerNodeExpression::WidenFromU16(expr) => {
                match expr.emit_compute(lowering)? {
                    IntegerValue::U16(value) => Ok(IntegerValue::U32(U32Parts {
                        low: value,
                        high: lowering.builder.get_felt_constant_from_start(0)?,
                    })),
                    other => bail!(
                        "expected u16 input when widening to u32, found {}-bit value",
                        other.bit_width()
                    ),
                }
            }
            FixedWidthIntegerNodeExpression::TruncateFromU16(expr) => {
                match expr.emit_compute(lowering)? {
                    IntegerValue::U16(value) => {
                        Ok(IntegerValue::U8(lowering.lowest_bits_felt(value, 8)?))
                    }
                    other => bail!(
                        "expected u16 input when truncating to u8, found {}-bit value",
                        other.bit_width()
                    ),
                }
            }
            FixedWidthIntegerNodeExpression::TruncateFromU32(expr) => {
                match expr.emit_compute(lowering)? {
                    IntegerValue::U32(value) => Ok(IntegerValue::U16(value.low)),
                    other => bail!(
                        "expected u32 input when truncating to u16, found {}-bit value",
                        other.bit_width()
                    ),
                }
            }
            FixedWidthIntegerNodeExpression::I32FromU32(expr)
            | FixedWidthIntegerNodeExpression::U32FromI32(expr) => expr.emit_compute(lowering),
            FixedWidthIntegerNodeExpression::Select {
                selector,
                if_true,
                if_false,
            } => {
                let selector = selector.emit_compute(lowering)?;
                let if_true = if_true.emit_compute(lowering)?;
                let if_false = if_false.emit_compute(lowering)?;
                lowering.select_integer(selector, if_true, if_false)
            }
            FixedWidthIntegerNodeExpression::WrappingAdd { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.overflowing_add(lhs, rhs).map(|(value, _)| value)
            }
            FixedWidthIntegerNodeExpression::WrappingSub { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.overflowing_sub(lhs, rhs).map(|(value, _)| value)
            }
            FixedWidthIntegerNodeExpression::WrappingShl { lhs, magnitude } => {
                let lhs = lhs.emit_compute(lowering)?;
                lowering.shift_left(lhs, *magnitude)
            }
            FixedWidthIntegerNodeExpression::WrappingShr { lhs, magnitude } => {
                let lhs = lhs.emit_compute(lowering)?;
                lowering.shift_right(lhs, *magnitude)
            }
            FixedWidthIntegerNodeExpression::LowestBits { value, num_bits } => {
                let value = value.emit_compute(lowering)?;
                lowering.lowest_bits(value, *num_bits)
            }
            FixedWidthIntegerNodeExpression::BinaryNot(expr) => {
                let value = expr.emit_compute(lowering)?;
                lowering.bitwise_not(value)
            }
            FixedWidthIntegerNodeExpression::BinaryAnd { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.bitwise_binop("and", lhs, rhs)
            }
            FixedWidthIntegerNodeExpression::BinaryOr { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.bitwise_binop("or", lhs, rhs)
            }
            FixedWidthIntegerNodeExpression::BinaryXor { lhs, rhs } => {
                let lhs = lhs.emit_compute(lowering)?;
                let rhs = rhs.emit_compute(lowering)?;
                lowering.bitwise_binop("xor", lhs, rhs)
            }
            // TODO(LLZK compute): support the remaining integer witness ops once their exact LLZK
            // lowering has been validated against the Rust witness evaluator.
            FixedWidthIntegerNodeExpression::MulLow { .. }
            | FixedWidthIntegerNodeExpression::MulHigh { .. }
            | FixedWidthIntegerNodeExpression::AddProduct { .. }
            | FixedWidthIntegerNodeExpression::DivAssumeNonzero { .. }
            | FixedWidthIntegerNodeExpression::RemAssumeNonzero { .. }
            | FixedWidthIntegerNodeExpression::SignedDivAssumeNonzeroNoOverflowBits { .. }
            | FixedWidthIntegerNodeExpression::SignedRemAssumeNonzeroNoOverflowBits { .. }
            | FixedWidthIntegerNodeExpression::SignedMulLowBits { .. }
            | FixedWidthIntegerNodeExpression::SignedMulHighBits { .. }
            | FixedWidthIntegerNodeExpression::SignedByUnsignedMulLowBits { .. }
            | FixedWidthIntegerNodeExpression::SignedByUnsignedMulHighBits { .. } => {
                bail!(
                    "integer operation {:?} is not yet supported in @compute",
                    self
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prover::cs::tables::TableDriver;
    use prover::field::Mersenne31Field;

    fn field(value: u64) -> Mersenne31Field {
        Mersenne31Field::from_u64_unchecked(value)
    }

    fn lookup_values<const N: usize>(
        table_driver: &TableDriver<Mersenne31Field>,
        table: TableType,
        inputs: &[u64],
    ) -> [u64; N] {
        let keys: Vec<_> = inputs.iter().copied().map(field).collect();
        table_driver
            .lookup_values::<N>(&keys, table.to_table_id())
            .map(|value| value.as_u64_reduced())
    }

    fn jump_branch_lookup_outputs(input_bits: u64, funct3: u64) -> [u64; 1] {
        let uf = input_bits & 1;
        let out_is_zero = (input_bits >> 1) & 1;
        let sign1 = (input_bits >> 2) & 1;
        let sign2 = (input_bits >> 3) & 1;

        let eq = out_is_zero != 0;
        let unsigned_lt = uf != 0;
        let signed_lt = if sign1 ^ sign2 == 1 {
            sign1 != 0
        } else {
            unsigned_lt
        };

        let flag = match funct3 {
            0b000 => eq,
            0b001 => !eq,
            0b010 | 0b100 => signed_lt,
            0b011 | 0b110 => unsigned_lt,
            0b101 => !signed_lt,
            0b111 => !unsigned_lt,
            _ => unreachable!(),
        };

        [flag as u64]
    }

    fn jump_cleanup_offset_outputs(input: u64) -> [u64; 2] {
        [(input >> 1) & 1, input & !0x3]
    }

    fn memory_get_offset_and_mask_with_trap_outputs(input: u64) -> [u64; 2] {
        let mem_address_low = input & 0xffff;
        let funct3 = (input >> 16) & 0b111;
        let is_store = ((input >> 17) & 1) != 0;
        let rd_is_x0 = ((input >> 18) & 1) != 0;

        let offset = mem_address_low & 0b11;
        let mut less_than_word = false;
        let mut is_trap = match (funct3, offset) {
            (0b010, offset) => offset != 0,
            (0b001, offset) | (0b101, offset) => {
                less_than_word = true;
                offset & 1 != 0
            }
            (0b000, _) | (0b100, _) => {
                less_than_word = true;
                false
            }
            _ => true,
        };
        let valid_funct3_for_load = matches!(funct3, 0b000 | 0b001 | 0b010 | 0b100 | 0b101);

        if valid_funct3_for_load && !is_store && rd_is_x0 {
            is_trap = false;
        }

        let use_high_limb = offset > 1;
        let mut bitmask = less_than_word as u64;
        bitmask |= (use_high_limb as u64) << 1;
        bitmask |= (is_trap as u64) << 2;

        [offset, bitmask]
    }

    fn rom_address_space_separator_outputs(input: u64) -> [u64; 2] {
        let bound = 1u64 << common_constants::ROM_SECOND_WORD_BITS;
        [(input >= bound) as u64, input % bound]
    }

    fn memory_load_halfword_or_byte_outputs(input: u64) -> [u64; 2] {
        let limb_value = input & 0xffff;
        let offset = (input >> 16) & 0b11;
        let funct3 = (input >> 18) & 0b111;
        let use_low_byte = offset & 1 == 0;

        match (funct3, offset) {
            (0b010, _) => [0, 0],
            (0b001, offset) => {
                if offset & 1 != 0 {
                    [0, 0]
                } else if (limb_value >> 15) != 0 {
                    [limb_value, 0xffff]
                } else {
                    [limb_value, 0]
                }
            }
            (0b101, offset) => {
                if offset & 1 != 0 {
                    [0, 0]
                } else {
                    [limb_value, 0]
                }
            }
            (0b000, _) => {
                let source = if use_low_byte {
                    limb_value & 0xff
                } else {
                    limb_value >> 8
                };
                if (source >> 7) != 0 {
                    [source | 0xff00, 0xffff]
                } else {
                    [source, 0]
                }
            }
            (0b100, _) => {
                let source = if use_low_byte {
                    limb_value & 0xff
                } else {
                    limb_value >> 8
                };
                [source, 0]
            }
            _ => [0, 0],
        }
    }

    fn mem_store_clear_original_ram_value_limb_outputs(input: u64) -> [u64; 2] {
        let limb_value = input & 0xffff;
        let offset = (input >> 16) & 0b11;
        let funct3 = (input >> 18) & 0b111;

        let cleaned_value = match (funct3, offset) {
            (0b010, _) | (0b001, _) => 0,
            (0b000, offset) => {
                let mask = if offset & 1 != 0 { 0x00ff } else { 0xff00 };
                limb_value & mask
            }
            _ => 0,
        };

        [cleaned_value, 0]
    }

    fn mem_store_clear_written_value_limb_outputs(input: u64) -> [u64; 2] {
        let limb_value = input & 0xffff;
        let offset = (input >> 16) & 0b11;
        let funct3 = (input >> 18) & 0b111;

        let cleaned_value = match (funct3, offset) {
            (0b010, _) => 0,
            (0b001, _) => limb_value,
            (0b000, offset) => {
                let value_to_store = limb_value & 0xff;
                if offset & 1 != 0 {
                    value_to_store << 8
                } else {
                    value_to_store
                }
            }
            _ => 0,
        };

        [cleaned_value, 0]
    }

    #[test]
    fn jump_branch_lookup_semantics_match_table_driver() {
        let bytecode_words = (1 << (16 + jump_branch_slt::ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;
        let table_driver = jump_branch_slt::get_table_driver(&vec![0u32; bytecode_words]);

        for input_bits in 0..(1 << 4) {
            for funct3 in 0..(1 << 3) {
                assert_eq!(
                    lookup_values::<1>(
                        &table_driver,
                        TableType::ConditionalJmpBranchSlt,
                        &[input_bits, funct3],
                    ),
                    jump_branch_lookup_outputs(input_bits, funct3),
                );
            }
        }

        for input in 0..(1 << 16) {
            assert_eq!(
                lookup_values::<2>(&table_driver, TableType::JumpCleanupOffset, &[input]),
                jump_cleanup_offset_outputs(input),
            );
        }
    }

    #[test]
    fn subword_lookup_semantics_match_table_driver() {
        let bytecode_words =
            (1 << (16 + load_store_subword_only::ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;
        let table_driver = load_store_subword_only::get_table_driver(&vec![0u32; bytecode_words]);

        for address_low in [0u64, 1, 2, 3, 0x1234, 0xffff] {
            for funct3 in 0..8u64 {
                for is_store in 0..=1u64 {
                    for rd_is_x0 in 0..=1u64 {
                        let input =
                            address_low | (funct3 << 16) | (is_store << 17) | (rd_is_x0 << 18);
                        assert_eq!(
                            lookup_values::<2>(
                                &table_driver,
                                TableType::MemoryGetOffsetAndMaskWithTrap,
                                &[input],
                            ),
                            memory_get_offset_and_mask_with_trap_outputs(input),
                        );
                    }
                }
            }
        }

        let rom_bound = 1u64 << common_constants::ROM_SECOND_WORD_BITS;
        for input in [
            0u64,
            1,
            rom_bound.saturating_sub(1),
            rom_bound,
            rom_bound + 1,
            0xffff,
        ] {
            assert_eq!(
                lookup_values::<2>(&table_driver, TableType::RomAddressSpaceSeparator, &[input]),
                rom_address_space_separator_outputs(input),
            );
        }

        let limb_values = [0u64, 1, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff];
        for limb_value in limb_values {
            for offset in 0..4u64 {
                for funct3 in 0..8u64 {
                    let input = limb_value | (offset << 16) | (funct3 << 18);
                    assert_eq!(
                        lookup_values::<2>(
                            &table_driver,
                            TableType::MemoryLoadHalfwordOrByte,
                            &[input],
                        ),
                        memory_load_halfword_or_byte_outputs(input),
                    );
                    assert_eq!(
                        lookup_values::<2>(
                            &table_driver,
                            TableType::MemStoreClearOriginalRamValueLimb,
                            &[input],
                        ),
                        mem_store_clear_original_ram_value_limb_outputs(input),
                    );
                    assert_eq!(
                        lookup_values::<2>(
                            &table_driver,
                            TableType::MemStoreClearWrittenValueLimb,
                            &[input],
                        ),
                        mem_store_clear_written_value_limb_outputs(input),
                    );
                }
            }
        }
    }

    #[test]
    fn oracle_placeholder_encoding_is_stable_for_parameterized_variants() {
        assert_eq!(
            encode_oracle_placeholder(Placeholder::PcInit),
            EncodedOraclePlaceholder {
                kind: 6,
                arg0: 0,
                arg1: 0,
            }
        );
        assert_eq!(
            encode_oracle_placeholder(Placeholder::ShuffleRamReadValue(2)),
            EncodedOraclePlaceholder {
                kind: 43,
                arg0: 2,
                arg1: 0,
            }
        );
        assert_eq!(
            encode_oracle_placeholder(Placeholder::DelegationIndirectReadValue {
                register_index: 5,
                word_index: 1,
            }),
            EncodedOraclePlaceholder {
                kind: 56,
                arg0: 5,
                arg1: 1,
            }
        );
    }
}
