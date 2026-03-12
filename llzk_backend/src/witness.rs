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
//! 3. lower writes that target witness or scratch columns into struct member updates; and
//! 4. leave unsupported lookup outputs and a small number of hard witness-only helpers as
//!    `llzk.nondet` holes until table-specific compute lowering is added.

use std::collections::BTreeMap;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use llzk::dialect::bool;
use llzk::dialect::felt;
use llzk::prelude::melior_dialects::arith;
use llzk::prelude::*;
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

use crate::builder::OpsBuilder;
use crate::codegen::StructVars;
use crate::field::FieldInfo;

const U8_MODULUS: u64 = 1 << 8;
const U16_MODULUS: u64 = 1 << 16;

/// Bundles the metadata required to lower witness generation into LLZK `@compute`.
///
/// The compiled artifact tells us where every logical variable lives in the witness layout, while
/// the SSA blocks preserve the witness placer's evaluation order and conditional write structure.
pub(crate) struct WitnessComputation<F: PrimeField + FieldInfo> {
    compiled: CompiledCircuitArtifact<F>,
    ssa: Vec<Vec<RawExpression<F>>>,
}

impl<F: PrimeField + FieldInfo> WitnessComputation<F> {
    /// Create a new witness computation plan from the one-row compiler output and witness SSA.
    pub fn new(compiled: CompiledCircuitArtifact<F>, ssa: Vec<Vec<RawExpression<F>>>) -> Self {
        Self { compiled, ssa }
    }

    /// Emit LLZK operations that reconstruct witness columns inside a struct `@compute` function.
    ///
    /// The lowering intentionally follows the SSA block structure produced by the witness placer so
    /// that every new helper can be reviewed against the existing Rust witness evaluator one block
    /// at a time.
    pub fn emit_compute<'ctx, 'sco>(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars,
    ) -> Result<()> {
        let self_value = builder.get_compute_self_value()?;
        for block in &self.ssa {
            if should_skip_block(&self.compiled.variable_mapping, block) {
                continue;
            }

            let mut lowering = ComputeLowering::new(
                builder,
                vars,
                self_value,
                &self.compiled.variable_mapping,
                &self.compiled.witness_layout.width_3_lookups,
                block,
            );
            block.emit_compute(&mut lowering)?;
        }
        Ok(())
    }
}

/// Keep the same "no memory writes" policy as the Rust witness generator.
///
/// LLZK `@compute` currently materializes only the witness struct returned by the function, so
/// blocks that exclusively update transient memory columns can be skipped without changing the
/// externally visible witness values. This mirrors the
/// `perform_assignments_to_memory == false` fast path in
/// `witness_eval_generator/src/derive_from_ssa/mod.rs::derive_from_ssa`, which performs the same
/// block-level scan before emitting Rust witness code.
fn should_skip_block<F: PrimeField>(
    layout: &BTreeMap<Variable, ColumnAddress>,
    expressions: &[RawExpression<F>],
) -> bool {
    let mut can_skip = true;
    for expr in expressions {
        if let RawExpression::WriteVariable { into_variable, .. } = expr {
            match layout[into_variable] {
                ColumnAddress::MemorySubtree(..) => {}
                _ => {
                    can_skip = false;
                    break;
                }
            }
        }
        if matches!(
            expr,
            RawExpression::PerformLookup { .. } | RawExpression::MaybePerformLookup { .. }
        ) {
            can_skip = false;
            break;
        }
    }
    can_skip
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
trait EmitLLZKInCompute<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> {
    type Output;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output>;
}

impl<'a, 'ctx: 'sco, 'sco, F, T> EmitLLZKInCompute<'a, 'ctx, 'sco, F> for Vec<T>
where
    F: PrimeField + FieldInfo,
    T: EmitLLZKInCompute<'a, 'ctx, 'sco, F, Output = ()>,
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

/// Lowers one SSA block into LLZK ops while keeping a slot-per-subexpression cache.
///
/// This mirrors the existing Rust witness generator's "one raw expression, one SSA slot" model so
/// that indices coming from `SubExpression(..)` nodes continue to line up exactly.
struct ComputeLowering<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> {
    builder: &'a OpsBuilder<'ctx, 'sco, F>,
    vars: &'a StructVars,
    self_value: Value<'ctx, 'sco>,
    variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
    lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
    block: &'a [RawExpression<F>],
    slots: Vec<SsaSlot<'ctx, 'sco>>,
}

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> ComputeLowering<'a, 'ctx, 'sco, F> {
    /// Create a fresh lowering state for one SSA block.
    fn new(
        builder: &'a OpsBuilder<'ctx, 'sco, F>,
        vars: &'a StructVars,
        self_value: Value<'ctx, 'sco>,
        variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
        lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
        block: &'a [RawExpression<F>],
    ) -> Self {
        Self {
            builder,
            vars,
            self_value,
            variable_mapping,
            lookup_sets,
            block,
            slots: Vec::new(),
        }
    }

    /// Append one SSA slot to the block-local cache.
    fn push_slot(&mut self, slot: SsaSlot<'ctx, 'sco>) {
        self.slots.push(slot);
    }

    /// Materialize a write-back to a witness or scratch column.
    ///
    /// The SSA still contains writes to transient memory columns, but LLZK `@compute` currently
    /// only returns the witness struct. Those writes are intentionally ignored for now. This
    /// matches the `ColumnAddress::MemorySubtree` handling in
    /// `witness_eval_generator/src/derive_from_ssa/mod.rs::SSAGenerator::add_expression`.
    fn lower_write(
        &mut self,
        into_variable: &Variable,
        source_subexpr: &Expression<F>,
        condition_subexpr_idx: Option<usize>,
    ) -> Result<()> {
        match self.variable_mapping[into_variable] {
            // TODO(LLZK compute): model memory writes once `@compute` can materialize memory
            // side-effects in addition to the returned witness struct.
            ColumnAddress::MemorySubtree(..) => return Ok(()),
            ColumnAddress::SetupSubtree(..) => {
                bail!("setup columns are read-only during witness lowering")
            }
            ColumnAddress::WitnessSubtree(..) | ColumnAddress::OptimizedOut(..) => {}
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
    fn read_variable(&self, variable: Variable) -> Result<Value<'ctx, 'sco>> {
        self.vars
            .try_get_compute_val(self.builder, self.self_value, &variable)?
            .ok_or_else(|| {
                anyhow!(
                    "variable {variable:?} is not exposed to @compute (column {:?})",
                    self.variable_mapping.get(&variable)
                )
            })
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

    /// Resolve the table id referenced by a lookup SSA node.
    ///
    /// The witness placer currently materializes table ids as constant subexpressions. We inspect
    /// the original block so lookup lowering can choose a deterministic implementation before any
    /// MLIR values are emitted, and cross-check against the compiled lookup layout when the SSA
    /// node also carries a `lookup_mapping_idx`.
    fn resolve_lookup_table(
        &self,
        table_id_subexpr_idx: usize,
        lookup_mapping_idx: Option<usize>,
    ) -> Result<TableType> {
        let table_expr = self
            .block
            .get(table_id_subexpr_idx)
            .ok_or_else(|| anyhow!("SSA slot {table_id_subexpr_idx} is out of bounds"))?;
        let table_id = match table_expr {
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU8(value)) => {
                u32::from(*value)
            }
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU16(value)) => {
                u32::from(*value)
            }
            RawExpression::Integer(FixedWidthIntegerNodeExpression::ConstantU32(value)) => *value,
            RawExpression::Field(FieldNodeExpression::Constant(value)) => {
                value.as_u64_reduced().try_into()?
            }
            _ => bail!(
                "lookup table ids must lower from constant SSA expressions, found {:?}",
                table_expr
            ),
        };
        let table = TableType::get_table_from_id(table_id);

        if let Some(lookup_mapping_idx) = lookup_mapping_idx {
            let expected = self
                .lookup_sets
                .get(lookup_mapping_idx)
                .ok_or_else(|| anyhow!("lookup mapping {lookup_mapping_idx} is out of bounds"))?;
            match expected.table_index {
                TableIndex::Constant(expected_table) => {
                    if expected_table != table {
                        bail!(
                            "SSA lookup mapping {lookup_mapping_idx} expects table {:?}, found {:?}",
                            expected_table,
                            table
                        );
                    }
                }
                // TODO(LLZK compute): support dynamic table ids in witness SSA once a concrete
                // circuit uses them.
                TableIndex::Variable(column) => {
                    bail!(
                        "dynamic lookup table ids are not yet supported in @compute (column {:?})",
                        column
                    )
                }
            }
        }

        Ok(table)
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

    /// Allocate witness holes for lookup outputs that are not yet lowered deterministically.
    fn new_lookup_outputs(&self, num_outputs: usize) -> Result<Vec<Value<'ctx, 'sco>>> {
        // TODO(LLZK compute): replace these witness holes with deterministic lookup lowering once
        // the remaining table families are modeled in `@compute`.
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
}

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
    for RawExpression<F>
{
    type Output = ();

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        let slot = if let Some(lookup) = LookupInvocation::from_raw(self) {
            SsaSlot::Lookup(lookup.emit_compute(lowering)?)
        } else {
            match self {
                RawExpression::Bool(expr) => {
                    SsaSlot::Value(ComputedValue::Bool(expr.emit_compute(lowering)?))
                }
                RawExpression::Field(expr) => {
                    SsaSlot::Value(ComputedValue::Field(expr.emit_compute(lowering)?))
                }
                RawExpression::Integer(expr) => {
                    SsaSlot::Value(ComputedValue::Integer(expr.emit_compute(lowering)?))
                }
                RawExpression::AccessLookup {
                    subindex,
                    output_index,
                } => {
                    let lookup_value = *lowering
                        .slot_as_lookup(*subindex)?
                        .get(*output_index)
                        .ok_or_else(|| anyhow!("lookup output {output_index} is out of bounds"))?;
                    SsaSlot::Value(ComputedValue::Field(lookup_value))
                }
                RawExpression::WriteVariable {
                    into_variable,
                    source_subexpr,
                    condition_subexpr_idx,
                } => {
                    lowering.lower_write(into_variable, source_subexpr, *condition_subexpr_idx)?;
                    SsaSlot::Unit
                }
                RawExpression::PerformLookup { .. } | RawExpression::MaybePerformLookup { .. } => {
                    unreachable!("lookup raw expressions are handled by LookupInvocation")
                }
            }
        };

        lowering.push_slot(slot);
        Ok(())
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
    for LookupInvocation
{
    type Output = Vec<Value<'ctx, 'sco>>;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output> {
        let table = lowering
            .resolve_lookup_table(self.table_id_subexpr_idx(), self.lookup_mapping_idx())?;
        let inputs = lowering.lookup_inputs_as_fields(self.input_subexpr_idxes())?;

        let outputs = match table {
            TableType::ConditionalJmpBranchSlt => {
                lowering.compute_conditional_jmp_branch_slt_lookup(&inputs, self.num_outputs())?
            }
            TableType::JumpCleanupOffset => {
                lowering.compute_jump_cleanup_offset_lookup(&inputs, self.num_outputs())?
            }
            _ => {
                // TODO(LLZK compute): add deterministic lowering for the remaining lookup tables
                // used by the supported circuits.
                lowering.new_lookup_outputs(self.num_outputs())?
            }
        };

        if let Some(mask_id_subexpr_idx) = self.mask_id_subexpr_idx() {
            let mask = lowering.slot_as_bool(mask_id_subexpr_idx)?;
            lowering.mask_lookup_outputs(mask, outputs)
        } else {
            Ok(outputs)
        }
    }
}

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
    for Expression<F>
{
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

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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
            // TODO(LLZK compute): thread witness oracle values through the LLZK compute pipeline
            // once the backend has an explicit oracle interface.
            FieldNodeExpression::OracleValue { .. } => {
                bail!("oracle-backed witness expressions are not yet supported in @compute")
            }
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

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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
            // TODO(LLZK compute): thread witness oracle values through the LLZK compute pipeline
            // once the backend has an explicit oracle interface.
            BoolNodeExpression::OracleValue { .. } => {
                bail!("oracle-backed witness expressions are not yet supported in @compute")
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

impl<'a, 'ctx: 'sco, 'sco, F: PrimeField + FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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
            FixedWidthIntegerNodeExpression::U32OracleValue { .. }
            | FixedWidthIntegerNodeExpression::U16OracleValue { .. }
            | FixedWidthIntegerNodeExpression::U8OracleValue { .. } => {
                // TODO(LLZK compute): thread witness oracle values through the LLZK compute
                // pipeline once the backend has an explicit oracle interface.
                bail!("oracle-backed integer expressions are not yet supported in @compute")
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
