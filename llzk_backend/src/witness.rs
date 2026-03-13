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
use prover::common_constants;
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

/// Compact description of the aligned ROM table contents used by `AlignedRomRead`.
///
/// The current LLZK `@compute` lowering cannot materialize an external ROM table directly, so we
/// summarize the bytecode image as the most common opcode plus sparse per-index overrides. This
/// keeps the generated IR small for the padded bytecode images used by the existing extraction
/// flow while still preserving exact lookup results.
#[derive(Clone)]
struct AlignedRomImage {
    default_opcode: u32,
    overrides: Vec<(usize, u32)>,
}

impl AlignedRomImage {
    /// Summarize the ROM image into a default opcode and the indices that differ from it.
    fn from_words(words: &[u32]) -> Self {
        let mut counts = BTreeMap::new();
        for &opcode in words {
            *counts.entry(opcode).or_insert(0usize) += 1;
        }

        let default_opcode = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(opcode, _)| opcode)
            .unwrap_or(prover::cs::machine::UNIMP_OPCODE);

        let overrides = words
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, opcode)| (opcode != default_opcode).then_some((index, opcode)))
            .collect();

        Self {
            default_opcode,
            overrides,
        }
    }
}

/// Split a 32-bit opcode into the low/high 16-bit limbs used by the aligned ROM table.
fn opcode_limbs(opcode: u32) -> (u16, u16) {
    (opcode as u16, (opcode >> 16) as u16)
}

/// Bundles the metadata required to lower witness generation into LLZK `@compute`.
///
/// The compiled artifact tells us where every logical variable lives in the witness layout, while
/// the SSA blocks preserve the witness placer's evaluation order and conditional write structure.
pub(crate) struct WitnessComputation<F: FieldInfo> {
    compiled: CompiledCircuitArtifact<F>,
    ssa: Vec<Vec<RawExpression<F>>>,
    aligned_rom_image: AlignedRomImage,
}

impl<F: FieldInfo> WitnessComputation<F> {
    /// Create a new witness computation plan from the one-row compiler output and witness SSA.
    ///
    /// The bytecode image is captured here as well so `AlignedRomRead` can be lowered into
    /// deterministic LLZK without depending on an external runtime table.
    pub fn new(
        compiled: CompiledCircuitArtifact<F>,
        ssa: Vec<Vec<RawExpression<F>>>,
        bytecode: Vec<u32>,
    ) -> Self {
        Self {
            compiled,
            ssa,
            aligned_rom_image: AlignedRomImage::from_words(&bytecode),
        }
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
                &self.aligned_rom_image,
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
trait EmitLLZKInCompute<'a, 'ctx: 'sco, 'sco, F: FieldInfo> {
    type Output;

    fn emit_compute(
        &self,
        lowering: &mut ComputeLowering<'a, 'ctx, 'sco, F>,
    ) -> Result<Self::Output>;
}

impl<'a, 'ctx: 'sco, 'sco, F, T> EmitLLZKInCompute<'a, 'ctx, 'sco, F> for Vec<T>
where
    F: FieldInfo,
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
struct ComputeLowering<'a, 'ctx: 'sco, 'sco, F: FieldInfo> {
    builder: &'a OpsBuilder<'ctx, 'sco, F>,
    vars: &'a StructVars<F>,
    self_value: Value<'ctx, 'sco>,
    variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
    lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
    aligned_rom_image: &'a AlignedRomImage,
    block: &'a [RawExpression<F>],
    slots: Vec<SsaSlot<'ctx, 'sco>>,
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> ComputeLowering<'a, 'ctx, 'sco, F> {
    /// Create a fresh lowering state for one SSA block.
    fn new(
        builder: &'a OpsBuilder<'ctx, 'sco, F>,
        vars: &'a StructVars<F>,
        self_value: Value<'ctx, 'sco>,
        variable_mapping: &'a BTreeMap<Variable, ColumnAddress>,
        lookup_sets: &'a [LookupSetDescription<F, COMMON_TABLE_WIDTH>],
        aligned_rom_image: &'a AlignedRomImage,
        block: &'a [RawExpression<F>],
    ) -> Self {
        Self {
            builder,
            vars,
            self_value,
            variable_mapping,
            lookup_sets,
            aligned_rom_image,
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
            RawExpression::Field(FieldNodeExpression::Constant(value)) => Some(
                TableType::get_table_from_id(value.as_u64_reduced().try_into()?),
            ),
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

    /// Deterministically lower the ROM word lookup used by subword ROM loads.
    ///
    /// The aligned ROM table is keyed by word index and returns the 32-bit opcode split into two
    /// 16-bit limbs. We lower it by starting from the most common opcode in the captured bytecode
    /// image and then patching the indices whose opcode differs via index-equality selects.
    fn compute_aligned_rom_read_lookup(
        &self,
        inputs: &[Value<'ctx, 'sco>],
        num_outputs: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        if inputs.len() != 1 {
            bail!("AlignedRomRead expects 1 input, found {}", inputs.len());
        }

        let word_index = inputs[0];
        let (default_low, default_high) = opcode_limbs(self.aligned_rom_image.default_opcode);
        let mut low = self
            .builder
            .get_felt_constant_from_start(u64::from(default_low))?;
        let mut high = self
            .builder
            .get_felt_constant_from_start(u64::from(default_high))?;

        for &(index, opcode) in &self.aligned_rom_image.overrides {
            let is_selected = self.field_eq_constant(word_index, index as u64)?;
            let (opcode_low, opcode_high) = opcode_limbs(opcode);
            low = self.select_value(
                is_selected,
                self.builder
                    .get_felt_constant_from_start(u64::from(opcode_low))?,
                low,
            )?;
            high = self.select_value(
                is_selected,
                self.builder
                    .get_felt_constant_from_start(u64::from(opcode_high))?,
                high,
            )?;
        }

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
}

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F> for RawExpression<F> {
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

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F> for LookupInvocation {
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

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F> for Expression<F> {
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

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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

impl<'a, 'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInCompute<'a, 'ctx, 'sco, F>
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

    fn aligned_rom_read_outputs(bytecode: &[u32], word_index: usize) -> [u64; 2] {
        let (low, high) = opcode_limbs(
            bytecode
                .get(word_index)
                .copied()
                .unwrap_or(prover::cs::machine::UNIMP_OPCODE),
        );
        [u64::from(low), u64::from(high)]
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
    fn aligned_rom_lookup_semantics_match_table_driver() {
        let bytecode_words =
            (1 << (16 + load_store_subword_only::ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;
        let mut bytecode = vec![0u32; bytecode_words];
        bytecode[0] = 0x0000_0013;
        bytecode[1] = 0x0010_8093;
        bytecode[0x1234] = 0xfeed_beef;
        bytecode[bytecode_words - 1] = 0xc000_1073;

        let table_driver = load_store_subword_only::get_table_driver(&bytecode);

        for word_index in [0usize, 1, 2, 0x1234, 0x4321, bytecode_words - 1] {
            assert_eq!(
                lookup_values::<2>(
                    &table_driver,
                    TableType::AlignedRomRead,
                    &[word_index as u64]
                ),
                aligned_rom_read_outputs(&bytecode, word_index),
            );
        }
    }
}
