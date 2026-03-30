use super::*;
use crate::cs::circuit::DisjunctiveLookup;
use crate::types::{Boolean, Num};

const U16_BOUND: u64 = 1 << 16;
const U14_BOUND: u64 = 1 << 14;
const U8_BOUND: u64 = 1 << 8;
const U4_BOUND: u64 = 1 << 4;

/// Allocates `width` boolean indicators and constrains them to be one-hot.
///
/// This helper is used by table handlers that need an explicit one-hot
/// decomposition in PCL.
fn add_one_hot_bits(
    module: &mut PicusModule,
    width: usize,
    next_fresh_var_id: &mut usize,
) -> Vec<PicusExpr> {
    let mut bits = Vec::with_capacity(width);
    for _ in 0..width {
        let b = fresh_picus_var_expr(next_fresh_var_id);
        module.constraints.push(PicusConstraint::new_bit(b.clone()));
        bits.push(b);
    }

    let sum = bits
        .iter()
        .cloned()
        .fold(PicusExpr::Const(0), |acc, b| acc + b);
    module
        .constraints
        .push(PicusConstraint::new_equality(sum, PicusExpr::Const(1)));

    bits
}

/// Translation for `JumpCleanupOffset` lookup.
///
/// Table intent: decompose a low PC limb into aligned and bit components used
/// when cleaning up jump targets.
///
/// Extraction strategy: keep the arithmetic decomposition (`a = cleaned + 2*bit_1 + bit_0`)
/// and alignment relation (`cleaned = 4*k`) with appropriate bit/range guards.
fn add_jump_cleanup_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let a = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let bit_1 = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let cleaned = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    let bit_0 = fresh_picus_var_expr(next_fresh_var_id);
    let k = fresh_picus_var_expr(next_fresh_var_id);

    module
        .constraints
        .push(PicusConstraint::new_lt(a.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(cleaned.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_bit(bit_0.clone()));
    module
        .constraints
        .push(PicusConstraint::new_bit(bit_1.clone()));
    module
        .constraints
        .push(PicusConstraint::new_lt(k.clone(), U14_BOUND.into()));

    // a = cleaned + 2*bit_1 + bit_0
    module.constraints.push(PicusConstraint::new_equality(
        a,
        cleaned.clone() + PicusExpr::Const(2) * bit_1 + bit_0,
    ));
    // cleaned = 4*k
    module.constraints.push(PicusConstraint::new_equality(
        cleaned,
        PicusExpr::Const(4) * k,
    ));
}

/// Translation for `ConditionalJmpBranchSlt` lookup.
///
/// Table intent: resolve branch/SLT condition flags from packed condition inputs.
///
/// Extraction strategy: preserve the full logical/arithmetic encoding because the
/// table captures control-sensitive semantics that are not represented by a simple
/// determinism summary.
fn add_conditional_jmp_branch_slt_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let a = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let f3 = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let flag = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    let uf = fresh_picus_var_expr(next_fresh_var_id);
    let out_is_zero = fresh_picus_var_expr(next_fresh_var_id);
    let sign1 = fresh_picus_var_expr(next_fresh_var_id);
    let sign2 = fresh_picus_var_expr(next_fresh_var_id);

    module
        .constraints
        .push(PicusConstraint::new_lt(a.clone(), 16u64.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(f3.clone(), 8u64.into()));
    module
        .constraints
        .push(PicusConstraint::new_bit(uf.clone()));
    module
        .constraints
        .push(PicusConstraint::new_bit(out_is_zero.clone()));
    module
        .constraints
        .push(PicusConstraint::new_bit(sign1.clone()));
    module
        .constraints
        .push(PicusConstraint::new_bit(sign2.clone()));
    module
        .constraints
        .push(PicusConstraint::new_bit(flag.clone()));

    // a = uf + 2*out_is_zero + 4*sign1 + 8*sign2
    module.constraints.push(PicusConstraint::new_equality(
        a,
        uf.clone()
            + PicusExpr::Const(2) * out_is_zero.clone()
            + PicusExpr::Const(4) * sign1.clone()
            + PicusExpr::Const(8) * sign2.clone(),
    ));

    let signs_different =
        sign1.clone() + sign2.clone() - (PicusExpr::Const(2) * sign1.clone() * sign2.clone());
    let unsigned_lt = uf.clone();
    let signed_lt = sign1.clone() * signs_different.clone()
        + unsigned_lt.clone() * (PicusExpr::Const(1) - signs_different);
    let eq = out_is_zero;

    // one-hot for funct3
    let mut f3_one_hot = Vec::with_capacity(8);
    for _ in 0..8 {
        let b = fresh_picus_var_expr(next_fresh_var_id);
        module.constraints.push(PicusConstraint::new_bit(b.clone()));
        f3_one_hot.push(b);
    }
    let one_hot_sum = f3_one_hot
        .iter()
        .cloned()
        .fold(PicusExpr::Const(0), |acc, x| acc + x);
    module.constraints.push(PicusConstraint::new_equality(
        one_hot_sum,
        PicusExpr::Const(1),
    ));

    let f3_reconstructed = f3_one_hot
        .iter()
        .enumerate()
        .fold(PicusExpr::Const(0), |acc, (i, b)| {
            acc + PicusExpr::Const(i as u64) * b.clone()
        });
    module
        .constraints
        .push(PicusConstraint::new_equality(f3, f3_reconstructed));

    let expected_flag = f3_one_hot[0].clone() * eq.clone()
        + f3_one_hot[1].clone() * (PicusExpr::Const(1) - eq.clone())
        + f3_one_hot[2].clone() * signed_lt.clone()
        + f3_one_hot[3].clone() * unsigned_lt.clone()
        + f3_one_hot[4].clone() * signed_lt.clone()
        + f3_one_hot[5].clone() * (PicusExpr::Const(1) - signed_lt.clone())
        + f3_one_hot[6].clone() * unsigned_lt.clone()
        + f3_one_hot[7].clone() * (PicusExpr::Const(1) - unsigned_lt);
    module
        .constraints
        .push(PicusConstraint::new_equality(flag, expected_flag));
}

/// Translation for `MemoryGetOffsetAndMaskWithTrap` lookup.
///
/// Table intent: map packed memory access metadata to offset and trap/mask bits.
///
/// Extraction strategy: use a compact summary with input/output range bounds and a
/// determinism axiom `det(input) => (det(offset) && det(bitmask))`.
fn add_memory_get_offset_and_mask_with_trap_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    _next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let input = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let offset = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let bitmask = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    module
        .constraints
        .push(PicusConstraint::new_lt(input.clone(), (1u64 << 21).into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(offset.clone(), U4_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(bitmask.clone(), U8_BOUND.into()));

    // det(input) => (det(offset) && det(bitmask))
    let det_input = PicusConstraint::new_det(input);
    let det_offset = PicusConstraint::new_det(offset);
    let det_bitmask = PicusConstraint::new_det(bitmask);
    module.constraints.push(PicusConstraint::Implies(
        Box::new(det_input),
        Box::new(PicusConstraint::And(
            Box::new(det_offset),
            Box::new(det_bitmask),
        )),
    ));
}

/// Translation for `RomAddressSpaceSeparator` lookup.
///
/// Table intent: split a high address limb into `(is_ram_range, rom_chunk)`.
///
/// Extraction strategy: keep explicit decomposition constraints linking
/// `address_high`, `rom_chunk`, and `is_ram_range`.
fn add_rom_address_space_separator_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let address_high = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let is_ram_range = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let rom_chunk = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);
    let rom_bound = 1u64 << common_constants::ROM_SECOND_WORD_BITS;
    let q_bound = 1u64 << (16 - common_constants::ROM_SECOND_WORD_BITS);

    let q_tail = fresh_picus_var_expr(next_fresh_var_id);
    let q = fresh_picus_var_expr(next_fresh_var_id);

    module
        .constraints
        .push(PicusConstraint::new_lt(address_high.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_bit(is_ram_range.clone()));
    module
        .constraints
        .push(PicusConstraint::new_lt(rom_chunk.clone(), rom_bound.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(q_tail.clone(), q_bound.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(q.clone(), q_bound.into()));

    // Address decomposition by ROM chunk size.
    module.constraints.push(PicusConstraint::new_equality(
        address_high,
        rom_chunk + PicusExpr::Const(rom_bound) * q.clone(),
    ));
    // Link is_ram_range to q != 0 by construction:
    // q = is_ram_range * (q_tail + 1).
    module.constraints.push(PicusConstraint::new_equality(
        q,
        is_ram_range * (q_tail + PicusExpr::Const(1)),
    ));
}

/// Translation for `MemoryLoadHalfwordOrByte` lookup.
///
/// Table intent: compute the `(low, high)` loaded value limbs for subword loads.
///
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(out_low) && det(out_high))`.
fn add_memory_load_halfword_or_byte_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    _next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let input = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let out_low = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let out_high = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    module
        .constraints
        .push(PicusConstraint::new_lt(input.clone(), (1u64 << (16 + 2 + 3)).into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(out_low.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(out_high.clone(), U16_BOUND.into()));

    let det_input = PicusConstraint::new_det(input);
    let det_out_low = PicusConstraint::new_det(out_low);
    let det_out_high = PicusConstraint::new_det(out_high);
    module.constraints.push(PicusConstraint::Implies(
        Box::new(det_input),
        Box::new(PicusConstraint::And(
            Box::new(det_out_low),
            Box::new(det_out_high),
        )),
    ));
}

/// Translation for `MemStoreClearOriginalRamValueLimb` lookup.
///
/// Table intent: clear the relevant bytes in the original RAM limb before merge.
///
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(cleaned) && det(unused))`.
fn add_mem_store_clear_original_ram_value_limb_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    _next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let input = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let cleaned = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let unused = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    module
        .constraints
        .push(PicusConstraint::new_lt(input.clone(), (1u64 << (16 + 2 + 3)).into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(cleaned.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(unused.clone(), U16_BOUND.into()));

    // det(input) => (det(cleaned) && det(unused))
    let det_input = PicusConstraint::new_det(input);
    let det_cleaned = PicusConstraint::new_det(cleaned);
    let det_unused = PicusConstraint::new_det(unused);
    module.constraints.push(PicusConstraint::Implies(
        Box::new(det_input),
        Box::new(PicusConstraint::And(
            Box::new(det_cleaned),
            Box::new(det_unused),
        )),
    ));
}

/// Translation for `MemStoreClearWrittenValueLimb` lookup.
///
/// Table intent: normalize/position the written value limb before merging into RAM.
///
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(cleaned) && det(unused))`.
fn add_mem_store_clear_written_value_limb_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    _next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    let input = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let cleaned = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let unused = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    module
        .constraints
        .push(PicusConstraint::new_lt(input.clone(), (1u64 << (16 + 2 + 3)).into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(cleaned.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(unused.clone(), U16_BOUND.into()));

    // det(input) => (det(cleaned) && det(unused))
    let det_input = PicusConstraint::new_det(input);
    let det_cleaned = PicusConstraint::new_det(cleaned);
    let det_unused = PicusConstraint::new_det(unused);
    module.constraints.push(PicusConstraint::Implies(
        Box::new(det_input),
        Box::new(PicusConstraint::And(
            Box::new(det_cleaned),
            Box::new(det_unused),
        )),
    ));
}

/// Translation for `AlignedRomRead` lookup.
///
/// Table intent: map ROM word index to low/high 16-bit instruction limbs.
///
/// Extraction strategy: bound index/output ranges and enforce determinism
/// `det(word_index) => (det(low) && det(high))`.
fn add_aligned_rom_read_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    row_multiplier: Option<&PicusExpr>,
) {
    let word_index = lookup_input_to_picus_expr_with_multiplier(&query.row[0], row_multiplier);
    let low = lookup_input_to_picus_expr_with_multiplier(&query.row[1], row_multiplier);
    let high = lookup_input_to_picus_expr_with_multiplier(&query.row[2], row_multiplier);

    // Aligned ROM table is keyed by word index in [0, 2^(16 + ROM_SECOND_WORD_BITS - 2)).
    let max_word_index = 1u64 << (16 + common_constants::ROM_SECOND_WORD_BITS - 2);
    module
        .constraints
        .push(PicusConstraint::new_lt(word_index.clone(), max_word_index.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(low.clone(), U16_BOUND.into()));
    module
        .constraints
        .push(PicusConstraint::new_lt(high.clone(), U16_BOUND.into()));

    // det(word_index) => (det(low) && det(high))
    let det_word_index = PicusConstraint::new_det(word_index);
    let det_low = PicusConstraint::new_det(low);
    let det_high = PicusConstraint::new_det(high);
    module.constraints.push(PicusConstraint::Implies(
        Box::new(det_word_index),
        Box::new(PicusConstraint::And(Box::new(det_low), Box::new(det_high))),
    ));

    // NOTE: exact (word_index -> low/high) value linkage still requires
    // embedding the concrete ROM table contents.
}

/// Converts a lookup input into a Picus expression and optionally scales it by a
/// row multiplier (used for flag-multiplied disjunctive encodings).
fn lookup_input_to_picus_expr_with_multiplier<F: PrimeField>(
    input: &crate::definitions::LookupInput<F>,
    row_multiplier: Option<&PicusExpr>,
) -> PicusExpr {
    let expr = lookup_input_to_picus_expr(input);
    if let Some(multiplier) = row_multiplier {
        multiplier.clone() * expr
    } else {
        expr
    }
}

/// Converts circuit boolean flavor (`Is`, `Not`, `Constant`) into a Picus expression.
fn boolean_to_picus_expr(flag: Boolean) -> PicusExpr {
    match flag {
        Boolean::Is(v) => variable_to_picus_expr(v),
        Boolean::Not(v) => PicusExpr::Const(1) - variable_to_picus_expr(v),
        Boolean::Constant(c) => PicusExpr::Const(c as u64),
    }
}

/// Returns true when row-multiplication by an activation flag is safe for the table.
///
/// Safety criterion here is that the multiplied inactive row still corresponds to a
/// valid table behavior for the summarization strategy.
fn table_supports_zero_row_multiply_in(table: TableType) -> bool {
    matches!(
        table,
        TableType::MemoryLoadHalfwordOrByte | TableType::MemStoreClearOriginalRamValueLimb
    )
}

/// Dispatches one constant-table lookup query to its table-specific translator.
///
/// `row_multiplier` is used for the "multiply-in" disjunctive variant when enabled.
fn add_lookup_constraints_for_table<F: PrimeField>(
    module: &mut PicusModule,
    query: &LookupQuery<F>,
    table: TableType,
    next_fresh_var_id: &mut usize,
    row_multiplier: Option<&PicusExpr>,
) {
    match table {
        TableType::ConditionalJmpBranchSlt => {
            add_conditional_jmp_branch_slt_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::JumpCleanupOffset => {
            add_jump_cleanup_lookup_constraints(module, query, next_fresh_var_id, row_multiplier);
        }
        TableType::MemoryGetOffsetAndMaskWithTrap => {
            add_memory_get_offset_and_mask_with_trap_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::RomAddressSpaceSeparator => {
            add_rom_address_space_separator_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::MemoryLoadHalfwordOrByte => {
            add_memory_load_halfword_or_byte_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::MemStoreClearOriginalRamValueLimb => {
            add_mem_store_clear_original_ram_value_limb_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::MemStoreClearWrittenValueLimb => {
            add_mem_store_clear_written_value_limb_lookup_constraints(
                module,
                query,
                next_fresh_var_id,
                row_multiplier,
            );
        }
        TableType::AlignedRomRead => {
            add_aligned_rom_read_lookup_constraints(module, query, row_multiplier);
        }
        _ => {}
    }
}

/// Adds translated constraints for regular (non-disjunctive) constant-table lookups.
pub(super) fn add_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    lookups: &[LookupQuery<F>],
    next_fresh_var_id: &mut usize,
) {
    for query in lookups {
        let LookupQueryTableType::Constant(table) = query.table else {
            continue;
        };

        add_lookup_constraints_for_table(module, query, table, next_fresh_var_id, None);
    }
}

/// Adds translated constraints for disjunctive lookup metadata emitted by optimization context.
///
/// For each disjunctive relation this function:
/// - adds postconditions that flags are boolean and satisfy `sum(flags) <= 1`,
/// - dispatches each case through the same table translator,
/// - either multiplies row expressions by flag for safe tables or guards case
///   constraints under `(flag = 1) => ...`.
pub(super) fn add_disjunctive_lookup_constraints<F: PrimeField>(
    module: &mut PicusModule,
    disjunctive_lookups: &[DisjunctiveLookup<F>],
    next_fresh_var_id: &mut usize,
) {
    for relation in disjunctive_lookups {
        let flags: Vec<PicusExpr> = relation
            .cases
            .iter()
            .map(|case| boolean_to_picus_expr(case.flag))
            .collect();

        for flag in &flags {
            module
                .postconditions
                .push(PicusConstraint::new_bit(flag.clone()));
        }
        let flag_sum = flags
            .iter()
            .cloned()
            .fold(PicusExpr::Const(0), |acc, f| acc + f);
        module
            .postconditions
            .push(PicusConstraint::new_leq(flag_sum, PicusExpr::Const(1)));

        for case in &relation.cases {
            let Num::Constant(table_id) = case.table else {
                continue;
            };
            let table = TableType::get_table_from_id(table_id.as_u64_reduced() as u32);
            let query = LookupQuery {
                row: case.row.clone(),
                table: LookupQueryTableType::Constant(table),
            };
            let flag_expr = boolean_to_picus_expr(case.flag);

            if table_supports_zero_row_multiply_in(table) {
                add_lookup_constraints_for_table(
                    module,
                    &query,
                    table,
                    next_fresh_var_id,
                    Some(&flag_expr),
                );
            } else {
                let base = module.constraints.len();
                add_lookup_constraints_for_table(module, &query, table, next_fresh_var_id, None);
                let added = module.constraints.split_off(base);
                let cond = PicusConstraint::new_equality(flag_expr, PicusExpr::Const(1));
                for c in added {
                    module.constraints.push(PicusConstraint::Implies(
                        Box::new(cond.clone()),
                        Box::new(c),
                    ));
                }
            }
        }
    }
}
