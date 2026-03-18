//! Encodings of lookup tables into LLZK.

use crate::builder::OpsBuilder;
use crate::codegen::StructVars;
use crate::constraints::EmitLlzkInConstrain as _;
use crate::field::FieldInfo;
use anyhow::Result;
use llzk::dialect::felt;
use melior::ir::Value;
use prover::common_constants;
use prover::cs::cs::circuit::DisjunctiveLookup;
use prover::cs::cs::circuit::LookupQuery;
use prover::cs::cs::circuit::LookupQueryTableType;
use prover::cs::tables::TableType;
use prover::cs::types::Num;

/// Add constraints that the LookupQuery represents based on the parsed table type.
/// If `conditional` is specified, then all generated constraints will be implications
/// based on `conditional`.
pub fn add_lookup_constraints_for_table<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    table: TableType,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    match table {
        TableType::ConditionalJmpBranchSlt => add_conditional_jmp_branch_slt_lookup_constraints(
            builder,
            vars,
            query,
            row_multiplier,
            conditional,
        ),
        TableType::JumpCleanupOffset => {
            add_jump_cleanup_lookup_constraints(builder, vars, query, row_multiplier, conditional)
        }
        TableType::MemoryGetOffsetAndMaskWithTrap => {
            add_memory_get_offset_and_mask_with_trap_lookup_constraints(
                builder,
                vars,
                query,
                row_multiplier,
                conditional,
            )
        }
        TableType::RomAddressSpaceSeparator => add_rom_address_space_separator_lookup_constraints(
            builder,
            vars,
            query,
            row_multiplier,
            conditional,
        ),
        TableType::MemoryLoadHalfwordOrByte => add_memory_load_halfword_or_byte_lookup_constraints(
            builder,
            vars,
            query,
            row_multiplier,
            conditional,
        ),
        TableType::MemStoreClearOriginalRamValueLimb => {
            add_mem_store_clear_original_ram_value_limb_lookup_constraints(
                builder,
                vars,
                query,
                row_multiplier,
                conditional,
            )
        }
        TableType::MemStoreClearWrittenValueLimb => {
            add_mem_store_clear_written_value_limb_lookup_constraints(
                builder,
                vars,
                query,
                row_multiplier,
                conditional,
            )
        }
        TableType::AlignedRomRead => add_aligned_rom_read_lookup_constraints(
            builder,
            vars,
            query,
            row_multiplier,
            conditional,
        ),
        _ => panic!("unsupported lookup table in LLZK lookup lowering: {table:#?}"),
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

/// Adds translated constraints for disjunctive lookup metadata emitted by optimization context.
///
/// For each disjunctive relation this function:
/// - adds postconditions that flags are boolean and satisfy `sum(flags) <= 1`,
/// - dispatches each case through the same table translator,
/// - either multiplies row expressions by flag for safe tables or guards case constraints under
///   `(flag = 1) => ...`.
pub fn add_disjunctive_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    relation: &DisjunctiveLookup<F>,
) -> Result<()> {
    let flags = relation
        .cases
        .iter()
        .map(|case| case.flag.emit_constrain(builder, vars))
        .collect::<Result<Vec<Value<'ctx, 'sco>>>>()?;

    for flag in &flags {
        builder.append_boolean_constraint(*flag)?;
    }
    let flag_sum = builder.append_sum(builder.unknown_location(), &flags)?;
    // flag_sum <= 1, meaning flag_sum must be boolean
    builder.append_boolean_constraint(flag_sum)?;

    for case in &relation.cases {
        let table_id = match case.table {
            Num::Var(_variable) => {
                panic!("variable table ids in disjunctive lookup queries are not yet supported")
            }
            Num::Constant(table_id) => table_id,
        };

        let table = TableType::get_table_from_id(table_id.as_u64_reduced() as u32);
        let query = LookupQuery {
            row: case.row.clone(),
            table: LookupQueryTableType::Constant(table),
        };
        let flag_expr = case.flag.emit_constrain(builder, vars)?;

        if table_supports_zero_row_multiply_in(table) {
            add_lookup_constraints_for_table(builder, vars, &query, table, Some(flag_expr), None)?;
        } else {
            // The lookup constraints in these tables are conditional on the flag_expr.
            add_lookup_constraints_for_table(builder, vars, &query, table, None, Some(flag_expr))?;
        }
    }
    Ok(())
}

/// Multiplies the value by the row_multiplier if one is provided, otherwise
/// yield the original value.
fn apply_row_multiplier<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    val: Value<'ctx, 'sco>,
) -> Result<Value<'ctx, 'sco>> {
    match row_multiplier {
        Some(coeff) => {
            let op = felt::mul(builder.unknown_location(), coeff, val)?;
            builder.append_op_with_result(op)
        }
        None => Ok(val),
    }
}

/// Translation for `JumpCleanupOffset` lookup.
///
/// Table generation: [`prover::cs::tables::jump_opcode_related::create_jump_cleanup_offset_table`]
///
/// Table intent: decompose a low PC limb into aligned and bit components used
/// when cleaning up jump targets.
///
/// Extraction strategy: keep the arithmetic decomposition (`a = cleaned + 2*bit_1 + bit_0`)
/// and alignment relation (`cleaned = 4*k`) with appropriate bit/range guards.
fn add_jump_cleanup_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let a = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let bit_1 = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let cleaned = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    let bit_0 = builder.new_nondet_felt()?;
    let k = builder.new_nondet_felt()?;

    builder.append_conditional_range_constraint(conditional, a, 16)?;
    builder.append_conditional_range_constraint(conditional, cleaned, 16)?;
    builder.append_conditional_boolean_constraint(conditional, bit_0)?;
    builder.append_conditional_boolean_constraint(conditional, bit_1)?;
    builder.append_conditional_range_constraint(conditional, k, 14)?;

    // a = cleaned + 2*bit_1 + bit_0
    //      2*bit_1
    let bit_1_mul = builder.append_op_with_result(felt::mul(
        builder.unknown_location(),
        builder.get_constant_from_start(builder.felt_type(), 2)?,
        bit_1,
    )?)?;
    //      2*bit_1 + bit_0
    let twit =
        builder.append_op_with_result(felt::add(builder.unknown_location(), bit_1_mul, bit_0)?)?;
    //      cleaned + 2*bit_1 + bit_0
    let a_computed =
        builder.append_op_with_result(felt::add(builder.unknown_location(), cleaned, twit)?)?;
    builder.append_conditional_constrain_eq(
        builder.unknown_location(),
        conditional,
        a,
        a_computed,
    )?;

    // cleaned = 4*k
    builder.append_conditional_constrain_eq(
        builder.unknown_location(),
        conditional,
        cleaned,
        builder.append_op_with_result(felt::mul(
            builder.unknown_location(),
            builder.get_constant_from_start(builder.felt_type(), 4)?,
            k,
        )?)?,
    )
}

/// Translation for `ConditionalJmpBranchSlt` lookup.
///
/// Table generation:
/// [`prover::cs::tables::branch_opcode_related::create_conditional_jmp_branch_slt_family_resolution_table`]
///
/// Table intent: resolve branch/SLT condition flags from packed condition inputs.
///
/// Extraction strategy: preserve the full logical/arithmetic encoding because the
/// table captures control-sensitive semantics that are not represented by a simple
/// determinism summary.
fn add_conditional_jmp_branch_slt_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let a = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let f3 = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let flag = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    let uf = builder.new_nondet_felt()?;
    let out_is_zero = builder.new_nondet_felt()?;
    let sign1 = builder.new_nondet_felt()?;
    let sign2 = builder.new_nondet_felt()?;

    builder.append_conditional_range_constraint(conditional, a, 4)?; // a < 16
    builder.append_conditional_range_constraint(conditional, f3, 3)?; // f3 < 8
    builder.append_conditional_boolean_constraint(conditional, uf)?;
    builder.append_conditional_boolean_constraint(conditional, out_is_zero)?;
    builder.append_conditional_boolean_constraint(conditional, sign1)?;
    builder.append_conditional_boolean_constraint(conditional, sign2)?;
    builder.append_conditional_boolean_constraint(conditional, flag)?;

    // a = uf + 2*out_is_zero + 4*sign1 + 8*sign2
    builder.append_conditional_constrain_eq(
        builder.unknown_location(),
        conditional,
        a,
        builder.append_sum(
            builder.unknown_location(),
            &[
                uf,
                builder.append_const_scaling(builder.unknown_location(), 2, out_is_zero)?,
                builder.append_const_scaling(builder.unknown_location(), 4, sign1)?,
                builder.append_const_scaling(builder.unknown_location(), 8, sign2)?,
            ],
        )?,
    )?;

    // signs_different = sign1 + sign2 - (2 * sign1 * sign2)
    let signs_different = builder.append_sum(
        builder.unknown_location(),
        &[
            sign1,
            sign2,
            builder.append_op_with_result(felt::neg(
                builder.unknown_location(),
                builder.append_product(
                    builder.unknown_location(),
                    &[builder.get_felt_constant_from_start(2)?, sign1, sign2],
                )?,
            )?)?,
        ],
    )?;
    let unsigned_lt = uf;
    // signed_lt = (sign1 * signs_different) + unsigned_lt * (1 - signs_different);
    let signed_lt = builder.append_op_with_result(felt::add(
        builder.unknown_location(),
        builder.append_product(builder.unknown_location(), &[sign1, signs_different])?,
        builder.append_product(
            builder.unknown_location(),
            &[
                unsigned_lt,
                builder.append_op_with_result(felt::sub(
                    builder.unknown_location(),
                    builder.get_felt_constant_from_start(1)?,
                    signs_different,
                )?)?,
            ],
        )?,
    )?)?;
    let eq = out_is_zero;

    // one-hot for funct3
    let f3_one_hot = builder.append_one_hot(builder.unknown_location(), 8)?;
    let f3_reconstructed =
        builder.append_one_hot_reconstruction(builder.unknown_location(), &f3_one_hot)?;
    builder.append_conditional_constrain_eq(
        builder.unknown_location(),
        conditional,
        f3,
        f3_reconstructed,
    )?;

    // expected_flag =
    //       f3_one_hot[0] * eq
    //     + f3_one_hot[1] * (1 - eq)
    //     + f3_one_hot[2] * signed_lt
    //     + f3_one_hot[3] * unsigned_lt
    //     + f3_one_hot[4] * signed_lt
    //     + f3_one_hot[5] * (1 - signed_lt)
    //     + f3_one_hot[6] * unsigned_lt
    //     + f3_one_hot[7] * (1 - unsigned_lt);

    let one_minus = |v| -> Result<Value<'ctx, 'sco>> {
        builder.append_op_with_result(felt::sub(
            builder.unknown_location(),
            builder.get_felt_constant_from_start(1)?,
            v,
        )?)
    };

    let expected_flag = builder.append_sum(
        builder.unknown_location(),
        &[
            builder.append_product(builder.unknown_location(), &[f3_one_hot[0], eq])?,
            builder.append_product(builder.unknown_location(), &[f3_one_hot[1], one_minus(eq)?])?,
            builder.append_product(builder.unknown_location(), &[f3_one_hot[2], signed_lt])?,
            builder.append_product(builder.unknown_location(), &[f3_one_hot[3], unsigned_lt])?,
            builder.append_product(builder.unknown_location(), &[f3_one_hot[4], signed_lt])?,
            builder.append_product(
                builder.unknown_location(),
                &[f3_one_hot[5], one_minus(signed_lt)?],
            )?,
            builder.append_product(builder.unknown_location(), &[f3_one_hot[6], unsigned_lt])?,
            builder.append_product(
                builder.unknown_location(),
                &[f3_one_hot[7], one_minus(unsigned_lt)?],
            )?,
        ],
    )?;
    // flag === expected_flag
    builder.append_conditional_constrain_eq(
        builder.unknown_location(),
        conditional,
        flag,
        expected_flag,
    )?;
    Ok(())
}

/// Translation for `MemoryGetOffsetAndMaskWithTrap` lookup.
///
/// Table intent: map packed memory access metadata to offset and trap/mask bits.
///
/// TODO:
/// Extraction strategy: use a compact summary with input/output range bounds and a
/// determinism axiom `det(input) => (det(offset) && det(bitmask))`.
fn add_memory_get_offset_and_mask_with_trap_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let input = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let offset = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let bitmask = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    builder.append_conditional_range_constraint(conditional, input, 21)?;
    builder.append_conditional_range_constraint(conditional, offset, 4)?;
    builder.append_conditional_range_constraint(conditional, bitmask, 8)?;

    // TODO: No way to encode this in LLZK
    // det(input) => (det(offset) && det(bitmask))
    // let det_input = PicusConstraint::new_det(input);
    // let det_offset = PicusConstraint::new_det(offset);
    // let det_bitmask = PicusConstraint::new_det(bitmask);
    // module.constraints.push(PicusConstraint::Implies(
    // Box::new(det_input),
    // Box::new(PicusConstraint::And(
    // Box::new(det_offset),
    // Box::new(det_bitmask),
    // )),
    // ));
    Ok(())
}

/// Translation for `RomAddressSpaceSeparator` lookup.
///
/// Table intent: split a high address limb into `(is_ram_range, rom_chunk)`.
///
/// Extraction strategy: keep explicit decomposition constraints linking
/// `address_high`, `rom_chunk`, and `is_ram_range`.
fn add_rom_address_space_separator_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let address_high = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let is_ram_range = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let rom_chunk = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    let rom_bound = 1u64 << common_constants::ROM_SECOND_WORD_BITS;

    let q_tail = builder.new_nondet_felt()?;
    let q = builder.new_nondet_felt()?;

    builder.append_conditional_range_constraint(conditional, address_high, 16)?;
    builder.append_conditional_boolean_constraint(conditional, is_ram_range)?;
    builder.append_conditional_range_constraint(
        conditional,
        rom_chunk,
        common_constants::ROM_SECOND_WORD_BITS,
    )?;
    builder.append_conditional_range_constraint(
        conditional,
        q_tail,
        16 - common_constants::ROM_SECOND_WORD_BITS,
    )?;
    builder.append_conditional_range_constraint(
        conditional,
        q_tail,
        16 - common_constants::ROM_SECOND_WORD_BITS,
    )?;

    let location = builder.unknown_location();
    // Address decomposition by ROM chunk size.
    // address_high = rom_chunk + (rom_bound * q)
    builder.append_conditional_constrain_eq(
        location,
        conditional,
        address_high,
        builder.append_sum(
            location,
            &[
                rom_chunk,
                builder.append_product(
                    location,
                    &[builder.get_felt_constant_from_start(rom_bound)?, q],
                )?,
            ],
        )?,
    )?;
    // Link is_ram_range to q != 0 by construction:
    // q = is_ram_range * (q_tail + 1).
    builder.append_conditional_constrain_eq(
        location,
        conditional,
        q,
        builder.append_product(
            location,
            &[
                is_ram_range,
                builder.append_sum(
                    location,
                    &[builder.get_felt_constant_from_start(1)?, q_tail],
                )?,
            ],
        )?,
    )?;
    Ok(())
}

/// Translation for `MemoryLoadHalfwordOrByte` lookup.
///
/// Table intent: compute the `(low, high)` loaded value limbs for subword loads.
///
/// TODO:
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(out_low) && det(out_high))`.
fn add_memory_load_halfword_or_byte_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let input = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let out_low = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let out_high = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    builder.append_conditional_range_constraint(conditional, input, 16 + 2 + 3)?;
    builder.append_conditional_range_constraint(conditional, out_low, 16)?;
    builder.append_conditional_range_constraint(conditional, out_high, 16)?;

    // TODO: Port to LLZK
    // let det_input = PicusConstraint::new_det(input);
    // let det_out_low = PicusConstraint::new_det(out_low);
    // let det_out_high = PicusConstraint::new_det(out_high);
    // module.constraints.push(PicusConstraint::Implies(
    //     Box::new(det_input),
    //     Box::new(PicusConstraint::And(
    //         Box::new(det_out_low),
    //         Box::new(det_out_high),
    //     )),
    // ));
    Ok(())
}

/// Translation for `MemStoreClearOriginalRamValueLimb` lookup.
///
/// Table intent: clear the relevant bytes in the original RAM limb before merge.
///
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(cleaned) && det(unused))`.
fn add_mem_store_clear_original_ram_value_limb_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let input = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let cleaned = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let unused = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    builder.append_conditional_range_constraint(conditional, input, 16 + 2 + 3)?;
    builder.append_conditional_range_constraint(conditional, cleaned, 16)?;
    builder.append_conditional_range_constraint(conditional, unused, 16)?;

    // det(input) => (det(cleaned) && det(unused))
    // let det_input = PicusConstraint::new_det(input);
    // let det_cleaned = PicusConstraint::new_det(cleaned);
    // let det_unused = PicusConstraint::new_det(unused);
    // module.constraints.push(PicusConstraint::Implies(
    //     Box::new(det_input),
    //     Box::new(PicusConstraint::And(
    //         Box::new(det_cleaned),
    //         Box::new(det_unused),
    //     )),
    // ));
    Ok(())
}

/// Translation for `MemStoreClearWrittenValueLimb` lookup.
///
/// Table intent: normalize/position the written value limb before merging into RAM.
///
/// TODO:
/// Extraction strategy: summarize with range bounds and determinism
/// `det(input) => (det(cleaned) && det(unused))`.
fn add_mem_store_clear_written_value_limb_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let input = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let cleaned = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let unused = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    builder.append_conditional_range_constraint(conditional, input, 16 + 2 + 3)?;
    builder.append_conditional_range_constraint(conditional, cleaned, 16)?;
    builder.append_conditional_range_constraint(conditional, unused, 16)?;

    // det(input) => (det(cleaned) && det(unused))
    // let det_input = PicusConstraint::new_det(input);
    // let det_cleaned = PicusConstraint::new_det(cleaned);
    // let det_unused = PicusConstraint::new_det(unused);
    // module.constraints.push(PicusConstraint::Implies(
    //     Box::new(det_input),
    //     Box::new(PicusConstraint::And(
    //         Box::new(det_cleaned),
    //         Box::new(det_unused),
    //     )),
    // ));
    Ok(())
}

/// Translation for `AlignedRomRead` lookup.
///
/// Table intent: map ROM word index to low/high 16-bit instruction limbs.
///
/// TODO:
/// Extraction strategy: bound index/output ranges and enforce determinism
/// `det(word_index) => (det(low) && det(high))`.
fn add_aligned_rom_read_lookup_constraints<'ctx, 'sco, F: FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco, F>,
    vars: &StructVars<F>,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
    conditional: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let word_index = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_constrain(builder, vars)?,
    )?;
    let low = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_constrain(builder, vars)?,
    )?;
    let high = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_constrain(builder, vars)?,
    )?;

    // Aligned ROM table is keyed by word index in [0, 2^(16 + ROM_SECOND_WORD_BITS - 2)).
    builder.append_conditional_range_constraint(
        conditional,
        word_index,
        16 + common_constants::ROM_SECOND_WORD_BITS - 2,
    )?;
    builder.append_conditional_range_constraint(conditional, low, 16)?;
    builder.append_conditional_range_constraint(conditional, high, 16)?;

    // det(word_index) => (det(low) && det(high))
    // let det_word_index = PicusConstraint::new_det(word_index);
    // let det_low = PicusConstraint::new_det(low);
    // let det_high = PicusConstraint::new_det(high);
    // module.constraints.push(PicusConstraint::Implies(
    //     Box::new(det_word_index),
    //     Box::new(PicusConstraint::And(Box::new(det_low), Box::new(det_high))),
    // ));

    // NOTE: exact (word_index -> low/high) value linkage still requires
    // embedding the concrete ROM table contents.
    Ok(())
}

#[cfg(test)]
mod tests {
    use prover::cs::cs::circuit::DisjunctiveLookupCase;
    use prover::cs::cs::circuit::LookupQueryTableType;
    use prover::cs::definitions::Variable;
    use prover::cs::one_row_compiler::LookupInput;
    use prover::cs::types::Boolean;
    use prover::field::Mersenne31Field;

    use super::*;
    use crate::test_helpers::assert_full_ir_eq;
    use crate::test_helpers::emit_test_constrain_ir;

    /// Create a [`LookupQuery`] for the given `row` in the given `table`.
    fn direct_lookup_query(table: TableType, row: [Variable; 3]) -> LookupQuery<Mersenne31Field> {
        LookupQuery {
            row: row.map(LookupInput::from),
            table: LookupQueryTableType::Constant(table),
        }
    }

    /// Generate an exact-fixture test for a direct constant-table lookup.
    ///
    /// Each generated test uses the standard three-column synthetic row and checks the emitted
    /// `@constrain` IR against the provided fixture.
    macro_rules! direct_lookup_fixture_test {
        ($test_name:ident, $table:ident, $fixture:literal) => {
            #[test]
            fn $test_name() {
                let row = [Variable(7), Variable(8), Variable(9)];
                let table = TableType::$table;
                let query = direct_lookup_query(table, row);
                let ir = emit_test_constrain_ir("lookup_test", &row, &[], |ops, vars| {
                    add_lookup_constraints_for_table(ops, vars, &query, table, None, None)
                });
                assert_full_ir_eq(&ir, include_str!($fixture));
            }
        };
    }

    /// Generate an exact-fixture test for a one-case disjunctive lookup relation.
    ///
    /// Each generated test uses one boolean guard plus a three-column row and verifies the full
    /// guarded lookup encoding against the provided fixture.
    macro_rules! disjunctive_lookup_fixture_test {
        ($test_name:ident, $table:ident, $fixture:literal) => {
            #[test]
            fn $test_name() {
                let flag = Variable(7);
                let row = [Variable(8), Variable(9), Variable(10)];
                let relation = DisjunctiveLookup {
                    relation_index: 0,
                    cases: vec![DisjunctiveLookupCase {
                        flag: Boolean::Is(flag),
                        row: row.map(LookupInput::from),
                        table: TableType::$table.to_num(),
                    }],
                };
                let ir = emit_test_constrain_ir(
                    "lookup_test",
                    &[flag, row[0], row[1], row[2]],
                    &[],
                    |ops, vars| add_disjunctive_lookup_constraints(ops, vars, &relation),
                );
                assert_full_ir_eq(&ir, include_str!($fixture));
            }
        };
    }

    direct_lookup_fixture_test!(
        jump_cleanup_lookup_emits_alignment_constraints,
        JumpCleanupOffset,
        "../testdata/lookups/jump_cleanup_lookup_emits_alignment_constraints.mlir"
    );
    direct_lookup_fixture_test!(
        conditional_jump_lookup_emits_one_hot_logic,
        ConditionalJmpBranchSlt,
        "../testdata/lookups/conditional_jump_lookup_emits_one_hot_logic.mlir"
    );
    direct_lookup_fixture_test!(
        memory_get_offset_and_mask_lookup_is_range_only,
        MemoryGetOffsetAndMaskWithTrap,
        "../testdata/lookups/memory_get_offset_and_mask_lookup_is_range_only.mlir"
    );
    direct_lookup_fixture_test!(
        rom_address_space_separator_lookup_emits_rom_bound_relation,
        RomAddressSpaceSeparator,
        "../testdata/lookups/rom_address_space_separator_lookup_emits_rom_bound_relation.mlir"
    );
    direct_lookup_fixture_test!(
        memory_load_halfword_or_byte_lookup_is_range_only,
        MemoryLoadHalfwordOrByte,
        "../testdata/lookups/memory_load_halfword_or_byte_lookup_is_range_only.mlir"
    );
    direct_lookup_fixture_test!(
        mem_store_clear_original_lookup_is_range_only,
        MemStoreClearOriginalRamValueLimb,
        "../testdata/lookups/mem_store_clear_original_lookup_is_range_only.mlir"
    );
    direct_lookup_fixture_test!(
        mem_store_clear_written_lookup_is_range_only,
        MemStoreClearWrittenValueLimb,
        "../testdata/lookups/mem_store_clear_written_lookup_is_range_only.mlir"
    );
    direct_lookup_fixture_test!(
        aligned_rom_read_lookup_is_range_only,
        AlignedRomRead,
        "../testdata/lookups/aligned_rom_read_lookup_is_range_only.mlir"
    );

    disjunctive_lookup_fixture_test!(
        disjunctive_lookup_uses_row_multiplier_for_safe_tables,
        MemoryLoadHalfwordOrByte,
        "../testdata/lookups/disjunctive_lookup_uses_row_multiplier_for_safe_tables.mlir"
    );
    disjunctive_lookup_fixture_test!(
        disjunctive_lookup_uses_conditional_constraints_for_unsafe_tables,
        JumpCleanupOffset,
        "../testdata/lookups/disjunctive_lookup_uses_conditional_constraints_for_unsafe_tables.mlir"
    );
}
