//! Encodings of lookup tables into LLZK.

use crate::builder::OpsBuilder;
use crate::codegen::EmitLLZKInStruct as _;
use crate::codegen::StructVars;
use crate::field::FieldInfo;
use anyhow::Result;
use llzk::dialect::constrain;
use llzk::dialect::felt;
use melior::ir::Value;
use prover::cs::cs::circuit::LookupQuery;
use prover::cs::tables::TableType;
use prover::field::PrimeField;

///
pub fn add_lookup_constraints_for_table<'ctx, 'sco, F: PrimeField + FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco>,
    vars: &StructVars,
    query: &LookupQuery<F>,
    table: TableType,
    row_multiplier: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    match table {
        TableType::JumpCleanupOffset => {
            add_jump_cleanup_lookup_constraints(builder, vars, query, row_multiplier)
        }
        TableType::ConditionalJmpBranchSlt => {
            add_conditional_jmp_branch_slt_lookup_constraints(builder, vars, query, row_multiplier)
        }
        _ => todo!(),
    }
}

fn apply_row_multiplier<'ctx, 'sco, F: PrimeField + FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco>,
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
fn add_jump_cleanup_lookup_constraints<'ctx, 'sco, F: PrimeField + FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco>,
    vars: &StructVars,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let a = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_llzk(builder, vars)?,
    )?;
    let bit_1 = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_llzk(builder, vars)?,
    )?;
    let cleaned = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_llzk(builder, vars)?,
    )?;

    let bit_0 = builder.new_nondet_felt::<F>()?;
    let k = builder.new_nondet_felt::<F>()?;

    builder.append_range_constraint::<F>(a, 16)?;
    builder.append_range_constraint::<F>(cleaned, 16)?;
    builder.append_boolean_constraint::<F>(bit_0)?;
    builder.append_boolean_constraint::<F>(bit_1)?;
    builder.append_range_constraint::<F>(k, 14)?;

    // a = cleaned + 2*bit_1 + bit_0
    //      2*bit_1
    let bit_1_mul = builder.append_op_with_result(felt::mul(
        builder.unknown_location(),
        builder.get_constant_from_start::<F>(builder.felt_type::<F>(), 2)?,
        bit_1,
    )?)?;
    //      2*bit_1 + bit_0
    let twit =
        builder.append_op_with_result(felt::add(builder.unknown_location(), bit_1_mul, bit_0)?)?;
    //      cleaned + 2*bit_1 + bit_0
    let a_computed =
        builder.append_op_with_result(felt::add(builder.unknown_location(), cleaned, twit)?)?;
    builder.append_op_with_no_results(constrain::eq(builder.unknown_location(), a, a_computed))?;

    // cleaned = 4*k
    builder.append_op_with_no_results(constrain::eq(
        builder.unknown_location(),
        cleaned,
        builder.append_op_with_result(felt::mul(
            builder.unknown_location(),
            builder.get_constant_from_start::<F>(builder.felt_type::<F>(), 4)?,
            k,
        )?)?,
    ))
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
fn add_conditional_jmp_branch_slt_lookup_constraints<'ctx, 'sco, F: PrimeField + FieldInfo>(
    builder: &OpsBuilder<'ctx, 'sco>,
    vars: &StructVars,
    query: &LookupQuery<F>,
    row_multiplier: Option<Value<'ctx, 'sco>>,
) -> Result<()> {
    let a = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[0].emit_llzk(builder, vars)?,
    )?;
    let f3 = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[1].emit_llzk(builder, vars)?,
    )?;
    let flag = apply_row_multiplier::<F>(
        builder,
        row_multiplier,
        query.row[2].emit_llzk(builder, vars)?,
    )?;

    let uf = builder.new_nondet_felt::<F>()?;
    let out_is_zero = builder.new_nondet_felt::<F>()?;
    let sign1 = builder.new_nondet_felt::<F>()?;
    let sign2 = builder.new_nondet_felt::<F>()?;

    builder.append_range_constraint::<F>(a, 4)?; // a < 16
    builder.append_range_constraint::<F>(f3, 3)?; // f3 < 8
    builder.append_boolean_constraint::<F>(uf)?;
    builder.append_boolean_constraint::<F>(out_is_zero)?;
    builder.append_boolean_constraint::<F>(sign1)?;
    builder.append_boolean_constraint::<F>(sign2)?;
    builder.append_boolean_constraint::<F>(flag)?;

    // a = uf + 2*out_is_zero + 4*sign1 + 8*sign2
    builder.append_op_with_no_results(constrain::eq(
        builder.unknown_location(),
        a,
        builder.append_sum::<F>(
            builder.unknown_location(),
            &[
                uf,
                builder.append_const_scaling::<F>(builder.unknown_location(), 2, out_is_zero)?,
                builder.append_const_scaling::<F>(builder.unknown_location(), 4, sign1)?,
                builder.append_const_scaling::<F>(builder.unknown_location(), 8, sign2)?,
            ],
        )?,
    ))?;

    // signs_different = sign1 + sign2 - (2 * sign1 * sign2)
    let signs_different = builder.append_sum::<F>(
        builder.unknown_location(),
        &[
            sign1,
            sign2,
            builder.append_op_with_result(felt::neg(
                builder.unknown_location(),
                builder.append_product::<F>(
                    builder.unknown_location(),
                    &[builder.get_felt_constant_from_start::<F>(2)?, sign1, sign2],
                )?,
            )?)?,
        ],
    )?;
    let unsigned_lt = uf;
    // signed_lt = (sign1 * signs_different) + unsigned_lt * (1 - signs_different);
    let signed_lt = builder.append_op_with_result(felt::add(
        builder.unknown_location(),
        builder.append_product::<F>(builder.unknown_location(), &[sign1, signs_different])?,
        builder.append_product::<F>(
            builder.unknown_location(),
            &[
                unsigned_lt,
                builder.append_op_with_result(felt::sub(
                    builder.unknown_location(),
                    builder.get_felt_constant_from_start::<F>(1)?,
                    signs_different,
                )?)?,
            ],
        )?,
    )?)?;
    let eq = out_is_zero;

    // one-hot for funct3
    let f3_one_hot = builder.append_one_hot::<F>(builder.unknown_location(), 8)?;
    let f3_reconstructed =
        builder.append_one_hot_reconstruction::<F>(builder.unknown_location(), &f3_one_hot)?;
    builder.append_op_with_no_results(constrain::eq(
        builder.unknown_location(),
        f3,
        f3_reconstructed,
    ))?;

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
            builder.get_felt_constant_from_start::<F>(1)?,
            v,
        )?)
    };

    let expected_flag = builder.append_sum::<F>(
        builder.unknown_location(),
        &[
            builder.append_product::<F>(builder.unknown_location(), &[f3_one_hot[0], eq])?,
            builder.append_product::<F>(
                builder.unknown_location(),
                &[f3_one_hot[1], one_minus(eq)?],
            )?,
            builder.append_product::<F>(builder.unknown_location(), &[f3_one_hot[2], signed_lt])?,
            builder
                .append_product::<F>(builder.unknown_location(), &[f3_one_hot[3], unsigned_lt])?,
            builder.append_product::<F>(builder.unknown_location(), &[f3_one_hot[4], signed_lt])?,
            builder.append_product::<F>(
                builder.unknown_location(),
                &[f3_one_hot[5], one_minus(signed_lt)?],
            )?,
            builder
                .append_product::<F>(builder.unknown_location(), &[f3_one_hot[6], unsigned_lt])?,
            builder.append_product::<F>(
                builder.unknown_location(),
                &[f3_one_hot[7], one_minus(unsigned_lt)?],
            )?,
        ],
    )?;
    // flag === expected_flag
    builder.append_op_with_no_results(constrain::eq(
        builder.unknown_location(),
        flag,
        expected_flag,
    ))?;
    Ok(())
}
