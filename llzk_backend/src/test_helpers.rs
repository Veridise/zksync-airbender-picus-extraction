//! Shared helpers for LLZK backend unit tests.

use std::collections::HashMap;

use anyhow::Result;
use llzk::prelude::*;
use prover::cs::definitions::Variable;
use prover::field::Mersenne31Field;

use crate::builder::ModuleEnv;
use crate::builder::OpsBuilder;
use crate::builder::StructBuilder;
use crate::codegen::AddCompute;
use crate::codegen::StructVars;
use crate::constraints::AddConstraints;

/// Normalize textual IR by trimming trailing whitespace at the end of each line.
///
/// The printer currently emits a few trailing spaces, so tests compare normalized strings rather
/// than relying on editor-specific whitespace handling.
fn normalize_ir(ir: &str) -> String {
    ir.trim()
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Assert that two IR strings are exactly equal after normalization.
pub(crate) fn assert_full_ir_eq(actual: &str, expected: &str) {
    assert_eq!(normalize_ir(actual), normalize_ir(expected));
}

/// Emit a synthetic `@constrain` body for unit tests.
///
/// The helper exposes each `input_var` as a plain felt input, each `member_var` as a plain felt
/// struct member, and then runs `emit` inside the generated `@constrain` body.
pub(crate) fn emit_test_constrain_ir(
    struct_name: &str,
    input_vars: &[Variable],
    member_vars: &[(Variable, &str)],
    emit: impl FnOnce(&OpsBuilder<'_, '_, Mersenne31Field>, &StructVars<Mersenne31Field>) -> Result<()>,
) -> String {
    let ctx = LlzkContext::new();
    let module = llzk_module(Location::unknown(&ctx));
    let env = ModuleEnv::<Mersenne31Field>::new(&ctx, &module);

    let mut struct_builder = StructBuilder::new(&env, struct_name);
    for _ in input_vars {
        struct_builder.with_input(env.felt_type());
    }
    for (_, name) in member_vars {
        struct_builder.with_member((*name).to_string(), env.felt_type(), false);
    }

    let arg_map = input_vars
        .iter()
        .enumerate()
        .map(|(idx, var)| (*var, (idx, None)))
        .collect::<HashMap<_, _>>();
    let member_map = member_vars
        .iter()
        .map(|(var, name)| (*var, ((*name).to_string(), None)))
        .collect::<HashMap<_, _>>();
    let vars = StructVars::from_test_maps(member_map, arg_map);

    let struct_op = struct_builder.build_in_module().unwrap();
    struct_op.add_compute(&env, |_ops| Ok(())).unwrap();
    struct_op
        .add_constraints(&env, |ops| emit(ops, &vars))
        .unwrap();
    verify_operation_with_diags(&module.as_operation()).unwrap();

    format!("{}", module.as_operation())
}
