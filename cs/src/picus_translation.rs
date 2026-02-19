use crate::constraint::{Constraint, Term};
use crate::cs::circuit::{Circuit, CircuitOutput, ShuffleRamMemQuery};
use crate::cs::cs_reference::BasicAssembly;
use crate::definitions::{
    OpcodeFamilyCircuitState, Variable, ADD_SUB_LUI_AUIPC_MOP_FAMILY_NUM_FLAGS,
};
use crate::machine::ops::unrolled::{
    add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode,
    add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode_and_decoded_bits,
    add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn,
};
use crate::machine::ops::{RS1_LOAD_LOCAL_TIMESTAMP, RS2_LOAD_LOCAL_TIMESTAMP};
use field::{Mersenne31Field, PrimeField};
use picus::{PicusConstraint, PicusExpr, PicusModule, PicusProgram};
use std::collections::BTreeMap;

const U16_BOUND: u64 = 1 << 16;

fn variable_to_picus_expr(var: Variable) -> PicusExpr {
    PicusExpr::Var(var.0 as usize)
}

fn neg_expr(expr: PicusExpr) -> PicusExpr {
    PicusExpr::Sub(Box::new(PicusExpr::Const(0)), Box::new(expr))
}

fn term_to_picus_expr<F: PrimeField>(term: &Term<F>) -> PicusExpr {
    match term {
        Term::Constant(c) => {
            let coeff = c.as_u64_reduced();
            let coeff_opp = F::CHARACTERISTICS - coeff;
            if coeff < coeff_opp {
                PicusExpr::Const(coeff)
            } else {
                neg_expr(PicusExpr::Const(coeff_opp))
            }
        }
        Term::Expression {
            coeff,
            inner,
            degree,
        } => {
            let coeff = coeff.as_u64_reduced();
            let coeff_opp = F::CHARACTERISTICS - coeff;
            let mut monomial = PicusExpr::Const(1);
            for var in inner.iter().take(*degree) {
                monomial = monomial * variable_to_picus_expr(*var);
            }

            if coeff < coeff_opp {
                if coeff == 1 {
                    monomial
                } else {
                    PicusExpr::Const(coeff) * monomial
                }
            } else if coeff_opp == 1 {
                neg_expr(monomial)
            } else {
                neg_expr(PicusExpr::Const(coeff_opp) * monomial)
            }
        }
    }
}

fn constraint_to_picus_constraint<F: PrimeField>(constraint: &Constraint<F>) -> PicusConstraint {
    let expr = constraint
        .terms
        .iter()
        .map(term_to_picus_expr::<F>)
        .fold(PicusExpr::Const(0), |acc, term_expr| acc + term_expr);

    PicusConstraint::Eq(Box::new(expr))
}

fn lookup_input_to_picus_expr<F: PrimeField>(
    input: &crate::definitions::LookupInput<F>,
) -> PicusExpr {
    match input {
        crate::definitions::LookupInput::Variable(variable) => variable_to_picus_expr(*variable),
        crate::definitions::LookupInput::Expression {
            linear_terms,
            constant_coeff,
        } => linear_terms.iter().fold(
            PicusExpr::Const(constant_coeff.as_u64_reduced()),
            |acc, (coeff, variable)| {
                acc + (PicusExpr::Const(coeff.as_u64_reduced()) * variable_to_picus_expr(*variable))
            },
        ),
    }
}

pub fn add_circuit_input_and_outputs<F: PrimeField>(
    module: &mut PicusModule,
    ram_queries: &[ShuffleRamMemQuery],
) {
    for query in ram_queries {
        if query.local_timestamp_in_cycle == RS1_LOAD_LOCAL_TIMESTAMP
            || query.local_timestamp_in_cycle == RS2_LOAD_LOCAL_TIMESTAMP
        {
            for val in query.read_value {
                let picus_var = variable_to_picus_expr(val);
                module.inputs.push(picus_var.clone());
                module.constraints.push(PicusConstraint::Lt(
                    Box::new(picus_var),
                    Box::new(PicusExpr::Const(U16_BOUND)),
                ));
            }
        } else {
            for val in query.write_value {
                let picus_var = variable_to_picus_expr(val);
                module.outputs.push(picus_var.clone());
            }
        }
    }
}

pub fn circuit_output_to_picus_program<F: PrimeField>(
    module_name: impl Into<String>,
    circuit_output: &CircuitOutput<F>,
    circuit_state: Option<&OpcodeFamilyCircuitState<F>>,
    decoded_bits: Option<&[usize]>,
) -> PicusProgram {
    let module_name = module_name.into();
    let mut module = PicusModule::new(module_name.clone());
    add_circuit_input_and_outputs::<F>(&mut module, &circuit_output.shuffle_ram_queries);

    let parsed_constraints: Vec<PicusConstraint> = circuit_output
        .constraints
        .iter()
        .map(|(constraint, _prevent_optimization)| constraint_to_picus_constraint(constraint))
        .collect();
    module.constraints.extend_from_slice(&parsed_constraints);

    for boolean_var in &circuit_output.boolean_vars {
        let picus_expr = variable_to_picus_expr(*boolean_var);
        module.constraints.push(PicusConstraint::new_bit(picus_expr));
    }

    for range_check_query in &circuit_output.range_check_expressions {
        let lookup_val = lookup_input_to_picus_expr(&range_check_query.input);
        let bound = 1u64
            .checked_shl(range_check_query.width as u32)
            .expect("range check width must be less than 64");
        module.constraints.push(PicusConstraint::Lt(
            Box::new(lookup_val),
            Box::new(PicusExpr::Const(bound)),
        ));
    }

    if let Some(cs) = circuit_state {
        let rd_is_zero_picus = variable_to_picus_expr(cs.decoder_data.rd_is_zero);
        let [rd_imm_low_var, rd_imm_high_var] =
            cs.decoder_data.imm.map(|v| variable_to_picus_expr(v));
        let [pc_low, pc_high] = cs.cycle_start_state.pc.map(|v| variable_to_picus_expr(v));
        let [next_pc_low, next_pc_high] = cs.cycle_end_state.pc.map(|v| variable_to_picus_expr(v));
        module.inputs.push(rd_is_zero_picus.clone());
        module.inputs.push(rd_imm_low_var.clone());
        module.inputs.push(rd_imm_high_var.clone());
        module.outputs.push(next_pc_low.clone());
        module.outputs.push(next_pc_high.clone());
        module
            .inputs
            .extend_from_slice(&[pc_low.clone(), pc_high.clone()]);
        module
            .constraints
            .push(PicusConstraint::new_bit(rd_is_zero_picus.clone()));
        module
            .constraints
            .push(PicusConstraint::new_lt(rd_imm_low_var, U16_BOUND.into()));
        module
            .constraints
            .push(PicusConstraint::new_lt(rd_imm_high_var, U16_BOUND.into()));
        module
            .constraints
            .push(PicusConstraint::new_lt(pc_low, U16_BOUND.into()));
        module
            .constraints
            .push(PicusConstraint::new_lt(pc_high, U16_BOUND.into()));
    }

    let mut modules = BTreeMap::new();
    if let Some(decoded_bits) = decoded_bits {
        // Emit one specialized module per opcode bit: bit_i = 1, all other bits = 0.
        for active_bit in 0..decoded_bits.len() {
            let mut env = BTreeMap::new();
            for (idx, bit_var_id) in decoded_bits.iter().copied().enumerate() {
                let value = if idx == active_bit { 1 } else { 0 };
                env.insert(bit_var_id, value);
            }
            let specialized = module.partial_eval(&env);
            modules.insert(specialized.name.clone(), specialized);
        }
    } else {
        modules.insert(module_name, module);
    }

    let mut program = PicusProgram::new(F::CHARACTERISTICS);
    program.add_modules(&mut modules);
    program
}

pub fn build_add_sub_lui_auipc_mop_circuit_output() -> CircuitOutput<Mersenne31Field> {
    let mut cs = BasicAssembly::<Mersenne31Field>::new();
    add_sub_lui_auipc_mop_table_addition_fn(&mut cs);
    add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(&mut cs);
    let (circuit_output, _) = cs.finalize();
    circuit_output
}

pub fn build_add_sub_lui_auipc_mop_circuit_output_with_decoded_bits() -> (
    CircuitOutput<Mersenne31Field>,
    OpcodeFamilyCircuitState<Mersenne31Field>,
    [usize; ADD_SUB_LUI_AUIPC_MOP_FAMILY_NUM_FLAGS],
) {
    let mut cs = BasicAssembly::<Mersenne31Field>::new();
    add_sub_lui_auipc_mop_table_addition_fn(&mut cs);
    let (input, decoded_mask_bits) =
        add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode_and_decoded_bits(&mut cs);
    let (circuit_output, _) = cs.finalize();
    (
        circuit_output,
        input,
        decoded_mask_bits.map(|v| v.0 as usize),
    )
}

pub fn build_add_sub_lui_auipc_mop_picus_program() -> PicusProgram {
    let (circuit_output, input, decoded_bits) =
        build_add_sub_lui_auipc_mop_circuit_output_with_decoded_bits();
    circuit_output_to_picus_program(
        "add_sub_lui_auipc_mop",
        &circuit_output,
        Some(&input),
        Some(decoded_bits.as_slice()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_lui_auipc_mop_translation_smoke_test() {
        let circuit_output = build_add_sub_lui_auipc_mop_circuit_output();
        let program =
            circuit_output_to_picus_program("add_sub_lui_auipc_mop", &circuit_output, None, None);
        let dumped = program.to_string();
        assert!(dumped.contains("(begin-module add_sub_lui_auipc_mop)"));
        assert!(dumped.contains("(prime-number"));
    }

    #[test]
    fn add_sub_lui_auipc_mop_one_hot_specialization_emits_one_module_per_bit() {
        let (circuit_output, input, decoded_bits) =
            build_add_sub_lui_auipc_mop_circuit_output_with_decoded_bits();
        let program = circuit_output_to_picus_program(
            "add_sub_lui_auipc_mop",
            &circuit_output,
            Some(&input),
            Some(decoded_bits.as_slice()),
        );
        let dumped = program.to_string();
        let module_count = dumped.matches("(begin-module ").count();
        assert_eq!(module_count, ADD_SUB_LUI_AUIPC_MOP_FAMILY_NUM_FLAGS);
    }
}
