use crate::constraint::{Constraint, Term};
use crate::cs::circuit::{Circuit, CircuitOutput, ShuffleRamMemQuery};
use crate::cs::cs_reference::BasicAssembly;
use crate::definitions::Variable;
use crate::machine::ops::unrolled::{
    add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode,
    add_sub_lui_auipc_mop::add_sub_lui_auipc_mop_table_addition_fn,
};
use crate::machine::ops::{RS1_LOAD_LOCAL_TIMESTAMP, RS2_LOAD_LOCAL_TIMESTAMP};
use field::{Mersenne31Field, PrimeField};
use picus::{PicusConstraint, PicusExpr, PicusModule, PicusProgram};
use std::collections::BTreeMap;

fn variable_to_picus_expr(var: Variable) -> PicusExpr {
    PicusExpr::Var(var.0 as usize)
}

fn term_to_picus_expr<F: PrimeField>(term: &Term<F>) -> PicusExpr {
    match term {
        Term::Constant(c) => PicusExpr::Const(c.as_u64_reduced()),
        Term::Expression {
            coeff,
            inner,
            degree,
        } => {
            let mut expr = PicusExpr::Const(coeff.as_u64_reduced());
            for var in inner.iter().take(*degree) {
                expr = expr * variable_to_picus_expr(*var);
            }
            expr
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
                    Box::new(PicusExpr::Const(65536)),
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
) -> PicusProgram {
    let module_name = module_name.into();
    let mut module = PicusModule::new(module_name.clone());
    add_circuit_input_and_outputs::<F>(&mut module, &circuit_output.shuffle_ram_queries);
    println!(
        "shuffle ram queries: {:?}",
        circuit_output.shuffle_ram_queries
    );
    let parsed_constraints: Vec<PicusConstraint> = circuit_output
        .constraints
        .iter()
        .map(|(constraint, _prevent_optimization)| constraint_to_picus_constraint(constraint))
        .collect();
    module.constraints.extend_from_slice(&parsed_constraints);
    
    for range_check_query in &circuit_output.range_check_expressions {
        let lookup_val = match &range_check_query.input {
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
        };
        let bound = 1u64
            .checked_shl(range_check_query.width as u32)
            .expect("range check width must be less than 64");
        module.constraints.push(PicusConstraint::Lt(
            Box::new(lookup_val),
            Box::new(PicusExpr::Const(bound)),
        ));
    }

    let mut modules = BTreeMap::new();
    modules.insert(module_name, module);

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

pub fn build_add_sub_lui_auipc_mop_picus_program() -> PicusProgram {
    let circuit_output = build_add_sub_lui_auipc_mop_circuit_output();
    circuit_output_to_picus_program("add_sub_lui_auipc_mop", &circuit_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_lui_auipc_mop_translation_smoke_test() {
        let circuit_output = build_add_sub_lui_auipc_mop_circuit_output();
        let program = circuit_output_to_picus_program("add_sub_lui_auipc_mop", &circuit_output);
        let dumped = program.to_string();
        println!("Dumped: {dumped}");
        assert!(dumped.contains("(begin-module add_sub_lui_auipc_mop)"));
        assert!(dumped.contains("(prime-number"));
    }
}
