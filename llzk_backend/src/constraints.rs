//! LLZK `@constrain` lowering.

use anyhow::anyhow;
use anyhow::Result;
use llzk::dialect::constrain;
use llzk::dialect::felt;
use llzk::prelude::*;
use prover::cs::constraint::Constraint;
use prover::cs::constraint::Term;
use prover::cs::cs::circuit::DisjunctiveLookup;
use prover::cs::cs::circuit::LookupQuery;
use prover::cs::cs::circuit::LookupQueryTableType;
use prover::cs::cs::circuit::RangeCheckQuery;
use prover::cs::definitions::LookupInput;
use prover::cs::types::Boolean;

use crate::builder::*;
use crate::codegen::StructVars;
use crate::field::FieldInfo;
use crate::lookups::add_disjunctive_lookup_constraints;
use crate::lookups::add_lookup_constraints_for_table;

/// Trait implemented by types that can emit LLZK IR within a struct `@constrain` function.
pub(crate) trait EmitLLZKInConstrain<'ctx: 'sco, 'sco, F: FieldInfo> {
    type Output;

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output>;
}

impl<'ctx: 'sco, 'sco, F: FieldInfo, T: EmitLLZKInConstrain<'ctx, 'sco, F, Output = ()>>
    EmitLLZKInConstrain<'ctx, 'sco, F> for Vec<T>
{
    type Output = ();

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        self.iter()
            .try_for_each(|t| t.emit_constrain(builder, vars))
    }
}

/// Extension trait for [`StructDefOpLike`] that adds a method for filling the `@constrain`
/// function.
pub(crate) trait AddConstraints<'ctx: 'op, 'op, F: FieldInfo>:
    StructDefOpLike<'ctx, 'op>
{
    /// Invokes the callback scoped in `@constrain`.
    ///
    /// All ops added with the [`OpsBuilder`] are automatically added to that function.
    fn add_constraints(
        &'op self,
        builder: &'ctx ModuleBuilder<'ctx, F>,
        f: impl FnOnce(&mut OpsBuilder<'ctx, 'op, F>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let constrain_fn = self.get_constrain_func().ok_or_else(|| {
            anyhow!(
                "struct {} is missing its @constrain function",
                StructDefOpLike::name(self)
            )
        })?;
        let mut ops_builder = OpsBuilder::new(builder, constrain_fn);
        f(&mut ops_builder)
    }
}

impl<'ctx: 'op, 'op, F: FieldInfo, T: StructDefOpMutLike<'ctx, 'op>> AddConstraints<'ctx, 'op, F>
    for T
{
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for RangeCheckQuery<F> {
    type Output = ();

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        match &self.input {
            LookupInput::Variable(variable) => {
                let val = vars.get_constrain_val(builder, variable)?;
                builder.append_range_constraint(val, self.width)?;
            }
            LookupInput::Expression { .. } => {
                panic!("range checks over lookup expressions are not yet supported")
            }
        }
        Ok(())
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for (Constraint<F>, bool) {
    type Output = ();

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        let (constraint, _prevent_optimization) = self;

        let zero = builder.get_constant_from_start(builder.felt_type(), 0)?;
        let values = constraint
            .terms
            .iter()
            .map(|term| term.emit_constrain(builder, vars))
            .collect::<Result<Vec<Value<'_, '_>>>>()?;
        let sum = builder.append_sum(builder.unknown_location(), &values)?;
        builder.append_constrain_eq(builder.unknown_location(), sum, zero)
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for Term<F> {
    type Output = Value<'ctx, 'sco>;

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        match self {
            Term::Constant(c) => {
                let coeff = c.as_u64_reduced();
                let coeff_opp = F::CHARACTERISTICS - coeff;
                let coeff_val = builder.get_constant_from_start(builder.felt_type(), coeff)?;
                Ok(if coeff < coeff_opp {
                    coeff_val
                } else {
                    builder
                        .append_op_with_result(felt::neg(builder.unknown_location(), coeff_val)?)?
                })
            }
            Term::Expression {
                coeff,
                inner,
                degree,
            } => {
                let coeff = coeff.as_u64_reduced();

                let coeff_opp = F::CHARACTERISTICS - coeff;
                let mut monomial = builder.get_constant_from_start(builder.felt_type(), 1)?;
                for var in inner.iter().take(*degree) {
                    let var_val = vars.get_constrain_val(builder, var)?;
                    let mul = felt::mul(builder.unknown_location(), monomial, var_val)?;
                    monomial = builder.append_op_with_result(mul)?;
                }

                Ok(if coeff < coeff_opp {
                    if coeff == 1 {
                        monomial
                    } else {
                        let coeff_val =
                            builder.get_constant_from_start(builder.felt_type(), coeff)?;
                        let mul = felt::mul(builder.unknown_location(), coeff_val, monomial)?;
                        builder.append_op_with_result(mul)?
                    }
                } else if coeff_opp == 1 {
                    builder
                        .append_op_with_result(felt::neg(builder.unknown_location(), monomial)?)?
                } else {
                    let coeff_opp_val =
                        builder.get_constant_from_start(builder.felt_type(), coeff_opp)?;
                    let mul = builder.append_op_with_result(felt::mul(
                        builder.unknown_location(),
                        coeff_opp_val,
                        monomial,
                    )?)?;
                    builder.append_op_with_result(felt::neg(builder.unknown_location(), mul)?)?
                })
            }
        }
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for LookupInput<F> {
    type Output = Value<'ctx, 'sco>;

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        match self {
            LookupInput::Variable(var) => vars.get_constrain_val(builder, var),
            LookupInput::Expression {
                linear_terms,
                constant_coeff,
            } => {
                let init = builder.get_constant_from_start(
                    builder.felt_type(),
                    constant_coeff.as_u64_reduced(),
                )?;
                linear_terms
                    .iter()
                    .map(|(coeff, var)| {
                        let coeff_val = builder
                            .get_constant_from_start(builder.felt_type(), coeff.as_u64_reduced())?;
                        builder.append_op_with_result(felt::mul(
                            builder.unknown_location(),
                            coeff_val,
                            vars.get_constrain_val(builder, var)?,
                        )?)
                    })
                    .try_fold(init, |sum, term_val| {
                        builder.append_op_with_result(felt::add(
                            builder.unknown_location(),
                            sum,
                            term_val?,
                        )?)
                    })
            }
        }
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for LookupQuery<F> {
    type Output = ();

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        match self.table {
            // TODO: Currently unsupported, skipped here and in PCL version
            LookupQueryTableType::Variable(_variable) => Ok(()),
            LookupQueryTableType::Constant(table_type) => {
                add_lookup_constraints_for_table(builder, vars, self, table_type, None, None)
            }
        }
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for DisjunctiveLookup<F> {
    type Output = ();

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        add_disjunctive_lookup_constraints(builder, vars, self)
    }
}

impl<'ctx: 'sco, 'sco, F: FieldInfo> EmitLLZKInConstrain<'ctx, 'sco, F> for Boolean {
    type Output = Value<'ctx, 'sco>;

    fn emit_constrain(
        &self,
        builder: &OpsBuilder<'ctx, 'sco, F>,
        vars: &StructVars<F>,
    ) -> Result<Self::Output> {
        match self {
            Boolean::Is(variable) => vars.get_constrain_val(builder, variable),
            Boolean::Not(variable) => builder.append_op_with_result(felt::sub(
                builder.unknown_location(),
                builder.get_felt_constant_from_start(1)?,
                vars.get_constrain_val(builder, variable)?,
            )?),
            Boolean::Constant(c) => builder.get_felt_constant_from_start(*c as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use llzk::prelude::*;
    use prover::cs::definitions::Variable;
    use prover::field::Mersenne31Field;
    use prover::field::PrimeField;

    use super::*;
    use crate::builder::StructBuilder;
    use crate::codegen::AddCompute;
    use crate::codegen::StructVars;

    fn field(value: u64) -> Mersenne31Field {
        Mersenne31Field::from_u64_unchecked(value)
    }

    fn emit_constrain_ir(
        input_vars: &[Variable],
        member_vars: &[(Variable, &str)],
        emit: impl FnOnce(
            &OpsBuilder<'_, '_, Mersenne31Field>,
            &StructVars<Mersenne31Field>,
        ) -> Result<()>,
    ) -> String {
        let ctx = LlzkContext::new();
        let module = llzk_module(Location::unknown(&ctx));
        let builder = ModuleBuilder::<Mersenne31Field>::new(&ctx, &module);

        let mut struct_builder = StructBuilder::new(builder.context(), "constraint_test");
        for _ in input_vars {
            struct_builder.with_input(builder.felt_type());
        }
        for (_, name) in member_vars {
            struct_builder.with_member((*name).to_string(), builder.felt_type(), false);
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

        let struct_op = struct_builder.build_in_module(builder.module()).unwrap();
        struct_op.add_compute(&builder, |_ops| Ok(())).unwrap();
        struct_op
            .add_constraints(&builder, |ops| emit(ops, &vars))
            .unwrap();
        verify_operation_with_diags(&module.as_operation()).unwrap();

        format!("{}", module.as_operation())
    }

    #[test]
    fn boolean_not_on_input_emits_sub_from_one() {
        let input = Variable(7);
        let ir = emit_constrain_ir(&[input], &[], |ops, vars| {
            let _ = Boolean::Not(input).emit_constrain(ops, vars)?;
            Ok(())
        });

        assert!(ir.contains("felt.sub"));
        assert!(ir.contains("%arg1"));
    }

    #[test]
    fn term_with_negative_unit_coefficient_emits_neg() {
        let input = Variable(7);
        let neg_one = field(Mersenne31Field::CHARACTERISTICS - 1);
        let term = Term::from((neg_one, input));
        let ir = emit_constrain_ir(&[input], &[], |ops, vars| {
            let _ = term.emit_constrain(ops, vars)?;
            Ok(())
        });

        assert!(ir.contains("felt.neg"));
    }

    #[test]
    fn lookup_expression_emits_mul_and_add_chain() {
        let lhs = Variable(7);
        let rhs = Variable(8);
        let input = LookupInput::Expression {
            linear_terms: vec![(field(3), lhs), (field(4), rhs)],
            constant_coeff: field(5),
        };
        let ir = emit_constrain_ir(&[lhs, rhs], &[], |ops, vars| {
            let _ = input.emit_constrain(ops, vars)?;
            Ok(())
        });

        assert!(ir.matches("felt.mul").count() >= 2);
        assert!(ir.matches("felt.add").count() >= 2);
        assert!(ir.contains("felt.const  3"));
        assert!(ir.contains("felt.const  4"));
        assert!(ir.contains("felt.const  5"));
    }

    #[test]
    fn constrain_access_reads_member_when_variable_is_not_input() {
        let member = Variable(42);
        let ir = emit_constrain_ir(&[], &[(member, "stored_member")], |ops, vars| {
            let _ = Boolean::Is(member).emit_constrain(ops, vars)?;
            Ok(())
        });

        assert!(ir.contains("struct.readm %arg0[@stored_member]"));
    }

    #[test]
    fn range_check_query_emits_compare_and_constraint() {
        let input = Variable(7);
        let query = RangeCheckQuery::new(input, 8);
        let ir = emit_constrain_ir(&[input], &[], |ops, vars| query.emit_constrain(ops, vars));

        assert!(ir.contains("bool.cmp lt"));
        assert!(ir.contains("constrain.eq"));
    }
}
