//! Conversion from PCL MLIR to PCL

use anyhow::Result;
use prover::field::PrimeField;
use std::cell::RefCell;
use std::collections::BTreeMap;

use llzk::prelude::Module;
use llzk::prelude::OperationLike;
use llzk::prelude::StringAttribute;
use melior::dialect::func;
use melior::ir::Operation;
use melior::ir::OperationRef;
use melior::ir::Value;
use melior::ir::ValueLike;
use picus::PicusCall;
use picus::PicusConstraint;
use picus::PicusExpr;
use picus::PicusModule;
use picus::PicusProgram;

/// Key type for Value
#[derive(Debug, Eq, PartialEq)]
pub struct ValueKey<'ctx, 'sco>(Value<'ctx, 'sco>);

impl<'ctx, 'sco> Ord for ValueKey<'ctx, 'sco> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.to_raw().ptr as usize).cmp(&(other.0.to_raw().ptr as usize))
    }
}

impl<'ctx, 'sco> PartialOrd for ValueKey<'ctx, 'sco> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Maintains mapping of MLIR values -> PCL names.
struct NameState<'ctx, 'sco> {
    names: RefCell<BTreeMap<ValueKey<'ctx, 'sco>, String>>,
}

impl<'ctx, 'sco> NameState<'ctx, 'sco> {
    fn new() -> Self {
        Self {
            names: BTreeMap::new().into(),
        }
    }

    fn get(&self, v: Value<'ctx, 'sco>, prefix: &str) -> String {
        let key = ValueKey(v);
        let mut names = self.names.borrow_mut();
        match names.get(&key) {
            Some(n) => n.clone(),
            None => {
                let name = format!("{}{}", prefix, names.len());
                assert!(
                    names.insert(key, name.clone()).is_none(),
                    "key should not exist"
                );
                name
            }
        }
    }
}

/// Convert the input PCL MLIR Module into PCL lisp.
pub fn to_pcl<F: PrimeField>(module: &Module) -> PicusProgram {
    let mut modules = BTreeMap::<String, PicusModule>::new();
    module
        .as_operation()
        .walk(llzk::prelude::WalkOrder::PreOrder, |op| {
            if op
                .name()
                .as_string_ref()
                .as_str()
                .expect("op name expected")
                == "func.func"
            {
                let picus_module = to_module(op).expect("unable to convert to PicusModule");
                let prior = modules.insert(picus_module.name.clone(), picus_module);
                assert!(prior.is_none(), "duplicate module name");
            }
            llzk::prelude::WalkResult::Advance
        });
    let mut prog = PicusProgram::new(F::CHARACTERISTICS);
    prog.add_modules(&mut modules);
    prog
}

fn fn_name<'ctx, 'sco>(func_op: &OperationRef<'ctx, 'sco>) -> Result<String> {
    let sym_name_attr = func_op.attribute("sym_name")?;
    let str_attr = StringAttribute::try_from(sym_name_attr)?;
    Ok(str_attr.value().to_string())
}

fn inputs<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusExpr> {
    todo!()
}

fn outputs<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusExpr> {
    todo!()
}

fn constraints<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusConstraint> {
    todo!()
}

fn postconditions<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusConstraint> {
    todo!()
}

fn assume_deterministic<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusExpr> {
    todo!()
}

fn calls<'ctx, 'sco>(
    func_op: &OperationRef<'ctx, 'sco>,
    ns: &NameState<'ctx, 'sco>,
) -> Vec<PicusCall> {
    todo!()
}

/// Convert a function into a PicusModule
fn to_module<'ctx, 'sco>(func_op: OperationRef<'ctx, 'sco>) -> Result<PicusModule> {
    let ns = NameState::new();
    Ok(PicusModule {
        name: fn_name(&func_op)?,
        inputs: inputs(&func_op, &ns),
        outputs: outputs(&func_op, &ns),
        constraints: constraints(&func_op, &ns),
        postconditions: postconditions(&func_op, &ns),
        assume_deterministic: assume_deterministic(&func_op, &ns),
        calls: calls(&func_op, &ns),
    })
}
