//! Builder types for encapsulating common codegen tasks.
//!
//! Contains:
//! - A generic builder with stateless factory methods.
//! - An operations builder meant for creating ops inside a function.
//! - A struct builder.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::Deref;

use anyhow::anyhow;
use anyhow::Result;
use llzk::builder::OpBuilder;
use llzk::dialect::bool;
use llzk::dialect::constrain;
use llzk::dialect::felt;
use llzk::prelude::dialect::array;
use llzk::prelude::dialect::r#struct;
use llzk::prelude::melior_dialects::arith;
use llzk::prelude::*;
use llzk::utils::IsA;
use prover::cs::definitions::REGISTER_SIZE;

use crate::field::FieldInfo;

/// Root builder with convenience factory methods and access to the root LLZK module.
pub struct ModuleBuilder<'ctx, F: FieldInfo> {
    context: &'ctx Context,
    /// The root LLZK module.
    module: &'ctx Module<'ctx>,
    _field: core::marker::PhantomData<F>,
}

impl<'ctx, F: FieldInfo> ModuleBuilder<'ctx, F> {
    /// Creates a new builder.
    pub fn new(context: &'ctx Context, module: &'ctx Module<'ctx>) -> Self {
        Self {
            context,
            module,
            _field: PhantomData,
        }
    }

    /// Returns a reference to the context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Returns a reference to the root module.
    pub fn module(&self) -> &Module<'ctx> {
        self.module
    }

    /// Returns the unknown location.
    pub fn unknown_location(&self) -> Location<'ctx> {
        Location::unknown(self.context)
    }

    /// Creates a `!felt.type`.
    pub fn felt_type(&self) -> Type<'ctx> {
        FeltType::with_field(self.context, F::field_name()).into()
    }

    /// Get the index type
    #[inline]
    pub fn index_type(&self) -> Type<'ctx> {
        Type::index(self.context)
    }

    /// Get an integer type
    pub fn int_type(&self, bits: u32) -> Type<'ctx> {
        IntegerType::new(self.context, bits).into()
    }

    /// Get the boolean type (i.e., i1)
    pub fn bool_type(&self) -> Type<'ctx> {
        IntegerType::new(self.context, 1).into()
    }

    /// Get a constant index-type integer attribute
    #[inline]
    pub fn index_attr(&self, integer: i64) -> Attribute<'ctx> {
        self.int_attr(self.index_type(), integer)
    }

    /// Create a constant felt attribute.
    pub fn felt_attr(&self, value: u64) -> FeltConstAttribute<'ctx> {
        FeltConstAttribute::new(self.context, value, Some(F::field_name()))
    }

    /// Create a constant int attribute of the given int type.
    #[inline]
    pub fn int_attr(&self, r#type: Type<'ctx>, integer: i64) -> Attribute<'ctx> {
        IntegerAttribute::new(r#type, integer).into()
    }

    /// Get a register type, which is a two-element felt array.
    /// TODO: This is probably too representation dependent, move elsewhere?
    pub fn register_type(&self) -> Type<'ctx> {
        ArrayType::new(
            self.felt_type(),
            &[self.index_attr(
                i64::try_from(REGISTER_SIZE).expect("REGISTER_SIZE is unexpectedly large"),
            )],
        )
        .into()
    }
}

/// Possible locations for insertion.
enum InsertionPoint {
    /// Beginning of function.
    Start,
    /// End of function (before the terminator, if any)
    End,
    /// At a concrete position.
    #[allow(dead_code)]
    At(usize),
}

/// Key type for caching const op values
#[derive(Debug, Eq, PartialEq)]
pub struct ConstOpKey<'ctx>(Type<'ctx>, u64);

impl<'ctx> Ord for ConstOpKey<'ctx> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1
            .cmp(&other.1)
            .then_with(|| self.0.to_string().cmp(&other.0.to_string()))
    }
}

impl<'ctx> PartialOrd for ConstOpKey<'ctx> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Operations builder that handles insertion of operations in the target function.
pub struct OpsBuilder<'ctx: 'sco, 'sco, F: FieldInfo> {
    builder: &'ctx ModuleBuilder<'ctx, F>,
    scope: FuncDefOpRef<'ctx, 'sco>,
    /// Cache of constant op values of specified type at the beginning of the
    /// function scope. Using a BTreeMap since [Type] is not hashable.
    const_vals: RefCell<BTreeMap<ConstOpKey<'ctx>, Value<'ctx, 'sco>>>,
}

impl<'ctx, 'sco, F: FieldInfo> OpsBuilder<'ctx, 'sco, F> {
    /// Creates a new builder.
    pub fn new(builder: &'ctx ModuleBuilder<'ctx, F>, scope: FuncDefOpRef<'ctx, 'sco>) -> Self {
        Self {
            scope,
            builder,
            const_vals: BTreeMap::new().into(),
        }
    }

    /// Appends an operation with no results at the end.
    #[inline]
    pub fn append_op_with_no_results(&self, op: Operation<'ctx>) -> Result<()> {
        let _ = self.insert_operation(InsertionPoint::End, op)?;
        Ok(())
    }

    /// Appends an operation with results at the end.
    #[inline]
    pub fn append_op_with_results<const N: usize>(
        &self,
        op: Operation<'ctx>,
    ) -> Result<[Value<'ctx, 'sco>; N]> {
        let op = self.insert_operation(InsertionPoint::End, op)?;
        self.extract_results(op)
    }

    /// Appends an operation with one result at the end.
    #[inline]
    pub fn append_op_with_result(&self, op: Operation<'ctx>) -> Result<Value<'ctx, 'sco>> {
        self.append_op_with_results::<1>(op).map(|v| v[0])
    }

    /// Inserts an operation with no results at the start.
    #[inline]
    #[allow(dead_code)]
    pub fn insert_op_with_no_results_at_start(&self, op: Operation<'ctx>) -> Result<()> {
        let _ = self.insert_operation(InsertionPoint::Start, op)?;
        Ok(())
    }

    /// Inserts an operation with results at the start.
    #[inline]
    #[allow(dead_code)]
    pub fn insert_op_with_results_at_start<const N: usize>(
        &self,
        op: Operation<'ctx>,
    ) -> Result<[Value<'ctx, 'sco>; N]> {
        let op = self.insert_operation(InsertionPoint::Start, op)?;
        self.extract_results(op)
    }

    /// Inserts an operation with one result at the start.
    #[inline]
    pub fn insert_op_with_result_at_start(&self, op: Operation<'ctx>) -> Result<Value<'ctx, 'sco>> {
        self.insert_op_with_results_at_start::<1>(op).map(|v| v[0])
    }

    /// Inserts an operation with no results at the given position.
    #[inline]
    #[allow(dead_code)]
    pub fn insert_op_with_no_results_at(&self, pos: usize, op: Operation<'ctx>) -> Result<()> {
        let _ = self.insert_operation(InsertionPoint::At(pos), op)?;
        Ok(())
    }

    /// Inserts an operation with results at the given position.
    #[inline]
    #[allow(dead_code)]
    pub fn insert_op_with_results_at<const N: usize>(
        &self,
        pos: usize,
        op: Operation<'ctx>,
    ) -> Result<[Value<'ctx, 'sco>; N]> {
        let op = self.insert_operation(InsertionPoint::At(pos), op)?;
        self.extract_results(op)
    }

    /// Inserts an operation with one result at the given position.
    #[inline]
    #[allow(dead_code)]
    pub fn insert_op_with_result_at(
        &self,
        pos: usize,
        op: Operation<'ctx>,
    ) -> Result<Value<'ctx, 'sco>> {
        self.insert_op_with_results_at::<1>(pos, op).map(|v| v[0])
    }

    fn extract_results<const N: usize>(
        &self,
        op: OperationRef<'ctx, 'sco>,
    ) -> Result<[Value<'ctx, 'sco>; N]> {
        op.results()
            .map(Into::into)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|values: Vec<_>| anyhow!("was expecting {N} results but got {}", values.len()))
    }

    fn first_block(&self) -> Result<BlockRef<'ctx, 'sco>> {
        assert_eq!(self.scope.region_count(), 1);
        self.scope
            .region(0)?
            .first_block()
            .ok_or_else(|| anyhow!("function's region is missing a block"))
    }

    fn last_block(&self) -> Result<BlockRef<'ctx, 'sco>> {
        self.blocks()
            .last()
            .ok_or_else(|| anyhow!("function's region is missing a block"))
    }

    fn blocks(&self) -> impl Iterator<Item = BlockRef<'ctx, 'sco>> {
        std::iter::successors(self.first_block().ok(), |blk: &BlockRef| {
            blk.next_in_region()
        })
    }

    fn operations(&self) -> impl Iterator<Item = OperationRef<'ctx, 'sco>> {
        self.blocks()
            .flat_map(|blk| std::iter::successors(blk.first_operation(), |op| op.next_in_block()))
    }

    fn extract_insertion_point(
        &self,
        point: InsertionPoint,
    ) -> Result<(Option<OperationRef<'ctx, 'sco>>, BlockRef<'ctx, 'sco>)> {
        Ok(match point {
            InsertionPoint::Start | InsertionPoint::At(0) => {
                let blk = self.first_block()?;
                (blk.first_operation(), blk)
            }
            InsertionPoint::End => {
                let blk = self.last_block()?;
                (blk.terminator(), blk)
            }
            InsertionPoint::At(pos) => {
                let op = self
                    .operations()
                    .nth(pos - 1)
                    .ok_or_else(|| anyhow!("operation position {pos} is out of bounds"))?;

                (
                    Some(op),
                    op.block()
                        .expect("operation ref comes from an iterator of blocks"),
                )
            }
        })
    }

    /// Generic insertion function. The other insertion functions should be convenience methods over
    /// this one.
    fn insert_operation(
        &self,
        point: InsertionPoint,
        operation: Operation<'ctx>,
    ) -> Result<OperationRef<'ctx, 'sco>> {
        let (point, blk) = self.extract_insertion_point(point)?;
        Ok(match point {
            Some(fst) => blk.insert_operation_before(fst, operation),
            None => blk.append_operation(operation),
        })
    }

    /// Insert a `constrain.eq` operation to constrain `lhs` equal to `rhs`. If
    /// `conditional` is supplied, then the constraint will be `conditional => (lhs === rhs)`
    /// (implemented as `!conditional || (lhs === rhs)` since LLZK has no implication operation).
    #[inline]
    pub fn append_constrain_eq(
        &self,
        location: Location<'ctx>,
        lhs: Value<'ctx, 'sco>,
        rhs: Value<'ctx, 'sco>,
    ) -> Result<()> {
        self.append_op_with_no_results(constrain::eq(location, lhs, rhs))
    }

    /// If not None, insert a `constrain.eq` operation to constrain `conditional => (lhs === rhs)`
    /// (implemented as `!conditional || (lhs === rhs)` since LLZK has no implication operation).
    /// Assumes `conditional` is a felt.type that is constrained to be in a boolean range.
    /// If `conditional` is None, just inserts a regular equality constraint between `lhs` and
    /// `rhs`.
    pub fn append_conditional_constrain_eq(
        &self,
        location: Location<'ctx>,
        conditional: Option<Value<'ctx, 'sco>>,
        lhs: Value<'ctx, 'sco>,
        rhs: Value<'ctx, 'sco>,
    ) -> Result<()> {
        match conditional {
            None => self.append_constrain_eq(location, lhs, rhs),
            Some(conditional) => {
                let not_conditional = self.append_eq_predicate(
                    location,
                    self.get_felt_constant_from_start(0)?,
                    conditional,
                )?;
                let sides_eq = self.append_eq_predicate(location, lhs, rhs)?;
                let implication =
                    self.append_op_with_result(bool::or(location, not_conditional, sides_eq)?)?;
                let truth = self.get_constant_from_start(self.bool_type(), 1)?;
                self.append_constrain_eq(location, implication, truth)
            }
        }
    }

    /// Compare `lhs` and `rhs` and return an `i1` predicate.
    ///
    /// LLZK uses different equality ops for felts and plain integer types, so conditional
    /// constraints need this helper instead of assuming every compared value is a felt.
    fn append_eq_predicate(
        &self,
        location: Location<'ctx>,
        lhs: Value<'ctx, 'sco>,
        rhs: Value<'ctx, 'sco>,
    ) -> Result<Value<'ctx, 'sco>> {
        anyhow::ensure!(
            lhs.r#type() == rhs.r#type(),
            "cannot compare values with different types: {} vs {}",
            lhs.r#type(),
            rhs.r#type()
        );

        if lhs.r#type() == self.felt_type() {
            self.append_op_with_result(bool::eq(location, lhs, rhs)?)
        } else if lhs.r#type() == self.index_type() || lhs.r#type().isa::<IntegerType>() {
            self.append_op_with_result(arith::cmpi(
                self.context,
                arith::CmpiPredicate::Eq,
                lhs,
                rhs,
                location,
            ))
        } else {
            anyhow::bail!("unsupported equality predicate type {}", lhs.r#type());
        }
    }

    /// Compute the inner values used to generate a boolean constraint.
    /// Used so both the conditional and unconditional constraint variants use
    /// the same logic.
    fn compute_boolean_constraint_expression(
        &self,
        val: Value<'ctx, 'sco>,
    ) -> Result<(Value<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        assert_eq!(val.r#type(), self.felt_type());
        let unk = self.unknown_location();
        let zero = self.get_constant_from_start(self.felt_type(), 0)?;
        let one = self.get_constant_from_start(self.felt_type(), 1)?;
        let minus_one = self.append_op_with_result(felt::sub(unk, val, one)?)?;
        let product = self.append_op_with_result(felt::mul(unk, val, minus_one)?)?;
        Ok((product, zero))
    }

    /// Append a boolean constraint for the given value.
    #[inline]
    pub fn append_boolean_constraint(&self, val: Value<'ctx, 'sco>) -> Result<()> {
        let (product, zero) = self.compute_boolean_constraint_expression(val)?;
        self.append_constrain_eq(self.unknown_location(), product, zero)
    }

    /// Append a conditional (if provided) boolean constraint for the given value.
    #[inline]
    pub fn append_conditional_boolean_constraint(
        &self,
        conditional: Option<Value<'ctx, 'sco>>,
        val: Value<'ctx, 'sco>,
    ) -> Result<()> {
        let (product, zero) = self.compute_boolean_constraint_expression(val)?;
        self.append_conditional_constrain_eq(self.unknown_location(), conditional, product, zero)
    }

    /// Compute the inner values ised to generate a range constraint.
    /// Used so both the conditional and unconditioanl constraint variants use the same logic.
    fn compute_range_constraint_expression(
        &self,
        val: Value<'ctx, 'sco>,
        width: usize,
    ) -> Result<(Value<'ctx, 'sco>, Value<'ctx, 'sco>)> {
        assert_eq!(val.r#type(), self.felt_type());
        let bound = self.get_constant_from_start(self.felt_type(), 1 << width)?;
        let bound_check =
            self.append_op_with_result(bool::lt(self.unknown_location(), val, bound)?)?;
        let truth = self.get_constant_from_start(self.int_type(1), 1)?;
        Ok((bound_check, truth))
    }

    /// Append a range constraint for the given value.
    /// Enforces that `val` must be within `width`.
    pub fn append_range_constraint(&self, val: Value<'ctx, 'sco>, width: usize) -> Result<()> {
        let (bound_check, truth) = self.compute_range_constraint_expression(val, width)?;
        self.append_constrain_eq(self.unknown_location(), bound_check, truth)
    }

    /// Append a range constraint for the given value.
    /// Enforces that `val` must be within `width` if `conditional` is provided and is true.
    pub fn append_conditional_range_constraint(
        &self,
        conditional: Option<Value<'ctx, 'sco>>,
        val: Value<'ctx, 'sco>,
        width: usize,
    ) -> Result<()> {
        let (bound_check, truth) = self.compute_range_constraint_expression(val, width)?;
        self.append_conditional_constrain_eq(
            self.unknown_location(),
            conditional,
            bound_check,
            truth,
        )
    }

    /// Get the value from the contained function scope.
    pub fn get_arg_value(&self, arg_no: usize) -> Result<Value<'ctx, 'sco>> {
        Ok(self.scope.argument(arg_no)?.into())
    }

    /// Return the struct instance created at the start of a `@compute` body.
    pub fn get_compute_self_value(&self) -> Result<Value<'ctx, 'sco>> {
        // LLZK exposes this directly on the function op, conveniently
        Ok(self.scope.self_value_of_compute()?)
    }

    /// Append a struct member read operation in the current function scope.
    pub fn append_member_read(
        &self,
        location: Location<'ctx>,
        component: Value<'ctx, 'sco>,
        result_type: Type<'ctx>,
        member_name: &str,
    ) -> Result<Value<'ctx, 'sco>> {
        let op = r#struct::readm(
            &OpBuilder::new(self.context),
            location,
            result_type,
            component,
            member_name,
        )?;
        self.append_op_with_result(op)
    }

    /// Append a struct member write operation in the current function scope.
    pub fn append_member_write(
        &self,
        location: Location<'ctx>,
        component: Value<'ctx, 'sco>,
        member_name: &str,
        value: Value<'ctx, 'sco>,
    ) -> Result<()> {
        let op = r#struct::writem(location, component, member_name, value)?;
        self.append_op_with_no_results(op)
    }

    /// Append an array read operation and return the read value.
    pub fn append_array_read(
        &self,
        location: Location<'ctx>,
        arr_ref: Value<'ctx, 'sco>,
        indices: &[Value<'ctx, 'sco>],
    ) -> Result<Value<'ctx, 'sco>> {
        let arr_ty = ArrayType::try_from(arr_ref.r#type())?;
        self.append_op_with_result(array::read(
            location,
            arr_ty.element_type(),
            arr_ref,
            indices,
        ))
    }

    /// Append an array write operation in the current function scope.
    pub fn append_array_write(
        &self,
        location: Location<'ctx>,
        arr_ref: Value<'ctx, 'sco>,
        indices: &[Value<'ctx, 'sco>],
        rvalue: Value<'ctx, 'sco>,
    ) -> Result<()> {
        self.append_op_with_no_results(array::write(location, arr_ref, indices, rvalue))
    }

    /// Lookup a previously generated constant in the function scope or
    /// create one if needed. Then return the SSA value.
    pub fn get_constant_from_start(&self, r#type: Type<'ctx>, i: u64) -> Result<Value<'ctx, 'sco>> {
        let key = ConstOpKey(r#type, i);
        let mut const_val_cache = self.const_vals.borrow_mut();
        match const_val_cache.get(&key) {
            Some(v) => Ok(*v),
            None => {
                let const_op = if r#type == self.index_type() || r#type.isa::<IntegerType>() {
                    arith::constant(
                        self.context,
                        self.int_attr(r#type, i64::try_from(i)?),
                        self.unknown_location(),
                    )
                } else if r#type == self.felt_type() {
                    felt::constant(self.unknown_location(), self.felt_attr(i))?
                } else {
                    anyhow::bail!("unsupported type {}", r#type)
                };
                let v = self.insert_op_with_result_at_start(const_op)?;
                anyhow::ensure!(
                    const_val_cache.insert(key, v).is_none(),
                    "replaced existing index const value in function preamble"
                );
                Ok(v)
            }
        }
    }

    /// Get a `felt.type` constant from the function prologue.
    #[inline]
    pub fn get_felt_constant_from_start(&self, i: u64) -> Result<Value<'ctx, 'sco>> {
        self.get_constant_from_start(self.felt_type(), i)
    }

    /// Perform the index constant insertion without producing a return value.
    pub fn insert_constant_at_start(&self, r#type: Type<'ctx>, i: u64) -> Result<()> {
        let _ = self.get_constant_from_start(r#type, i)?;
        Ok(())
    }

    /// Create a new nondet value of the specified type.
    #[inline]
    pub fn new_nondet(&self, r#type: Type<'ctx>) -> Result<Value<'ctx, 'sco>> {
        self.append_op_with_result(llzk::dialect::llzk::nondet(self.unknown_location(), r#type))
    }

    /// Create a new nondet felt.
    #[inline]
    pub fn new_nondet_felt(&self) -> Result<Value<'ctx, 'sco>> {
        self.new_nondet(self.felt_type())
    }

    /// Fold the given values using the binary operation provided.
    fn append_fold<FN>(
        &self,
        location: Location<'ctx>,
        operation_fn: FN,
        values: &[Value<'ctx, 'sco>],
    ) -> Result<Value<'ctx, 'sco>>
    where
        FN: Fn(
            Location<'ctx>,
            Value<'ctx, 'sco>,
            Value<'ctx, 'sco>,
        ) -> Result<Operation<'ctx>, llzk::error::Error>,
    {
        values
            .iter()
            .map(|v| Ok(*v))
            .reduce(|acc, v| {
                let add = operation_fn(location, acc?, v?)?;
                self.append_op_with_result(add)
            })
            .ok_or_else(|| anyhow!("must provide values to append_fold"))?
    }

    /// Perform addition using `felt.add` over all specified values.
    #[inline]
    pub fn append_sum(
        &self,
        location: Location<'ctx>,
        values: &[Value<'ctx, 'sco>],
    ) -> Result<Value<'ctx, 'sco>> {
        self.append_fold::<_>(location, felt::add, values)
    }

    /// Perform multiplication using `felt.mul` over all specified values.
    #[inline]
    pub fn append_product(
        &self,
        location: Location<'ctx>,
        values: &[Value<'ctx, 'sco>],
    ) -> Result<Value<'ctx, 'sco>> {
        self.append_fold::<_>(location, felt::mul, values)
    }

    /// Append a multiplication by the given constant felt value using `felt.mul`.
    pub fn append_const_scaling(
        &self,
        location: Location<'ctx>,
        const_coeff: u64,
        val: Value<'ctx, 'sco>,
    ) -> Result<Value<'ctx, 'sco>> {
        self.append_op_with_result(felt::mul(
            location,
            self.get_felt_constant_from_start(const_coeff)?,
            val,
        )?)
    }

    /// Create a vector of N `felt.type` nondets constrained such that:
    /// - They are all boolean
    /// - Only one of them is 1 (i.e., one bit hot)
    pub fn append_one_hot(
        &self,
        location: Location<'ctx>,
        bits: usize,
    ) -> Result<Vec<Value<'ctx, 'sco>>> {
        let bits = (0..bits)
            .map(|_| {
                let bit = self.new_nondet_felt()?;
                self.append_boolean_constraint(bit)?;
                Ok(bit)
            })
            .collect::<Result<Vec<Value<'ctx, 'sco>>>>()?;
        let sum = self.append_sum(location, &bits)?;
        self.append_op_with_no_results(constrain::eq(
            location,
            sum,
            self.get_felt_constant_from_start(1)?,
        ))?;
        Ok(bits)
    }

    /// Convert a one-hot bit vector into the original single value.
    pub fn append_one_hot_reconstruction(
        &self,
        location: Location<'ctx>,
        bits: &[Value<'ctx, 'sco>],
    ) -> Result<Value<'ctx, 'sco>> {
        let (_, res) = bits
            .iter()
            .enumerate()
            .map(|(i, v)| Ok((i, *v)))
            .reduce(
                |a: Result<(usize, Value<'ctx, 'sco>)>, x: Result<(usize, Value<'ctx, 'sco>)>| {
                    let (_, acc) = a?;
                    let (i, bit) = x?;
                    let new_val = self.append_sum(
                        location,
                        &[
                            acc,
                            self.append_product(
                                location,
                                &[self.get_felt_constant_from_start(u64::try_from(i)?)?, bit],
                            )?,
                        ],
                    )?;
                    Ok((i, new_val))
                },
            )
            .ok_or_else(|| anyhow!("must provide non-empty bits slice"))??;
        Ok(res)
    }
}

impl<'ctx, 'sco, F: FieldInfo> Deref for OpsBuilder<'ctx, 'sco, F> {
    type Target = ModuleBuilder<'ctx, F>;

    fn deref(&self) -> &Self::Target {
        self.builder
    }
}

/// Macro for casting a `Result<T, E>` into `Result<Operation, E>` where `T: Into<Operation>`
macro_rules! as_op {
    ($op:expr) => {
        $op.map(Operation::from)
    };
}

/// Builder for creating structs.
pub struct StructBuilder<'ctx, 'str> {
    /// Reference to the context.
    context: &'ctx Context,
    /// Location for the struct and its direct child ops.
    location: Option<Location<'ctx>>,
    /// Name of the struct.
    name: &'str str,
    /// Inputs of the struct (excluding self in @constrain).
    inputs: Vec<Type<'ctx>>,
    /// List of members. Contains the name, type and whether is marked public or not.
    members: Vec<(String, Type<'ctx>, bool)>,
}

impl<'ctx, 'str> StructBuilder<'ctx, 'str> {
    /// Creates a new builder.
    pub fn new(context: &'ctx Context, name: &'str str) -> Self {
        Self {
            context,
            location: None,
            name,
            inputs: vec![],
            members: vec![],
        }
    }

    /// Adds an input to the list.
    pub fn with_input(&mut self, input: Type<'ctx>) -> &mut Self {
        self.inputs.push(input);
        self
    }

    /// Sets the location of the struct.
    #[allow(dead_code)]
    pub fn with_location(&mut self, location: Location<'ctx>) -> &mut Self {
        self.location = Some(location);
        self
    }

    /// Adds a member to the struct.
    pub fn with_member(&mut self, name: String, r#type: Type<'ctx>, is_public: bool) -> &mut Self {
        self.members.push((name, r#type, is_public));
        self
    }

    /// Create the struct type for this struct builder.
    fn struct_type(&self) -> StructType<'ctx> {
        StructType::from_str(self.context, self.name)
    }

    fn location(&self) -> Location<'ctx> {
        self.location
            .unwrap_or_else(|| Location::unknown(self.context))
    }

    /// Creates a struct using the build data.
    pub fn build(&self) -> Result<StructDefOp<'ctx>, LlzkError> {
        let inputs = self
            .inputs
            .iter()
            .map(|arg| (*arg, self.location()))
            .collect::<Vec<_>>();

        let members = self.members.iter().map(|(name, typ, is_pub)| {
            as_op!(dialect::r#struct::member(
                self.location(),
                name,
                *typ,
                true,
                *is_pub
            ))
        });

        let compute = as_op!(dialect::r#struct::helpers::compute_fn(
            self.location(),
            self.struct_type(),
            &inputs,
            None,
        ));
        let constrain = as_op!(dialect::r#struct::helpers::constrain_fn(
            self.location(),
            self.struct_type(),
            &inputs,
            None,
        ));

        dialect::r#struct::def(
            self.location(),
            self.name,
            &[],
            std::iter::chain(members, [compute, constrain]),
        )
    }

    /// Builds the struct, inserts it into the module, then returns a reference to it.
    pub fn build_in_module<'m>(
        &self,
        module: &'m Module<'ctx>,
    ) -> Result<StructDefOpRef<'ctx, 'm>, LlzkError> {
        let op = self.build()?;
        let op_ref = module.body().append_operation(op.into());
        op_ref.try_into()
    }
}
