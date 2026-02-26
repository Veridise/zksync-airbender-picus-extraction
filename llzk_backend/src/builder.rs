//! Builder types for encapsulating common codegen tasks.
//!
//! Contains:
//! - A generic builder with stateless factory methods.
//! - An operations builder meant for creating ops inside a function.
//! - A struct builder.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    iter::Map,
    ops::Deref,
};

use anyhow::{anyhow, Result};
use llzk::{
    builder::OpBuilder,
    dialect::{bool, constrain, felt},
    prelude::{
        dialect::{array, r#struct},
        melior_dialects::arith,
        *,
    },
    utils::IsA,
};
use prover::cs::definitions::REGISTER_SIZE;

/// Generic builder with convenience factory methods.
pub struct Builder<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> Builder<'ctx> {
    /// Creates a new builder.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Returns a reference to the context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Returns the unknown location.
    pub fn unknown_location(&self) -> Location<'ctx> {
        Location::unknown(self.context)
    }

    /// Creates a `!felt.type`.
    pub fn felt_type(&self) -> Type<'ctx> {
        // TODO: eventually we will want to use whatever field the circuit output
        // is parameterized on, but that will also require possibly injecting a field
        // spec at the module level for unsupported fields.
        FeltType::with_field(self.context, "mersenne31").into()
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

    /// Get a constant index-type integer attribute
    #[inline]
    pub fn index_attr(&self, integer: i64) -> Attribute<'ctx> {
        self.int_attr(self.index_type(), integer)
    }

    /// Create a constant felt attribute.
    pub fn felt_attr(&self, value: u64) -> FeltConstAttribute<'ctx> {
        FeltConstAttribute::new(self.context, value)
    }

    /// Create a constant int attribute of the given int type.
    #[inline]
    pub fn int_attr(&self, r#type: Type<'ctx>, integer: i64) -> Attribute<'ctx> {
        IntegerAttribute::new(r#type, integer).into()
    }

    /// Get a register type, which is a two-element felt array.
    /// TODO: This is probably too representation dependent, move elsewhere.
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
pub struct OpsBuilder<'ctx, 'sco> {
    builder: Builder<'ctx>,
    scope: FuncDefOpRef<'ctx, 'sco>,
    /// Cache of constant op values of specified type at the beginning of the
    /// function scope. Using a BTreeMap since [Type] is not hashable.
    const_vals: RefCell<BTreeMap<ConstOpKey<'ctx>, Value<'ctx, 'sco>>>,
}

impl<'ctx, 'sco> OpsBuilder<'ctx, 'sco> {
    /// Creates a new builder.
    pub fn new(context: &'ctx Context, scope: FuncDefOpRef<'ctx, 'sco>) -> Self {
        Self {
            scope,
            builder: Builder::new(context),
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
    pub fn insert_op_with_no_results_at_start(&self, op: Operation<'ctx>) -> Result<()> {
        let _ = self.insert_operation(InsertionPoint::Start, op)?;
        Ok(())
    }

    /// Inserts an operation with results at the start.
    #[inline]
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
    pub fn insert_op_with_no_results_at(&self, pos: usize, op: Operation<'ctx>) -> Result<()> {
        let _ = self.insert_operation(InsertionPoint::At(pos), op)?;
        Ok(())
    }

    /// Inserts an operation with results at the given position.
    #[inline]
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
                    .skip(pos - 1)
                    .next()
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

    /// Append a boolean constraint for the given value.
    pub fn append_boolean_constraint(&self, val: Value<'ctx, 'sco>) -> Result<()> {
        assert_eq!(val.r#type(), self.felt_type());
        let unk = self.unknown_location();
        let zero = self.get_constant_from_start(self.felt_type(), 0)?;
        let one = self.get_constant_from_start(self.felt_type(), 1)?;
        let minus_one = self.append_op_with_result(felt::sub(unk, val, one)?)?;
        let product = self.append_op_with_result(felt::mul(unk, val, minus_one)?)?;
        self.append_op_with_no_results(constrain::eq(unk, product, zero))
    }

    /// Append a range constraint for the given value.
    /// Enforces that `val` must be within `width`.
    pub fn append_range_constraint(&self, val: Value<'ctx, 'sco>, width: usize) -> Result<()> {
        assert_eq!(val.r#type(), self.felt_type());
        let unk = self.unknown_location();
        let bound = self.get_constant_from_start(self.felt_type(), 1 << width)?;
        let bound_check = self.append_op_with_result(bool::lt(unk, val, bound)?)?;
        let truth = self.get_constant_from_start(self.int_type(1), 1)?;
        self.append_op_with_no_results(constrain::eq(unk, bound_check, truth))
    }

    /// Get the value from the contained function scope.
    pub fn get_arg_value(&self, arg_no: usize) -> Result<Value<'ctx, 'sco>> {
        Ok(self.scope.argument(arg_no)?.into())
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

    // Perform the index constant insertion without producing a return value.
    pub fn insert_constant_at_start(&self, r#type: Type<'ctx>, i: u64) -> Result<()> {
        let _ = self.get_constant_from_start(r#type, i)?;
        Ok(())
    }
}

impl<'ctx> Deref for OpsBuilder<'ctx, '_> {
    type Target = Builder<'ctx>;

    fn deref(&self) -> &Self::Target {
        &self.builder
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
    /// List of members. Contains the name, type and wether is marked public or not.
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
