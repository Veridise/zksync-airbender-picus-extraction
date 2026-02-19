//! Builder types for encapsulating common codegen tasks.
//!
//! Contains:
//! - A generic builder with stateless factory methods.
//! - An operations builder meant for creating ops inside a function.
//! - A struct builder.

use std::ops::Deref;

use anyhow::{anyhow, Result};
use llzk::prelude::*;

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

    /// Creates a `!felt.type`.
    pub fn felt_type(&self) -> Type<'ctx> {
        FeltType::new(self.context).into()
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

/// Operations builder that handles insertion of operations in the target function.
pub struct OpsBuilder<'ctx, 'sco> {
    builder: Builder<'ctx>,
    scope: FuncDefOpRef<'ctx, 'sco>,
}

impl<'ctx, 'sco> OpsBuilder<'ctx, 'sco> {
    /// Creates a new builder.
    pub(crate) fn new(scope: FuncDefOpRef<'ctx, 'sco>) -> Self {
        let context = unsafe { scope.context().to_ref() };
        Self {
            scope,
            builder: Builder::new(context),
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
    members: Vec<(&'str str, Type<'ctx>, bool)>,
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
    pub fn with_member(
        &mut self,
        name: &'str str,
        r#type: Type<'ctx>,
        is_public: bool,
    ) -> &mut Self {
        self.members.push((name, r#type, is_public));
        self
    }

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
