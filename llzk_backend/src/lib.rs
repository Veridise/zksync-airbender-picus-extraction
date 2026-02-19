use anyhow::anyhow;
use llzk::prelude::*;

use crate::builder::OpsBuilder;

pub mod builder;

/// This enum holds the possible representations for SSA values
pub enum SsaAddress<'ctx, 'val> {
    /// Represents a single variable that is neither an input or an output.
    /// It's encoded as a struct member of [`FeltType`].
    Intermediate(Value<'ctx, 'val>),
}

/// Extension trait for [`StructDefOpLike`] that adds a method for filling the `@constrain` function.
pub trait AddConstraints<'ctx: 'op, 'op>: StructDefOpLike<'ctx, 'op> {
    /// Invokes the callback scoped in `@constrain`.
    ///
    /// All ops added with the [`OpsBuilder`] are automatically added to that function.
    fn add_constraints(
        &'op self,
        f: impl FnOnce(&mut OpsBuilder<'ctx, 'op>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let constrain_fn = self.get_constrain_func().ok_or_else(|| {
            anyhow!(
                "struct {} is missing its @constrain function",
                StructDefOpLike::name(self)
            )
        })?;
        let mut builder = OpsBuilder::new(constrain_fn);
        f(&mut builder)
    }
}
