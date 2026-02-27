use prover::field::Mersenne31Field;

/// Trait for obtaining information from the circuit's field that is useful for building IR.
pub trait FieldInfo {
    /// Returns the name of the field in a format compatible with LLZK.
    fn field_name() -> &'static str;
}

impl FieldInfo for Mersenne31Field {
    fn field_name() -> &'static str {
        "mersenne31"
    }
}
