//! Utilities for specifying the output of the code generation process.

#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    /// LLZK IR
    Llzk,
    /// Picus Constraint Language MLIR Dialect (mid-level IR between LLZK and PCL)
    PclMlir,
    /// Picus Constraint Language
    Pcl,
}

impl OutputFormat {
    /// Standard file extension for the given output format.
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Llzk => "llzk",
            OutputFormat::PclMlir => "mlir",
            OutputFormat::Pcl => "pcl",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Llzk => write!(f, "llzk"),
            OutputFormat::PclMlir => write!(f, "pcl-mlir"),
            OutputFormat::Pcl => write!(f, "pcl"),
        }
    }
}
