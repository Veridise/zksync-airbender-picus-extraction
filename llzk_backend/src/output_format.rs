#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    Llzk,
    Pcl,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Llzk => write!(f, "llzk"),
            OutputFormat::Pcl => write!(f, "pcl"),
        }
    }
}
