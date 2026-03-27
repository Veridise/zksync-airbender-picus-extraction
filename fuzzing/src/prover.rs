use clap::Parser;

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long, default_value_t = 100)]
    pub iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub samples: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

pub fn run(cli: Cli) {
    // Add the code here.
}
