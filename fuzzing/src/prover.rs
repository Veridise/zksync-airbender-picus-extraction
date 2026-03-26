use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;

use prover::cs::one_row_compiler::CompiledCircuitArtifact;
use prover::field::Mersenne31Field;
use prover::nd_source_std::set_iterator;
use prover::nd_source_std::ThreadLocalBasedSource;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;
use serde::Serialize;
use verifier_common::proof_flattener::flatten_full_unrolled_proof;
use verifier_common::DefaultLeafInclusionVerifier;

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
