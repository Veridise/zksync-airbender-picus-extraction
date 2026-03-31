use std::fs;
use std::io;
use std::path::PathBuf;

use clap::Parser;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use rand::rngs::StdRng;
use rand::SeedableRng;

mod circuits;
mod crashes;
mod mutations;
mod seeds;
mod state;

use circuits::attempt_proof_generation;
use circuits::classify_generated_proof;
use circuits::CircuitKind;
use circuits::CircuitRegistry;
use circuits::ProverAttempt;
use crashes::BugReport;
use crashes::ExecutionOutcome;
use mutations::mutate_seed_case;
use state::FuzzerState;

/// Command-line arguments for the prover fuzzer scaffold.
#[derive(Debug, Parser)]
pub struct Cli {
    /// Directory containing `.bin`/`.text` seed program pairs.
    #[arg(short = 'i', long)]
    pub input_dir: PathBuf,
    /// Directory used to store fuzzer state such as cache entries and crashes.
    #[arg(short = 'o', long)]
    pub output_dir: PathBuf,
    /// Number of fuzz-loop iterations to execute.
    #[arg(long, default_value_t = 100)]
    pub iterations: usize,
    /// Reserved sampling parameter for future loop heuristics.
    #[arg(long, default_value_t = 1)]
    pub samples: usize,
    /// RNG seed used to make scaffold behavior reproducible.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

/// Resolved runtime configuration derived from the CLI.
#[derive(Debug, Clone)]
pub struct FuzzerConfig {
    /// Input corpus directory passed by the user.
    pub input_dir: PathBuf,
    /// Root output directory passed by the user.
    pub output_dir: PathBuf,
    /// Cache directory nested under [`FuzzerConfig::output_dir`].
    pub cache_dir: PathBuf,
    /// Crash directory nested under [`FuzzerConfig::output_dir`].
    pub crash_dir: PathBuf,
    /// Number of fuzz-loop iterations to execute.
    pub iterations: usize,
    /// Reserved sampling parameter for future loop heuristics.
    pub samples: usize,
    /// RNG seed used by the fuzzer.
    pub seed: u64,
}

/// Top-level prover fuzzer orchestrator.
pub struct Fuzzer {
    /// Static runtime configuration.
    config: FuzzerConfig,
    /// Mutable runtime state.
    state: FuzzerState,
    /// Deterministic RNG used for seed selection and mutation.
    rng: StdRng,
    /// Registry of circuit adapters used by the scaffold.
    registry: CircuitRegistry,
}

/// Identifies the original seed/circuit pair from which a mutation was derived.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SeedCaseRef {
    /// Name of the seed program used as the mutation base.
    pub seed_program: String,
    /// Circuit family targeted by the mutated input.
    pub circuit: CircuitKind,
}

/// Proof object returned by the current scaffold.
#[derive(Clone, Debug)]
pub enum GeneratedProof {
    AddSubLuiAuipcMop(UnrolledModeProof),
}

/// Runs the prover fuzzer scaffold from parsed CLI arguments.
pub fn run(cli: Cli) {
    let config = cli.into();
    let mut fuzzer = Fuzzer::new(config);

    if let Err(err) = fuzzer.initialize().and_then(|()| fuzzer.run_loop()) {
        panic!("prover-fuzz failed: {err}");
    }
}

impl From<Cli> for FuzzerConfig {
    /// Expands CLI arguments into the runtime configuration shape used by the fuzzer.
    fn from(cli: Cli) -> Self {
        FuzzerConfig {
            cache_dir: cli.output_dir.join("cache"),
            crash_dir: cli.output_dir.join("crashes"),
            input_dir: cli.input_dir,
            output_dir: cli.output_dir,
            iterations: cli.iterations,
            samples: cli.samples,
            seed: cli.seed,
        }
    }
}

impl Fuzzer {
    /// Constructs a new fuzzer with deterministic RNG state and an empty runtime state.
    fn new(config: FuzzerConfig) -> Self {
        Self {
            rng: StdRng::seed_from_u64(config.seed),
            config,
            state: FuzzerState::default(),
            registry: CircuitRegistry::new(),
        }
    }

    /// Prepares directories, loads seed programs, and materializes the in-memory seed database.
    fn initialize(&mut self) -> io::Result<()> {
        prepare_output_dirs(&self.config)?;
        self.state = FuzzerState::new(&self.config, &self.registry)?;

        Ok(())
    }

    /// Executes the main fuzz loop for the configured number of iterations.
    fn run_loop(&mut self) -> io::Result<()> {
        for iteration in 0..self.config.iterations {
            let outcome = self.run_one_iteration(iteration)?;
            if let ExecutionOutcome::Interesting(report) = outcome {
                self.state.save_bug(report, &self.config.crash_dir)?;
            }
        }

        Ok(())
    }

    /// Runs one fuzz iteration: choose a seed, mutate it, attempt proving, and classify the result.
    fn run_one_iteration(&mut self, _iteration: usize) -> io::Result<ExecutionOutcome> {
        let seed_case = seeds::choose_seed_case(&self.state.seed_cases, &mut self.rng)?;
        let mutated = mutate_seed_case(&seed_case, &mut self.rng);

        match attempt_proof_generation(&mutated, &self.registry) {
            ProverAttempt::Crash => Ok(ExecutionOutcome::DiscardedProverCrash),
            ProverAttempt::Success(proof) => {
                let bug_type = classify_generated_proof(&mutated, &proof, &self.registry);
                Ok(ExecutionOutcome::Interesting(BugReport::new(
                    mutated, bug_type,
                )))
            }
        }
    }
}

/// Ensures the fuzzer output root and its required subdirectories exist.
fn prepare_output_dirs(config: &FuzzerConfig) -> io::Result<()> {
    fs::create_dir_all(&config.output_dir)?;
    fs::create_dir_all(&config.cache_dir)?;
    fs::create_dir_all(&config.crash_dir)?;
    Ok(())
}
