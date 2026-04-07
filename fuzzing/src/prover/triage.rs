use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::prover::circuits::CircuitRegistry;
use crate::prover::crashes::CrashArtifact;
use crate::prover::crashes::CrashStep;
use crate::prover::seeds::load_seed_case_from_cache;
use crate::prover::seeds::SeedCase;
use crate::prover::seeds::StoredProofInputs;
use crate::prover::triage::analysis::analyze_once;
use crate::prover::triage::analysis::AnalysisTrace;
use crate::prover::triage::analysis::CheckpointDiff;
use crate::prover::triage::report::TriageVerdict;

mod analysis;
mod report;

use report::TriageReport;

/// CLI arguments for replaying and classifying a persisted prover-fuzzer crash.
#[derive(Debug, Clone, Args)]
pub struct TriageCli {
    /// Path to the persisted crash artifact JSON file.
    #[arg(long)]
    pub crash: PathBuf,
    /// Directory used to store fuzzer state such as cache entries and crashes.
    #[arg(short = 'o', long)]
    pub output_dir: PathBuf,
    /// Emit the triage report as JSON after the human-readable summary.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Number of replays to run for each input before trusting the comparison.
    #[arg(long, default_value_t = 2)]
    pub stability_runs: usize,
}

/// Runs offline crash triage against a persisted crash artifact and the cached base seed corpus.
pub fn run(cli: TriageCli) -> anyhow::Result<()> {
    log::info!("Triaging crash {}", cli.crash.display());
    // Triage is intentionally offline: load the recorded crash and recover the original
    // cached seed input so we can compare "base vs mutated" under the same replay path.
    let crash = CrashArtifact::read(&cli.crash)
        .with_context(|| format!("failed to load crash artifact `{}`", cli.crash.display()))?;
    let base = load_seed_case_from_cache(
        &cli.output_dir.join("cache"),
        &crash.seed_program,
        crash.circuit,
    )
    .with_context(|| {
        format!(
            "failed to recover base seed from cache for `{}` / `{}`",
            crash.seed_program,
            crash.circuit.slug()
        )
    })?;
    log::info!("Found crash seed: {} / {}", base.seed_program, base.circuit);

    let registry = CircuitRegistry::new();
    let stability_runs = cli.stability_runs.max(1);
    log::info!("Number of runs per side: {stability_runs}");
    let report = triage_crash(&registry, &crash, &base, stability_runs);

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{report}");
    }

    Ok(())
}

/// Replays the base and mutated inputs, compares their compact traces, and builds the final report.
fn triage_crash(
    registry: &CircuitRegistry,
    crash: &CrashArtifact,
    base: &SeedCase,
    stability_runs: usize,
) -> TriageReport {
    // Replay both sides multiple times first. If either side is unstable under replay we do not
    // trust any later diff and classify the crash as inconclusive instead.
    let base_trace =
        match stable_trace_for_input(registry, &base.base_input, stability_runs, "seed") {
            Ok(trace) => trace,
            Err(instability) => return TriageReport::inconclusive(base, crash, instability),
        };
    let mutated_trace =
        match stable_trace_for_input(registry, &crash.mutated_input, stability_runs, "mutated") {
            Ok(trace) => trace,
            Err(instability) => return TriageReport::inconclusive(base, crash, instability),
        };

    log::info!("Runs completed!");
    let first_diff = base_trace.diff(&mutated_trace);
    TriageReport::new(
        classify_verdict(crash.step, &first_diff),
        crash,
        first_diff,
        None,
        base_trace,
        mutated_trace,
    )
}

/// Replays one fixed input multiple times and rejects it if the compact trace is not stable.
fn stable_trace_for_input(
    registry: &CircuitRegistry,
    input: &StoredProofInputs,
    runs: usize,
    descr: &str,
) -> Result<AnalysisTrace, CheckpointDiff> {
    log::info!("Run 1 for {descr}...");
    // The triage comparison is only useful if replay is deterministic for a fixed input.
    // We therefore require every run of the same input to produce the exact same compact trace.
    let first = analyze_once(registry, input)
        .map_err(|err| CheckpointDiff::proof(format!("analysis replay failed: {err}")))?;
    for run in 1..runs {
        log::info!("Run {run} for {descr}...");
        let next = analyze_once(registry, input)
            .map_err(|err| CheckpointDiff::proof(format!("analysis replay failed: {err}")))?;
        if next != first {
            return Err(first.diff(&next).unwrap_or_else(|| {
                CheckpointDiff::proof("replay produced inconsistent analysis traces".to_owned())
            }));
        }
    }

    Ok(first)
}

/// Classifies the crash using the recorded crash step and the first observed divergence.
fn classify_verdict(step: CrashStep, diff: &Option<CheckpointDiff>) -> TriageVerdict {
    // Verdicts are step-specific:
    // - recorded prover bugs keep crashes once replay diverges at any prover stage
    // - recorded validator bugs also keep prover-internal stage divergence, because the target
    //   under test is still the prover: identical proofs from semantically different internal
    //   executions are interesting and should not be discarded as false positives
    match step {
        CrashStep::Prover | CrashStep::Validator => match diff.as_ref() {
            Some(_) => TriageVerdict::PotentiallyReal,
            None => TriageVerdict::FalsePositive,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use super::*;
    use crate::prover::circuits::CircuitKind;
    use crate::prover::crashes::BugType;
    use crate::prover::seeds::CacheEntry;
    use crate::prover::triage::analysis::oracle::OracleShapeSummary;

    #[test]
    fn validator_step_keeps_stage_level_divergence() {
        let diff = Some(CheckpointDiff::stage2("stage 2 changed".to_owned()));
        assert_eq!(
            classify_verdict(CrashStep::Validator, &diff),
            TriageVerdict::PotentiallyReal
        );
    }

    #[test]
    fn prover_step_keeps_proof_level_diff() {
        let diff = Some(CheckpointDiff::proof("proof changed".to_owned()));
        assert_eq!(
            classify_verdict(CrashStep::Prover, &diff),
            TriageVerdict::PotentiallyReal
        );
    }

    #[test]
    fn load_seed_case_from_cache_recovers_matching_input() {
        let temp = temp_dir("triage-cache");
        let cache_path = temp.join("seed.data");
        let entry = CacheEntry {
            seed: "seed_program".to_owned(),
            inputs: vec![StoredProofInputs::InitsAndTeardowns(())],
        };
        fs::write(
            &cache_path,
            serde_json::to_vec_pretty(&entry).expect("serialize cache entry"),
        )
        .expect("write cache entry");

        let recovered =
            load_seed_case_from_cache(&temp, "seed_program", CircuitKind::InitsAndTeardowns)
                .expect("recover seed case");

        assert_eq!(recovered.seed_program, "seed_program");
        assert_eq!(recovered.circuit, CircuitKind::InitsAndTeardowns);
        assert_eq!(
            recovered.base_input,
            StoredProofInputs::InitsAndTeardowns(())
        );

        fs::remove_dir_all(&temp).expect("cleanup temp dir");
    }

    #[test]
    fn load_seed_case_from_cache_errors_for_missing_circuit() {
        let temp = temp_dir("triage-miss");
        let cache_path = temp.join("seed.data");
        let entry = CacheEntry {
            seed: "seed_program".to_owned(),
            inputs: vec![StoredProofInputs::InitsAndTeardowns(())],
        };
        fs::write(
            &cache_path,
            serde_json::to_vec_pretty(&entry).expect("serialize cache entry"),
        )
        .expect("write cache entry");

        let err = load_seed_case_from_cache(&temp, "seed_program", CircuitKind::AddSubLuiAuipcMop)
            .expect_err("missing circuit must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);

        fs::remove_dir_all(&temp).expect("cleanup temp dir");
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{unique}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }
}
