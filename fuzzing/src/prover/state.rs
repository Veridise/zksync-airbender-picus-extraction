use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::prover::circuits::CircuitRegistry;
use crate::prover::crashes::BugReport;
use crate::prover::crashes::CrashArtifact;
use crate::prover::seeds::expand_seed_cases;
use crate::prover::seeds::load_or_create_cache_entries;
use crate::prover::seeds::CacheEntry;
use crate::prover::seeds::SeedCase;
use crate::prover::seeds::SeedProgram;
use crate::prover::FuzzerConfig;

/// In-memory state accumulated across a fuzzing run.
#[derive(Debug, Default)]
pub struct FuzzerState {
    /// Seed programs discovered from the input corpus directory.
    pub programs: Vec<SeedProgram>,
    /// Cache entries loaded or constructed during initialization.
    pub cache_entries: Vec<CacheEntry>,
    /// Flattened per-circuit seed cases derived from the cache.
    pub seed_cases: Vec<SeedCase>,
    /// Next crash id to allocate when persisting a bug report.
    next_crash_id: u64,
}

impl FuzzerState {
    /// Builds the initial in-memory fuzzer state from the configured corpus and output dirs.
    pub fn new(config: &FuzzerConfig, registry: &CircuitRegistry) -> io::Result<Self> {
        let programs = SeedProgram::find_programs(&config.input_dir)?;
        let cache_entries = load_or_create_cache_entries(&programs, registry, &config.cache_dir)?;
        let seed_cases = expand_seed_cases(&cache_entries);
        let next_crash_id = discover_next_crash_id(&config.crash_dir)?;

        Ok(Self {
            programs,
            cache_entries,
            seed_cases,
            next_crash_id,
        })
    }

    /// Allocates a new crash id, persists the corresponding artifact, and returns its path.
    pub fn save_bug(&mut self, report: BugReport, crash_dir: &Path) -> io::Result<PathBuf> {
        let crash_id = self.next_crash_id;
        self.next_crash_id += 1;

        let artifact = CrashArtifact::new(crash_id, report);
        let path = crash_dir.join(artifact.file_name());
        artifact.write(&path)?;

        Ok(path)
    }
}

fn discover_next_crash_id(crash_dir: &Path) -> io::Result<u64> {
    let mut max_seen = None;

    for entry in std::fs::read_dir(crash_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };

        if let Some(raw_id) = name
            .strip_prefix("id:")
            .and_then(|rest| rest.split(',').next())
        {
            if let Ok(id) = raw_id.parse::<u64>() {
                max_seen = Some(max_seen.map_or(id, |current: u64| current.max(id)));
            }
        }
    }

    Ok(max_seen.map_or(0, |id| id + 1))
}
