use std::cell::OnceCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::hash::Hash;
use std::hash::Hasher;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use prover::worker::Worker;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;

use crate::prover::circuits::CircuitKind;
use crate::prover::circuits::CircuitRegistry;
use crate::rv32im::binary::Binary;
use crate::rv32im::prover::circuits::ProofInputs;
use crate::rv32im::prover::prepare_execution;
use crate::rv32im::VM;

#[derive(Debug)]
pub struct SeedProgram {
    name: String,
    binary_path: PathBuf,
    text_path: PathBuf,
    binary_bytes: OnceCell<Vec<u8>>,
    text_bytes: OnceCell<Vec<u8>>,
    hash: OnceCell<String>,
}

impl SeedProgram {
    pub fn new(name: String, binary_path: PathBuf, text_path: PathBuf) -> Self {
        Self {
            name,
            binary_path,
            text_path,
            binary_bytes: OnceCell::new(),
            text_bytes: OnceCell::new(),
            hash: OnceCell::new(),
        }
    }

    pub fn find_programs(input_dir: &Path) -> io::Result<Vec<SeedProgram>> {
        let mut bin_paths = BTreeMap::<String, PathBuf>::new();
        let mut text_paths = BTreeMap::<String, PathBuf>::new();

        for entry in fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if !entry.file_type()?.is_file() {
                continue;
            }

            match path.extension().and_then(OsStr::to_str) {
                Some("bin") => {
                    if let Some(stem) = file_stem_string(&path) {
                        bin_paths.insert(stem, path);
                    }
                }
                Some("text") => {
                    if let Some(stem) = file_stem_string(&path) {
                        text_paths.insert(stem, path);
                    }
                }
                _ => {}
            }
        }

        let mut programs = Vec::with_capacity(bin_paths.len());
        for (name, bin_path) in bin_paths {
            let text_path = text_paths.remove(&name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing .text file for seed program `{name}`"),
                )
            })?;

            programs.push(SeedProgram::new(name, bin_path, text_path));
        }

        Ok(programs)
    }

    pub fn binary(&self) -> io::Result<Binary<'_>> {
        Ok(Binary::new(self.binary_bytes()?, Some(self.text_bytes()?)))
    }

    pub fn cache_file_name(&self) -> io::Result<String> {
        Ok(format!("{}-{}.data", self.name, self.hash()?))
    }

    fn hash(&self) -> io::Result<&str> {
        if let Some(hash) = self.hash.get() {
            return Ok(hash);
        }

        let hash = short_program_hash(self.binary_bytes()?, self.text_bytes()?);
        let _ = self.hash.set(hash);
        Ok(self.hash.get().expect("content hash initialized"))
    }

    fn binary_bytes(&self) -> io::Result<&[u8]> {
        if let Some(bytes) = self.binary_bytes.get() {
            return Ok(bytes);
        }

        let bytes = fs::read(&self.binary_path)?;
        let _ = self.binary_bytes.set(bytes);
        Ok(self.binary_bytes.get().expect("binary bytes initialized"))
    }

    fn text_bytes(&self) -> io::Result<&[u8]> {
        if let Some(bytes) = self.text_bytes.get() {
            return Ok(bytes);
        }

        let bytes = fs::read(&self.text_path)?;
        let _ = self.text_bytes.set(bytes);
        Ok(self.text_bytes.get().expect("text bytes initialized"))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    pub seed: String,
    pub inputs: Vec<StoredProofInputs>,
}

impl CacheEntry {
    pub fn load_or_create(
        program: SeedProgram,
        registry: &CircuitRegistry,
        cache_dir: &Path,
    ) -> io::Result<Self> {
        let path = cache_dir.join(program.cache_file_name()?);
        let entry = if path.exists() {
            Self::load(&path)?
        } else {
            let entry = Self::create(program, registry)?;
            entry.write(&path)?;
            entry
        };

        Ok(entry)
    }

    fn create(program: SeedProgram, registry: &CircuitRegistry) -> io::Result<Self> {
        let binary = program.binary()?;
        let mut vm = VM::new(&binary);
        vm.run();
        let worker = Worker::new_with_num_threads(1);
        let snapshot = vm.snapshot();
        let prepared = prepare_execution(snapshot, &worker);

        Ok(Self {
            seed: program.name,
            inputs: registry
                .circuits()
                .iter()
                .map(|kind| registry.generate_inputs(*kind, snapshot, &prepared))
                .collect(),
        })
    }

    fn load(path: &Path) -> io::Result<Self> {
        let contents = fs::read(path)?;
        serde_json::from_slice(&contents).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to deserialize cache entry `{}`: {err}",
                    path.display()
                ),
            )
        })
    }

    fn write(&self, path: &Path) -> io::Result<()> {
        let payload = serde_json::to_vec_pretty(self).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to serialize cache entry `{}`: {err}",
                    path.display()
                ),
            )
        })?;
        fs::write(path, payload)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SeedCase {
    pub seed_program: String,
    pub circuit: CircuitKind,
    pub base_input: StoredProofInputs,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum StoredProofInputs {
    AddSubLuiAuipcMop(ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>),
    JumpBranchSlt(()),
    XorAndOrShiftCsr(()),
    MulDiv(()),
    LoadStore(()),
    SubwordLoadStore(()),
    InitsAndTeardowns(()),
    BlakeDelegation(()),
    KeccakDelegation(()),
}

impl StoredProofInputs {
    pub fn circuit(&self) -> CircuitKind {
        match self {
            Self::AddSubLuiAuipcMop(inputs) => CircuitKind::from_family_idx(inputs.family_idx())
                .expect("stored proof inputs contain an unsupported circuit family idx"),
            Self::JumpBranchSlt(_) => CircuitKind::JumpBranchSlt,
            Self::XorAndOrShiftCsr(_) => CircuitKind::XorAndOrShiftCsr,
            Self::MulDiv(_) => CircuitKind::MulDiv,
            Self::LoadStore(_) => CircuitKind::LoadStore,
            Self::SubwordLoadStore(_) => CircuitKind::SubwordLoadStore,
            Self::InitsAndTeardowns(_) => CircuitKind::InitsAndTeardowns,
            Self::BlakeDelegation(_) => CircuitKind::BlakeDelegation,
            Self::KeccakDelegation(_) => CircuitKind::KeccakDelegation,
        }
    }
}

pub fn expand_seed_cases(entries: impl IntoIterator<Item = CacheEntry>) -> Vec<SeedCase> {
    entries
        .into_iter()
        .flat_map(|entry| {
            entry.inputs.into_iter().map(move |base_input| SeedCase {
                seed_program: entry.seed.clone(),
                circuit: base_input.circuit(),
                base_input,
            })
        })
        .collect()
}

pub fn choose_seed_case(seed_cases: &[SeedCase], rng: &mut StdRng) -> io::Result<SeedCase> {
    seed_cases
        .choose(rng)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no seed cases available"))
}

fn file_stem_string(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(OsStr::to_str)
        .map(ToOwned::to_owned)
}

fn short_program_hash(bin_bytes: &[u8], text_bytes: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    bin_bytes.hash(&mut hasher);
    text_bytes.hash(&mut hasher);
    format!("{:016x}", hasher.finish())[..8].to_owned()
}
