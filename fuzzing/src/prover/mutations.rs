use rand::rngs::StdRng;

use crate::prover::SeedCaseRef;
use crate::prover::seeds::SeedCase;
use crate::prover::seeds::StoredProofInputs;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MutationRecord {
    pub kind: MutationKind,
    pub target: String,
    pub summary: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MutationKind {
    ByteFlip,
    FieldReplace,
    Truncate,
    Splice,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MutatedInput {
    pub original: SeedCaseRef,
    pub mutated_input: StoredProofInputs,
    pub mutations: Vec<MutationRecord>,
}

pub fn mutate_seed_case(seed_case: &SeedCase, rng: &mut StdRng) -> MutatedInput {
    let _ = (seed_case, rng);
    todo!("mutate a seed proof input and record the applied mutations")
}
