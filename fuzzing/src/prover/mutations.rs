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
    NoOp,
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
    let _ = rng;

    MutatedInput {
        original: SeedCaseRef {
            seed_program: seed_case.seed_program.clone(),
            circuit: seed_case.circuit,
        },
        mutated_input: seed_case.base_input.clone(),
        mutations: vec![MutationRecord {
            kind: MutationKind::NoOp,
            target: "proof_input".to_owned(),
            summary: "left seed input unchanged".to_owned(),
        }],
    }
}
