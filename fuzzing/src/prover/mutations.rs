use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

use crate::prover::seeds::SeedCase;
use crate::prover::seeds::StoredProofInputs;
use crate::prover::SeedCaseRef;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MutationRecord {
    summary: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MutatedInput {
    pub original: SeedCaseRef,
    pub mutated_input: StoredProofInputs,
    pub mutations: Vec<MutationRecord>,
}

pub trait Mutator {
    fn mutate(&self, seed_case: &SeedCase, rng: &mut StdRng) -> MutatedInput;
}

pub struct NoOpMutator;

impl Mutator for NoOpMutator {
    fn mutate(&self, seed_case: &SeedCase, _: &mut StdRng) -> MutatedInput {
        MutatedInput {
            original: SeedCaseRef {
                seed_program: seed_case.seed_program.clone(),
                circuit: seed_case.circuit,
            },
            mutated_input: seed_case.base_input.clone(),
            mutations: vec![MutationRecord {
                summary: "no-op mutator".to_owned(),
            }],
        }
    }
}

pub struct MutatorRegistry {
    mutators: Vec<Box<dyn Mutator>>,
}

impl MutatorRegistry {
    /// Empty registry used for seed validation.
    ///
    /// It actually has one mutator, the [`NoOpMutator`].
    pub fn empty() -> Self {
        Self {
            mutators: vec![Box::new(NoOpMutator)],
        }
    }

    pub fn new() -> Self {
        Self {
            /// TODO: Change with actual mutators
            mutators: vec![Box::new(NoOpMutator)],
        }
    }

    /// Chooses a mutator from the registry at random.
    pub fn choose(&self, rng: &mut StdRng) -> &dyn Mutator {
        self.mutators
            .choose(rng)
            .map(|b| b.as_ref())
            .expect("registry not empty")
    }
}
