use rand::rngs::StdRng;

use crate::prover::mutations::MutatedInput;
use crate::prover::mutations::MutationRecord;
use crate::prover::mutations::Mutator;
use crate::prover::seeds::SeedCase;
use crate::prover::SeedCaseRef;

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
