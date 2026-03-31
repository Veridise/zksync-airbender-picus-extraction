use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

use crate::prover::mutations::nop::NoOpMutator;
use crate::prover::seeds::SeedCase;
use crate::prover::seeds::StoredProofInputs;
use crate::prover::SeedCaseRef;
use crate::rv32im::prover::circuits::ProofInputs;

mod nop;

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

impl MutatedInput {
    pub fn new(seed: &SeedCase, mutated_input: StoredProofInputs, descr: &str) -> Self {
        Self {
            original: SeedCaseRef {
                seed_program: seed.seed_program.clone(),
                circuit: seed.circuit,
            },
            mutated_input,
            mutations: vec![MutationRecord {
                summary: descr.to_owned(),
            }],
        }
    }
}

pub trait Mutator {
    fn name(&self) -> &'static str;

    fn mutate(&self, seed_case: &SeedCase, rng: &mut StdRng) -> MutatedInput {
        let mut mutated = seed_case.base_input.clone();
        self.mutate_input(&mut mutated, rng);
        MutatedInput::new(seed_case, mutated, self.name())
    }

    fn mutate_input(&self, input: &mut StoredProofInputs, rng: &mut StdRng) {
        match input {
            StoredProofInputs::AddSubLuiAuipcMop(proof_inputs)
            | StoredProofInputs::JumpBranchSlt(proof_inputs)
            | StoredProofInputs::XorAndOrShiftCsr(proof_inputs)
            | StoredProofInputs::MulDiv(proof_inputs) => {
                self.mutate_non_mem_inputs(proof_inputs, rng)
            }

            StoredProofInputs::LoadStore(proof_inputs, _)
            | StoredProofInputs::SubwordLoadStore(proof_inputs, _) => {
                self.mutate_mem_inputs(proof_inputs, rng)
            }

            StoredProofInputs::InitsAndTeardowns(_) => {}
            StoredProofInputs::BlakeDelegation(_) => {}
            StoredProofInputs::KeccakDelegation(_) => {}
        };
    }

    fn mutate_non_mem_inputs(
        &self,
        input: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    );

    fn mutate_mem_inputs(
        &self,
        input: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    );
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
            // TODO: Change with actual mutators
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
