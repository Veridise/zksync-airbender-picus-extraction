use std::io;

use crate::prover::GeneratedProof;
use crate::prover::crashes::BugType;
use crate::prover::mutations::MutatedInput;
use crate::prover::seeds::SeedProgram;
use crate::prover::seeds::StoredProofInputs;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CircuitKind {
    AddSubLuiAuipcMop,
    JumpBranchSlt,
    XorAndOrShiftCsr,
    MulDiv,
    LoadStore,
    SubwordLoadStore,
    InitsAndTeardowns,
    BlakeDelegation,
    KeccakDelegation,
}

#[derive(Debug)]
pub struct CircuitRegistry {
    circuits: Vec<CircuitHandle>,
}

#[derive(Clone, Debug)]
pub struct CircuitHandle {
    pub kind: CircuitKind,
}

#[derive(Clone, Debug)]
pub enum ProverAttempt {
    Crash,
    Success(GeneratedProof),
}

impl CircuitKind {
    pub fn all() -> &'static [CircuitKind] {
        &[
            Self::AddSubLuiAuipcMop,
            Self::JumpBranchSlt,
            Self::XorAndOrShiftCsr,
            Self::MulDiv,
            Self::LoadStore,
            Self::SubwordLoadStore,
            Self::InitsAndTeardowns,
            Self::BlakeDelegation,
            Self::KeccakDelegation,
        ]
    }

    pub fn slug(&self) -> &'static str {
        match self {
            Self::AddSubLuiAuipcMop => "add_sub_lui_auipc_mop",
            Self::JumpBranchSlt => "jump_branch_slt",
            Self::XorAndOrShiftCsr => "xor_and_or_shift_csr",
            Self::MulDiv => "mul_div",
            Self::LoadStore => "load_store",
            Self::SubwordLoadStore => "subword_load_store",
            Self::InitsAndTeardowns => "inits_and_teardowns",
            Self::BlakeDelegation => "blake_delegation",
            Self::KeccakDelegation => "keccak_delegation",
        }
    }
}

impl CircuitRegistry {
    pub fn new() -> Self {
        let circuits = CircuitKind::all()
            .iter()
            .copied()
            .map(|kind| CircuitHandle { kind })
            .collect();

        Self { circuits }
    }

    pub fn supports(&self, kind: CircuitKind) -> bool {
        self.circuits.iter().any(|handle| handle.kind == kind)
    }

    pub fn generate_seed_input(
        &self,
        kind: CircuitKind,
        program: &SeedProgram,
    ) -> io::Result<StoredProofInputs> {
        let _ = (kind, program);
        todo!("generate cached proof inputs for a given circuit and seed program")
    }

    pub fn prove(&self, input: &StoredProofInputs) -> ProverAttempt {
        let _ = input;
        todo!("run the prover for a mutated circuit input")
    }

    pub fn validate(&self, input: &StoredProofInputs, proof: &GeneratedProof) -> BugType {
        let _ = (input, proof);
        todo!("validate a generated proof and classify the outcome")
    }
}

pub fn attempt_proof_generation(input: &MutatedInput, registry: &CircuitRegistry) -> ProverAttempt {
    registry.prove(&input.mutated_input)
}

pub fn classify_generated_proof(
    input: &MutatedInput,
    proof: &GeneratedProof,
    registry: &CircuitRegistry,
) -> BugType {
    registry.validate(&input.mutated_input, proof)
}
