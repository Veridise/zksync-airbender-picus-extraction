use prover::common_constants::ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX;
use prover::common_constants::INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX;
use prover::common_constants::JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX;
use prover::common_constants::LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX;
use prover::common_constants::LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX;
use prover::common_constants::MUL_DIV_CIRCUIT_FAMILY_IDX;
use prover::common_constants::SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX;
use prover::cs::tables::TableDriver;

use crate::prover::crashes::BugType;
use crate::prover::mutations::MutatedInput;
use crate::prover::seeds::StoredProofInputs;
use crate::prover::GeneratedProof;
use crate::rv32im::prover::circuits::add_sub_lui_auipc_mop::AddSubLuiAuipcMop;
use crate::rv32im::prover::circuits::add_sub_lui_auipc_mop::prove_add_sub_lui_auipc_mop_from_inputs;
use crate::rv32im::prover::circuits::add_sub_lui_auipc_mop::validate_add_sub_lui_auipc_mop_proof;
use crate::rv32im::prover::circuits::CircuitProver;
use crate::rv32im::prover::PreparedExecution;
use crate::rv32im::vm::VMSnapshot;

const BLAKE_DELEGATION_KIND_MASK: u8 = 0x20;
const KECCAK_DELEGATION_KIND_MASK: u8 = 0x40;
const DELEGATION_KIND_MASK: u8 = BLAKE_DELEGATION_KIND_MASK | KECCAK_DELEGATION_KIND_MASK;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CircuitKind {
    AddSubLuiAuipcMop = ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX,
    JumpBranchSlt = JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX,
    XorAndOrShiftCsr = SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
    MulDiv = MUL_DIV_CIRCUIT_FAMILY_IDX,
    LoadStore = LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX,
    SubwordLoadStore = LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX,
    InitsAndTeardowns = INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX,
    BlakeDelegation = SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX | BLAKE_DELEGATION_KIND_MASK,
    KeccakDelegation = SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX | KECCAK_DELEGATION_KIND_MASK,
}

#[derive(Debug)]
pub struct CircuitRegistry {
    circuits: Vec<CircuitKind>,
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
            // Self::JumpBranchSlt,
            // Self::XorAndOrShiftCsr,
            // Self::MulDiv,
            // Self::LoadStore,
            // Self::SubwordLoadStore,
            // Self::InitsAndTeardowns,
            // Self::BlakeDelegation,
            // Self::KeccakDelegation,
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

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn family_idx(self) -> u8 {
        match self {
            Self::BlakeDelegation | Self::KeccakDelegation => self.as_u8() & !DELEGATION_KIND_MASK,
            _ => self.as_u8(),
        }
    }

    pub fn from_family_idx(family_idx: u8) -> Option<Self> {
        match family_idx {
            ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX => Some(Self::AddSubLuiAuipcMop),
            JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX => Some(Self::JumpBranchSlt),
            SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX => Some(Self::XorAndOrShiftCsr),
            MUL_DIV_CIRCUIT_FAMILY_IDX => Some(Self::MulDiv),
            LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX => Some(Self::LoadStore),
            LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX => Some(Self::SubwordLoadStore),
            INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX => Some(Self::InitsAndTeardowns),
            _ => None,
        }
    }
}

impl CircuitRegistry {
    pub fn new() -> Self {
        let circuits = CircuitKind::all().iter().copied().collect();

        Self { circuits }
    }

    pub fn circuits(&self) -> &[CircuitKind] {
        &self.circuits
    }

    pub fn generate_inputs(
        &self,
        kind: CircuitKind,
        snapshot: VMSnapshot,
        prepared: &PreparedExecution,
    ) -> StoredProofInputs {
        let mut table_driver = TableDriver::new();
        match kind {
            CircuitKind::AddSubLuiAuipcMop => StoredProofInputs::AddSubLuiAuipcMop(
                AddSubLuiAuipcMop.create_proof_input(snapshot, prepared, &mut table_driver),
            ),
            CircuitKind::JumpBranchSlt => todo!(),
            CircuitKind::XorAndOrShiftCsr => todo!(),
            CircuitKind::MulDiv => todo!(),
            CircuitKind::LoadStore => todo!(),
            CircuitKind::SubwordLoadStore => todo!(),
            CircuitKind::InitsAndTeardowns => todo!(),
            CircuitKind::BlakeDelegation => todo!(),
            CircuitKind::KeccakDelegation => todo!(),
        }
    }

    pub fn prove(&self, input: &StoredProofInputs) -> ProverAttempt {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match input {
            StoredProofInputs::AddSubLuiAuipcMop(inputs) => ProverAttempt::Success(
                GeneratedProof::AddSubLuiAuipcMop(prove_add_sub_lui_auipc_mop_from_inputs(
                    inputs.clone(),
                )),
            ),
            StoredProofInputs::JumpBranchSlt(_) => todo!(),
            StoredProofInputs::XorAndOrShiftCsr(_) => todo!(),
            StoredProofInputs::MulDiv(_) => todo!(),
            StoredProofInputs::LoadStore(_) => todo!(),
            StoredProofInputs::SubwordLoadStore(_) => todo!(),
            StoredProofInputs::InitsAndTeardowns(_) => todo!(),
            StoredProofInputs::BlakeDelegation(_) => todo!(),
            StoredProofInputs::KeccakDelegation(_) => todo!(),
        }));

        match result {
            Ok(attempt) => attempt,
            Err(_) => ProverAttempt::Crash,
        }
    }

    pub fn validate(&self, input: &StoredProofInputs, proof: &GeneratedProof) -> BugType {
        match (input, proof) {
            (
                StoredProofInputs::AddSubLuiAuipcMop(inputs),
                GeneratedProof::AddSubLuiAuipcMop(proof),
            ) => match validate_add_sub_lui_auipc_mop_proof(inputs, proof) {
                Ok(()) => BugType::ValidationBug,
                Err(()) => BugType::ProofGenerationBug,
            },
            _ => panic!(
                "proof/input circuit mismatch during validation: input={:?}",
                input.circuit()
            ),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_kind_encodings_match_expected_values() {
        let cases = [
            (
                CircuitKind::AddSubLuiAuipcMop,
                ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX,
                ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::JumpBranchSlt,
                JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX,
                JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::XorAndOrShiftCsr,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::MulDiv,
                MUL_DIV_CIRCUIT_FAMILY_IDX,
                MUL_DIV_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::LoadStore,
                LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX,
                LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::SubwordLoadStore,
                LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX,
                LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::InitsAndTeardowns,
                INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX,
                INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::BlakeDelegation,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX | BLAKE_DELEGATION_KIND_MASK,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
            ),
            (
                CircuitKind::KeccakDelegation,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX | KECCAK_DELEGATION_KIND_MASK,
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
            ),
        ];

        for (kind, encoded, family_idx) in cases {
            assert_eq!(kind.as_u8(), encoded, "{kind:?} encoded value mismatch");
            assert_eq!(
                kind.family_idx(),
                family_idx,
                "{kind:?} family idx mismatch"
            );
        }
    }

    #[test]
    fn circuit_kind_from_family_idx_matches_expected_values() {
        let cases = [
            (
                ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::AddSubLuiAuipcMop),
            ),
            (
                JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::JumpBranchSlt),
            ),
            (
                SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::XorAndOrShiftCsr),
            ),
            (MUL_DIV_CIRCUIT_FAMILY_IDX, Some(CircuitKind::MulDiv)),
            (
                LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::LoadStore),
            ),
            (
                LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::SubwordLoadStore),
            ),
            (
                INITS_AND_TEARDOWNS_FORMAL_CIRCUIT_FAMILY_IDX,
                Some(CircuitKind::InitsAndTeardowns),
            ),
            (0, None),
            (128, None),
        ];

        for (family_idx, expected) in cases {
            assert_eq!(CircuitKind::from_family_idx(family_idx), expected);
        }
    }
}
