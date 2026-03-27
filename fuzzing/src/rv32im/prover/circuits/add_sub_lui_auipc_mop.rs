use prover::common_constants;
use prover::common_constants::ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX;
use prover::cs::machine::ops::unrolled::compile_unrolled_circuit_state_transition;
use prover::cs::one_row_compiler::CompiledCircuitArtifact;
use prover::cs::tables::TableDriver;
use prover::field::Field as _;
use prover::field::Mersenne31Field;
use prover::field::Mersenne31Quartic;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use prover::tests::unrolled::add_sub_lui_auipc_mod;
use prover::unrolled::NonMemoryCircuitOracle;
use prover::SimpleWitnessProxy;
use riscv_transpiler::vm::DelegationsAndFamiliesCounters;
use riscv_transpiler::vm::SimpleSnapshotter;
use riscv_transpiler::vm::SimpleTape;
use riscv_transpiler::vm::State;

use crate::rv32im::prover::accumulators::Accumulators;
use crate::rv32im::prover::circuits::CircuitProver;
use crate::rv32im::prover::circuits::NonMemoryCircuitProver;
use crate::rv32im::prover::factories::PreprocessingData;
use crate::rv32im::prover::sets::ReadSets;
use crate::rv32im::prover::sets::WriteSets;
use crate::rv32im::prover::PreparedExecution;
use crate::rv32im::prover::Prover;
use crate::rv32im::prover::TRACE_LEN_LOG2;
use crate::rv32im::types::CountersT;
use crate::rv32im::vm::VMSnapshot;

pub struct AddSubLuiAuipcMop;

impl NonMemoryCircuitProver<ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX> for AddSubLuiAuipcMop {
    fn compile_circuit(&self) -> CompiledCircuitArtifact<Mersenne31Field> {
        use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::*;
        compile_unrolled_circuit_state_transition::<Mersenne31Field>(
            &|cs| add_sub_lui_auipc_mop_table_addition_fn(cs),
            &|cs| add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(cs),
            1 << 20,
            TRACE_LEN_LOG2,
        )
    }

    fn name(&self) -> &str {
        "ADD/SUB/LUI/AUIPC/MOP"
    }

    fn fill_table(&self, _table_driver: &mut TableDriver<Mersenne31Field>) {}

    fn witness_eval(w: &mut SimpleWitnessProxy<NonMemoryCircuitOracle<'_>>) {
        add_sub_lui_auipc_mod::witness_eval_fn(w)
    }

    fn check_constraints(&self, proof: &UnrolledModeProof, is_empty: bool) {
        if is_empty {
            assert_eq!(
                proof.permutation_grand_product_accumulator,
                Mersenne31Quartic::ONE
            );
        }
        assert!(proof.delegation_argument_accumulator.is_none());
    }

    fn accumulate(&self, accumulators: &mut Accumulators, proof: &UnrolledModeProof) {
        accumulators
            .permutation_argument_mut()
            .mul_assign(&proof.permutation_grand_product_accumulator);
    }
}

impl Prover {
    pub fn prove_add_sub_lui_auipc_mop(
        &self,
        accumulators: &mut Accumulators,
        snapshot: VMSnapshot,
        prepared: &PreparedExecution,
        read_sets: &mut ReadSets,
        write_sets: &mut WriteSets,
    ) {
        let circuit = AddSubLuiAuipcMop;
        circuit.prove(
            snapshot,
            prepared,
            accumulators,
            read_sets,
            write_sets,
            self,
            self.worker(),
        );
    }
}
