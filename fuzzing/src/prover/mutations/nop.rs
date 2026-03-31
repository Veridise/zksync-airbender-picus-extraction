use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;

use crate::prover::mutations::MutatedInput;
use crate::prover::mutations::Mutator;
use crate::prover::seeds::SeedCase;
use crate::rv32im::prover::circuits::ProofInputs;

pub struct NoOpMutator;

impl Mutator for NoOpMutator {
    fn mutate(&self, seed_case: &SeedCase, _: &mut StdRng) -> MutatedInput {
        MutatedInput::new(seed_case, seed_case.base_input.clone(), self.name())
    }

    fn name(&self) -> &'static str {
        "no-op mutator"
    }

    fn mutate_non_mem_inputs(
        &self,
        _: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        _: &mut StdRng,
    ) {
        unimplemented!()
    }

    fn mutate_mem_inputs(
        &self,
        _: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        _: &mut StdRng,
    ) {
        unimplemented!()
    }
}
