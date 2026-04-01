use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;

use crate::prover::mutations::choose_distinct_indices;
use crate::prover::mutations::Mutator;
use crate::rv32im::prover::circuits::ProofInputs;

pub struct BufferDuplicateRowMutator;

impl Mutator for BufferDuplicateRowMutator {
    fn name(&self) -> &'static str {
        "buffer duplicate row mutator"
    }

    fn mutate_non_mem_inputs(
        &self,
        input: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if let Some((src, dst)) = choose_distinct_indices(input.buffer.len(), rng) {
            input.buffer[dst] = input.buffer[src];
        }
    }

    fn mutate_mem_inputs(
        &self,
        input: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if let Some((src, dst)) = choose_distinct_indices(input.buffer.len(), rng) {
            input.buffer[dst] = input.buffer[src];
        }
    }
}
