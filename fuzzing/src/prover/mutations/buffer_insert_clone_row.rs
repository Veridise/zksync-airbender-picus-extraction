use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::prover::mutations::Mutator;
use crate::rv32im::prover::circuits::ProofInputs;

pub struct BufferInsertCloneRowMutator;

impl Mutator for BufferInsertCloneRowMutator {
    fn name(&self) -> &'static str {
        "buffer insert clone row mutator"
    }

    fn mutate_non_mem_inputs(
        &self,
        input: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if input.buffer.is_empty() {
            return;
        }

        let src = rng.random_range(0..input.buffer.len());
        let dst = rng.random_range(0..=input.buffer.len());
        let row = input.buffer[src];
        input.buffer.insert(dst, row);
    }

    fn mutate_mem_inputs(
        &self,
        input: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if input.buffer.is_empty() {
            return;
        }

        let src = rng.random_range(0..input.buffer.len());
        let dst = rng.random_range(0..=input.buffer.len());
        let row = input.buffer[src];
        input.buffer.insert(dst, row);
    }
}
