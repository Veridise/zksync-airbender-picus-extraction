use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;

use crate::prover::mutations::choose_distinct_indices;
use crate::prover::mutations::Mutator;
use crate::rv32im::prover::circuits::ProofInputs;

pub struct PreprocessedTableRowSwapMutator;

impl Mutator for PreprocessedTableRowSwapMutator {
    fn name(&self) -> &'static str {
        "preprocessed table row swap mutator"
    }

    fn mutate_non_mem_inputs(
        &self,
        input: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if let Some((a, b)) = choose_distinct_indices(input.decoder_table_data.len(), rng) {
            input.decoder_table_data.swap(a, b);
        }
    }

    fn mutate_mem_inputs(
        &self,
        input: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        if let Some((a, b)) = choose_distinct_indices(input.decoder_table_data.len(), rng) {
            input.decoder_table_data.swap(a, b);
        }
    }
}
