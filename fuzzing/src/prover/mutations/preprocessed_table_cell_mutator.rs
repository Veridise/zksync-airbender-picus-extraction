use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use rand::rngs::StdRng;

use crate::prover::mutations::mutate_preprocessed_table_cell;
use crate::prover::mutations::Mutator;
use crate::rv32im::prover::circuits::ProofInputs;

pub struct PreprocessedTableCellMutator;

impl Mutator for PreprocessedTableCellMutator {
    fn name(&self) -> &'static str {
        "preprocessed table cell mutator"
    }

    fn mutate_non_mem_inputs(
        &self,
        input: &mut ProofInputs<NonMemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        mutate_preprocessed_table_cell(&mut input.decoder_table_data, rng);
    }

    fn mutate_mem_inputs(
        &self,
        input: &mut ProofInputs<MemoryOpcodeTracingDataWithTimestamp>,
        rng: &mut StdRng,
    ) {
        mutate_preprocessed_table_cell(&mut input.decoder_table_data, rng);
    }
}
