use std::alloc::Global;
use std::collections::BTreeSet;
use std::collections::HashMap;

use prover::check_satisfied;
use prover::common_constants::TimestampScalar;
use prover::common_constants::ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX;
use prover::common_constants::BIGINT_OPS_WITH_CONTROL_CSR_REGISTER;
use prover::common_constants::BLAKE2S_DELEGATION_CSR_REGISTER;
use prover::common_constants::INITIAL_TIMESTAMP;
use prover::common_constants::JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX;
use prover::common_constants::KECCAK_SPECIAL5_CSR_REGISTER;
use prover::common_constants::LOAD_STORE_SUBWORD_ONLY_CIRCUIT_FAMILY_IDX;
use prover::common_constants::LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX;
use prover::common_constants::MUL_DIV_CIRCUIT_FAMILY_IDX;
use prover::common_constants::SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX;
use prover::common_constants::TIMESTAMP_STEP;
use prover::common_constants::{self};
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::oracle::ExecutorFamilyDecoderData;
use prover::cs::definitions::NUM_DELEGATION_ARGUMENT_KEY_PARTS;
use prover::cs::definitions::NUM_MACHINE_STATE_LINEARIZATION_CHALLENGES;
use prover::cs::definitions::NUM_MEM_ARGUMENT_KEY_PARTS;
use prover::cs::delegation::blake2_round_with_extended_control::define_blake2_with_extended_control_delegation_circuit;
use prover::cs::machine::machine_configurations::create_csr_table_for_delegation;
use prover::cs::machine::ops::unrolled::compile_unrolled_circuit_state_transition;
use prover::cs::machine::ops::unrolled::load_store::create_load_store_special_tables;
use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_circuit_with_preprocessed_bytecode;
use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_table_addition_fn;
use prover::cs::machine::ops::unrolled::load_store_subword_only::subword_only_load_store_table_driver_fn;
use prover::cs::machine::ops::unrolled::load_store_word_only::create_word_only_load_store_special_tables;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_circuit_with_preprocessed_bytecode;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_table_addition_fn;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_table_driver_fn;
use prover::cs::machine::ops::unrolled::materialize_flattened_decoder_table;
use prover::cs::machine::ops::unrolled::opcodes_for_full_machine_with_mem_word_access_specialization;
use prover::cs::machine::ops::unrolled::opcodes_for_full_machine_with_unsigned_mul_div_only_with_mem_word_access_specialization;
use prover::cs::machine::ops::unrolled::DecoderTableEntry;
use prover::cs::machine::NON_DETERMINISM_CSR;
use prover::cs::one_row_compiler::CompiledCircuitArtifact;
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::cs::tables::LookupWrapper;
use prover::cs::tables::TableDriver;
use prover::cs::tables::TableType;
use prover::cs::utils::split_timestamp;
use prover::definitions::produce_pc_into_permutation_accumulator_raw;
use prover::definitions::AuxArgumentsBoundaryValues;
use prover::definitions::ExternalChallenges;
use prover::definitions::ExternalDelegationArgumentChallenges;
use prover::definitions::ExternalMachineStateArgumentChallenges;
use prover::definitions::ExternalMemoryArgumentChallenges;
use prover::definitions::ExternalValues;
use prover::evaluate_delegation_memory_witness;
use prover::evaluate_witness;
use prover::fft::materialize_powers_serial_starting_with_elem;
use prover::fft::LdePrecomputations;
use prover::fft::Twiddles;
use prover::field::Field as _;
use prover::field::Mersenne31Complex;
use prover::field::Mersenne31Field;
use prover::field::Mersenne31Quartic;
use prover::mem_utils::produce_register_contribution_into_memory_accumulator;
use prover::merkle_trees::DefaultTreeConstructor;
use prover::prover_stages;
use prover::prover_stages::prove;
use prover::prover_stages::unrolled_prover::prove_configured_for_unrolled_circuits;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use prover::prover_stages::ProverData;
use prover::prover_stages::SetupPrecomputations;
use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use prover::tests::blake2s_delegation_with_transpiler;
use prover::tests::keccak_special5_delegation_with_transpiler;
use prover::tests::unrolled::add_sub_lui_auipc_mod;
use prover::tests::unrolled::ensure_memory_trace_consistency;
use prover::tests::unrolled::jump_branch_slt;
use prover::tests::unrolled::mul_div;
use prover::tests::unrolled::mul_div_unsigned_only;
use prover::tests::unrolled::parse_delegation_ram_accesses_from_full_trace;
use prover::tests::unrolled::parse_shuffle_ram_accesses_from_full_trace;
use prover::tests::unrolled::parse_state_permutation_elements_from_full_trace;
use prover::tests::unrolled::shift_binop_csrrw;
use prover::tests::unrolled::subword_load_store;
use prover::tests::unrolled::word_load_store;
use prover::tests::GpuComparisonArgs;
use prover::tracers::oracles::transpiler_oracles::delegation::Blake2sDelegationOracle;
use prover::tracers::oracles::transpiler_oracles::delegation::KeccakDelegationOracle;
use prover::unrolled::evaluate_init_and_teardown_memory_witness;
use prover::unrolled::evaluate_init_and_teardown_witness;
use prover::unrolled::evaluate_memory_witness_for_executor_family;
use prover::unrolled::evaluate_witness_for_executor_family;
use prover::unrolled::MemoryCircuitOracle;
use prover::unrolled::NonMemoryCircuitOracle;
use prover::worker::Worker;
use prover::ExecutorFamilyWitnessEvaluationAuxData;
use prover::RamShuffleMemStateRecord;
use prover::WitnessEvaluationData;
use prover::WitnessEvaluationDataForExecutionFamily;
use prover::DEFAULT_TRACE_PADDING_MULTIPLE;
use riscv_transpiler::replayer::ReplayerRam;
use riscv_transpiler::replayer::ReplayerVM;
use riscv_transpiler::vm::Counters;
use riscv_transpiler::vm::DelegationsAndFamiliesCounters;
use riscv_transpiler::vm::RamWithRomRegion;
use riscv_transpiler::vm::ReplayBuffer as _;
use riscv_transpiler::vm::SimpleSnapshotter;
use riscv_transpiler::vm::SimpleTape;
use riscv_transpiler::vm::State;
use riscv_transpiler::witness::BlakeDelegationDestinationHolder;
use riscv_transpiler::witness::DelegationWitness;
use riscv_transpiler::witness::KeccakDelegationDestinationHolder;
use riscv_transpiler::witness::MemDestinationHolder;
use riscv_transpiler::witness::NonMemDestinationHolder;

use crate::rv32im::prover::checks::validate_inits_and_teardowns;
use crate::rv32im::prover::checks::validate_sets;
use crate::rv32im::vm::CountersT;

mod accumulators;
mod checks;
mod circuits;
mod factories;
mod sets;

use accumulators::Accumulators;
use checks::validate_counters;
use factories::make_external_challenges;
use factories::make_preprocessing_data;
use sets::ReadSets;
use sets::WriteSets;

const TRACE_LEN_LOG2: usize = 24;
const NUM_CYCLES_PER_CHUNK: usize = (1 << TRACE_LEN_LOG2) - 1;
const CHECK_MEMORY_PERMUTATION_ONLY: bool = false;

const SUPPORT_SIGNED: bool = false;
const INITIAL_PC: u32 = 0;
const NUM_INIT_AND_TEARDOWN_SETS: usize = 6;
const NUM_DELEGATION_CYCLES: usize = (1 << 20) - 1;

const LDE_FACTOR: usize = 2;
const TREE_CAP_SIZE: usize = 32;
const TRACE_LEN: usize = 1 << TRACE_LEN_LOG2;

struct Prover {
    worker: Worker,
    default_security_config: prover_stages::ProofSecurityConfig,
    external_challenges: ExternalChallenges,
}

impl Prover {
    fn new() -> Self {
        let default_security_config =
            prover_stages::ProofSecurityConfig::for_queries_only(5, 28, 63);

        let worker = Worker::new_with_num_threads(1);
        Self {
            default_security_config,
            worker,
            external_challenges: make_external_challenges(),
        }
    }

    fn external_challenges(&self) -> &ExternalChallenges {
        &self.external_challenges
    }

    fn worker(&self) -> &Worker {
        &self.worker
    }

    fn default_security_config(&self) -> &prover_stages::ProofSecurityConfig {
        &self.default_security_config
    }

    fn run_prover(
        &self,
        compiled_circuit: &CompiledCircuitArtifact<Mersenne31Field>,
        full_trace: WitnessEvaluationDataForExecutionFamily<DEFAULT_TRACE_PADDING_MULTIPLE, Global>,
        setup: &SetupPrecomputations<
            DEFAULT_TRACE_PADDING_MULTIPLE,
            Global,
            DefaultTreeConstructor,
        >,
        twiddles: &Twiddles<Mersenne31Complex, Global>,
        lde_precomputations: &LdePrecomputations<Global>,
    ) -> (
        ProverData<DEFAULT_TRACE_PADDING_MULTIPLE, Global, DefaultTreeConstructor>,
        UnrolledModeProof,
    ) {
        self.run_prover_with_auxdata(
            compiled_circuit,
            full_trace,
            setup,
            twiddles,
            lde_precomputations,
            &[],
        )
    }

    fn run_prover_with_auxdata(
        &self,
        compiled_circuit: &CompiledCircuitArtifact<Mersenne31Field>,
        full_trace: WitnessEvaluationDataForExecutionFamily<DEFAULT_TRACE_PADDING_MULTIPLE, Global>,
        setup: &SetupPrecomputations<
            DEFAULT_TRACE_PADDING_MULTIPLE,
            Global,
            DefaultTreeConstructor,
        >,
        twiddles: &Twiddles<Mersenne31Complex, Global>,
        lde_precomputations: &LdePrecomputations<Global>,
        aux_boundary_data: &[AuxArgumentsBoundaryValues],
    ) -> (
        ProverData<DEFAULT_TRACE_PADDING_MULTIPLE, Global, DefaultTreeConstructor>,
        UnrolledModeProof,
    ) {
        println!("Trying to prove");

        let now = std::time::Instant::now();
        let proof = prove_configured_for_unrolled_circuits::<
            DEFAULT_TRACE_PADDING_MULTIPLE,
            _,
            DefaultTreeConstructor,
        >(
            compiled_circuit,
            &vec![],
            self.external_challenges(),
            full_trace,
            aux_boundary_data,
            &setup,
            &twiddles,
            &lde_precomputations,
            None,
            LDE_FACTOR,
            TREE_CAP_SIZE,
            self.default_security_config(),
            self.worker(),
        );
        println!("Proving time is {:?}", now.elapsed());
        proof
    }
}

pub fn prove_vm_result(
    snapshotter: &mut SimpleSnapshotter<CountersT, { common_constants::ROM_SECOND_WORD_BITS }>,
    state: &mut State<CountersT>,
    ram: &mut RamWithRomRegion<{ common_constants::ROM_SECOND_WORD_BITS }>,
    text_section: &[u32],
    tape: &SimpleTape,
    binary: &[u32],
    cycles_bound: usize,
) {
    let prover = Prover::new();
    let _total_snapshots = snapshotter.snapshots.len();

    let exact_cycles_passed = (state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP;

    println!("Passed exactly {} cycles", exact_cycles_passed);

    let counters = snapshotter.snapshots.last().unwrap().state.counters;

    let shuffle_ram_touched_addresses = ram.collect_inits_and_teardowns(prover.worker(), Global);

    use prover::tracers::oracles::chunk_lazy_init_and_teardown;
    let total_unique_teardowns: usize = shuffle_ram_touched_addresses
        .iter()
        .map(|el| el.len())
        .sum();

    println!("Touched {} unique addresses", total_unique_teardowns);

    let (num_trivial, inits_and_teardowns) = chunk_lazy_init_and_teardown::<Global, _>(
        1,
        NUM_CYCLES_PER_CHUNK * NUM_INIT_AND_TEARDOWN_SETS,
        &shuffle_ram_touched_addresses,
        prover.worker(),
    );
    assert_eq!(num_trivial, 0, "trivial padding is not expected in tests");

    let flattened_inits_and_teardowns: Vec<_> = shuffle_ram_touched_addresses
        .into_iter()
        .flatten()
        .collect();

    println!("Finished at PC = 0x{:08x}", state.pc);
    for (reg_idx, reg) in state.registers.iter().enumerate() {
        println!("x{} = {}", reg_idx, reg.value);
    }

    let mut expected_final_state = *state;
    expected_final_state.counters = Default::default();

    let external_challenges = make_external_challenges();
    // evaluate memory witness
    let preprocessing_data = make_preprocessing_data(text_section);
    let mut accumulators = Accumulators::new(*state, &external_challenges);
    let mut read_sets = ReadSets::new(*state);
    let mut write_sets = WriteSets::new();

    validate_counters(&counters);

    prover.prove_add_sub_lui_auipc_mop(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
    );

    prover.prove_jump_branch_slt(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
    );

    prover.prove_xor_and_or_shift_csr(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
    );

    prover.prove_mul_div(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
    );

    prover.prove_load_store(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
        binary,
    );

    prover.prove_subword_load_store(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
        binary,
    );
    // Machine state permutation ended
    validate_sets(&read_sets, &write_sets);

    prover.prove_init_and_teardowns(
        &mut accumulators,
        snapshotter,
        &counters,
        tape,
        cycles_bound,
        expected_final_state,
        &mut read_sets,
        &mut write_sets,
        &preprocessing_data,
        &inits_and_teardowns,
    );
    // now prove delegation circuits
    prover.prove_blake_delegation(
        &mut accumulators,
        &counters,
        snapshotter,
        &mut read_sets,
        &mut write_sets,
        tape,
        cycles_bound,
        expected_final_state,
    );

    prover.prove_keccak_delegation(
        &mut accumulators,
        &counters,
        snapshotter,
        &mut read_sets,
        &mut write_sets,
        tape,
        cycles_bound,
        expected_final_state,
    );

    dbg!(accumulators.permutation_argument());
    dbg!(accumulators.delegation_argument());

    // inits and teardowns
    validate_inits_and_teardowns(
        &read_sets,
        &write_sets,
        &flattened_inits_and_teardowns,
        &accumulators,
        total_unique_teardowns,
    );
}
