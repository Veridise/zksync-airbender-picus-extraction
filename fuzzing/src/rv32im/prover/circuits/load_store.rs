use std::alloc::Global;

use prover::check_satisfied;
use prover::common_constants;
use prover::common_constants::ADD_SUB_LUI_AUIPC_MOP_CIRCUIT_FAMILY_IDX;
use prover::common_constants::LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::machine::ops::unrolled::compile_unrolled_circuit_state_transition;
use prover::cs::machine::ops::unrolled::load_store_word_only::create_word_only_load_store_special_tables;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_circuit_with_preprocessed_bytecode;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_table_addition_fn;
use prover::cs::machine::ops::unrolled::load_store_word_only::word_only_load_store_table_driver_fn;
use prover::cs::machine::ops::unrolled::materialize_flattened_decoder_table;
use prover::cs::tables::TableDriver;
use prover::fft::LdePrecomputations;
use prover::fft::Twiddles;
use prover::field::Field as _;
use prover::field::Mersenne31Field;
use prover::field::Mersenne31Quartic;
use prover::prover_stages::SetupPrecomputations;
use prover::risc_v_simulator::machine_mode_only_unrolled::MemoryOpcodeTracingDataWithTimestamp;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use prover::tests::unrolled::add_sub_lui_auipc_mod;
use prover::tests::unrolled::ensure_memory_trace_consistency;
use prover::tests::unrolled::parse_shuffle_ram_accesses_from_full_trace;
use prover::tests::unrolled::parse_state_permutation_elements_from_full_trace;
use prover::tests::unrolled::word_load_store;
use prover::unrolled::evaluate_memory_witness_for_executor_family;
use prover::unrolled::evaluate_witness_for_executor_family;
use prover::unrolled::MemoryCircuitOracle;
use prover::unrolled::NonMemoryCircuitOracle;
use riscv_transpiler::replayer::ReplayerRam;
use riscv_transpiler::replayer::ReplayerVM;
use riscv_transpiler::vm::Counters as _;
use riscv_transpiler::vm::DelegationsAndFamiliesCounters;
use riscv_transpiler::vm::ReplayBuffer as _;
use riscv_transpiler::vm::SimpleSnapshotter;
use riscv_transpiler::vm::SimpleTape;
use riscv_transpiler::vm::State;
use riscv_transpiler::witness::MemDestinationHolder;
use riscv_transpiler::witness::NonMemDestinationHolder;

use crate::rv32im::prover::accumulators::Accumulators;
use crate::rv32im::prover::factories::PreprocessingData;
use crate::rv32im::prover::sets::ReadSets;
use crate::rv32im::prover::sets::WriteSets;
use crate::rv32im::prover::Prover;
use crate::rv32im::prover::LDE_FACTOR;
use crate::rv32im::prover::NUM_CYCLES_PER_CHUNK;
use crate::rv32im::prover::TRACE_LEN;
use crate::rv32im::prover::TRACE_LEN_LOG2;
use crate::rv32im::prover::TREE_CAP_SIZE;
use crate::rv32im::vm::CountersT;

impl Prover {
    pub fn prove_load_store(
        &self,
        accumulators: &mut Accumulators,
        snapshotter: &SimpleSnapshotter<CountersT, { common_constants::ROM_SECOND_WORD_BITS }>,
        counters: &DelegationsAndFamiliesCounters,
        tape: &SimpleTape,
        cycles_bound: usize,
        expected_final_state: State<CountersT>,
        read_sets: &mut ReadSets,
        write_sets: &mut WriteSets,
        preprocessing_data: &PreprocessingData,
        bytecode: &[u32],
    ) {
        println!("Will try to prove word LOAD/STORE circuit");

        let extra_tables = create_word_only_load_store_special_tables::<
            _,
            { common_constants::ROM_SECOND_WORD_BITS },
        >(bytecode);
        let word_load_store_circuit = {
            compile_unrolled_circuit_state_transition::<Mersenne31Field>(
                &|cs| {
                    word_only_load_store_table_addition_fn(cs);
                    for (table_type, table) in extra_tables.clone() {
                        cs.add_table_with_content(table_type, table);
                    }
                },
                &|cs| {
                    word_only_load_store_circuit_with_preprocessed_bytecode::<
                        _,
                        _,
                        { common_constants::ROM_SECOND_WORD_BITS },
                    >(cs)
                },
                1 << 20,
                TRACE_LEN_LOG2,
            )
        };

        let mut table_driver = TableDriver::<Mersenne31Field>::new();
        word_only_load_store_table_driver_fn(&mut table_driver);
        for (table_type, table) in extra_tables.clone() {
            table_driver.add_table_with_content(table_type, table);
        }

        let num_calls =
            counters.get_calls_to_circuit_family::<LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX>();
        dbg!(num_calls);

        let mut state = snapshotter.initial_snapshot.state;
        let mut ram_log_buffers = snapshotter
            .reads_buffer
            .make_range(0..snapshotter.reads_buffer.len());

        let mut ram = ReplayerRam::<{ common_constants::ROM_SECOND_WORD_BITS }> {
            ram_log: &mut ram_log_buffers,
        };

        let mut buffer = vec![MemoryOpcodeTracingDataWithTimestamp::default(); num_calls];
        let mut buffers = vec![&mut buffer[..]];
        let mut tracer = MemDestinationHolder::<LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX> {
            buffers: &mut buffers[..],
        };

        ReplayerVM::<CountersT>::replay_basic_unrolled::<_, _>(
            &mut state,
            &mut ram,
            tape,
            &mut (),
            cycles_bound,
            &mut tracer,
        );
        assert_eq!(expected_final_state, state);

        let (decoder_table_data, witness_gen_data) =
            &preprocessing_data[&LOAD_STORE_WORD_ONLY_CIRCUIT_FAMILY_IDX];
        let decoder_table_data = materialize_flattened_decoder_table(decoder_table_data);

        let oracle = MemoryCircuitOracle {
            inner: &buffer[..],
            decoder_table: witness_gen_data,
        };

        let is_empty = oracle.inner.is_empty();

        let memory_trace = evaluate_memory_witness_for_executor_family::<_, Global>(
            &word_load_store_circuit,
            NUM_CYCLES_PER_CHUNK,
            &oracle,
            self.worker(),
            Global,
        );

        let full_trace = evaluate_witness_for_executor_family::<_, Global>(
            &word_load_store_circuit,
            word_load_store::witness_eval_fn,
            NUM_CYCLES_PER_CHUNK,
            &oracle,
            &table_driver,
            self.worker(),
            Global,
        );

        ensure_memory_trace_consistency(&memory_trace, &full_trace);

        parse_state_permutation_elements_from_full_trace(
            &word_load_store_circuit,
            &full_trace,
            write_sets.write_set_mut(),
            read_sets.read_set_mut(),
        );
        parse_shuffle_ram_accesses_from_full_trace(
            &word_load_store_circuit,
            &full_trace,
            write_sets.memory_write_set_mut(),
            read_sets.memory_read_set_mut(),
        );

        let is_satisfied = check_satisfied(
            &word_load_store_circuit,
            &full_trace.exec_trace,
            full_trace.num_witness_columns,
        );
        assert!(is_satisfied);

        let twiddles: Twiddles<_, Global> = Twiddles::new(TRACE_LEN, self.worker());
        let lde_precomputations =
            LdePrecomputations::new(TRACE_LEN, LDE_FACTOR, &[0, 1], self.worker());
        let setup = SetupPrecomputations::from_tables_and_trace_len_with_decoder_table(
            &table_driver,
            &decoder_table_data,
            TRACE_LEN,
            &word_load_store_circuit.setup_layout,
            &twiddles,
            &lde_precomputations,
            LDE_FACTOR,
            TREE_CAP_SIZE,
            self.worker(),
        );

        let (_, proof) = self.run_prover(
            &word_load_store_circuit,
            full_trace,
            &setup,
            &twiddles,
            &lde_precomputations,
        );

        if is_empty {
            assert_eq!(
                proof.permutation_grand_product_accumulator,
                Mersenne31Quartic::ONE
            );
        }
        assert!(proof.delegation_argument_accumulator.is_none());

        accumulators
            .permutation_argument_mut()
            .mul_assign(&proof.permutation_grand_product_accumulator);
    }
}
