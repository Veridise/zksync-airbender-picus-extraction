use std::alloc::Global;

use prover::check_satisfied;
use prover::common_constants;
use prover::common_constants::BLAKE2S_DELEGATION_CSR_REGISTER;
use prover::common_constants::KECCAK_SPECIAL5_CSR_REGISTER;
use prover::common_constants::SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX;
use prover::cs::cs::circuit::Circuit as _;
use prover::cs::machine::machine_configurations::create_csr_table_for_delegation;
use prover::cs::machine::ops::unrolled::compile_unrolled_circuit_state_transition;
use prover::cs::machine::ops::unrolled::materialize_flattened_decoder_table;
use prover::cs::tables::LookupWrapper;
use prover::cs::tables::TableDriver;
use prover::cs::tables::TableType;
use prover::fft::LdePrecomputations;
use prover::fft::Twiddles;
use prover::field::Field as _;
use prover::field::Mersenne31Field;
use prover::field::Mersenne31Quartic;
use prover::prover_stages::SetupPrecomputations;
use prover::risc_v_simulator::machine_mode_only_unrolled::NonMemoryOpcodeTracingDataWithTimestamp;
use prover::tests::unrolled::ensure_memory_trace_consistency;
use prover::tests::unrolled::parse_shuffle_ram_accesses_from_full_trace;
use prover::tests::unrolled::parse_state_permutation_elements_from_full_trace;
use prover::tests::unrolled::shift_binop_csrrw;
use prover::unrolled::evaluate_memory_witness_for_executor_family;
use prover::unrolled::evaluate_witness_for_executor_family;
use prover::unrolled::NonMemoryCircuitOracle;
use riscv_transpiler::replayer::ReplayerRam;
use riscv_transpiler::replayer::ReplayerVM;
use riscv_transpiler::vm::Counters as _;
use riscv_transpiler::vm::DelegationsAndFamiliesCounters;
use riscv_transpiler::vm::ReplayBuffer as _;
use riscv_transpiler::vm::SimpleSnapshotter;
use riscv_transpiler::vm::SimpleTape;
use riscv_transpiler::vm::State;
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
    pub fn prove_xor_and_or_shift_csr(
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
    ) {
        let csr_table = create_csr_table_for_delegation::<Mersenne31Field>(
            true,
            &[
                BLAKE2S_DELEGATION_CSR_REGISTER,
                KECCAK_SPECIAL5_CSR_REGISTER,
            ],
            TableType::SpecialCSRProperties.to_table_id(),
        );

        println!("Will try to prove XOR/AND/OR/SHIFT/CSR circuit");
        use prover::cs::machine::ops::unrolled::shift_binary_csr::*;

        let shift_binop_csrrw_circuit = {
            compile_unrolled_circuit_state_transition::<Mersenne31Field>(
                &|cs| {
                    shift_binop_csrrw_table_addition_fn(cs);
                    // and we need to add CSR table
                    cs.add_table_with_content(
                        TableType::SpecialCSRProperties,
                        LookupWrapper::Dimensional3(csr_table.clone()),
                    );
                },
                &|cs| shift_binop_csrrw_circuit_with_preprocessed_bytecode::<_, _>(cs),
                1 << 20,
                TRACE_LEN_LOG2,
            )
        };

        let mut table_driver = TableDriver::<Mersenne31Field>::new();
        shift_binop_csrrw_table_driver_fn(&mut table_driver);
        table_driver.add_table_with_content(
            TableType::SpecialCSRProperties,
            LookupWrapper::Dimensional3(csr_table),
        );

        let num_calls =
            counters.get_calls_to_circuit_family::<SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX>();
        dbg!(num_calls);

        let mut state = snapshotter.initial_snapshot.state;
        let mut ram_log_buffers = snapshotter
            .reads_buffer
            .make_range(0..snapshotter.reads_buffer.len());

        let mut ram = ReplayerRam::<{ common_constants::ROM_SECOND_WORD_BITS }> {
            ram_log: &mut ram_log_buffers,
        };
        let mut buffer = vec![NonMemoryOpcodeTracingDataWithTimestamp::default(); num_calls];
        let mut buffers = vec![&mut buffer[..]];
        let mut tracer = NonMemDestinationHolder::<SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX> {
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
            &preprocessing_data[&SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX];
        let decoder_table_data = materialize_flattened_decoder_table(decoder_table_data);

        let oracle = NonMemoryCircuitOracle {
            inner: &buffer[..],
            decoder_table: witness_gen_data,
            default_pc_value_in_padding: 4,
        };

        let is_empty = oracle.inner.is_empty();

        let memory_trace = evaluate_memory_witness_for_executor_family::<_, Global>(
            &shift_binop_csrrw_circuit,
            NUM_CYCLES_PER_CHUNK,
            &oracle,
            self.worker(),
            Global,
        );

        let full_trace = evaluate_witness_for_executor_family::<_, Global>(
            &shift_binop_csrrw_circuit,
            shift_binop_csrrw::witness_eval_fn,
            NUM_CYCLES_PER_CHUNK,
            &oracle,
            &table_driver,
            self.worker(),
            Global,
        );

        ensure_memory_trace_consistency(&memory_trace, &full_trace);

        parse_state_permutation_elements_from_full_trace(
            &shift_binop_csrrw_circuit,
            &full_trace,
            write_sets.write_set_mut(),
            read_sets.read_set_mut(),
        );
        parse_shuffle_ram_accesses_from_full_trace(
            &shift_binop_csrrw_circuit,
            &full_trace,
            write_sets.memory_write_set_mut(),
            read_sets.memory_read_set_mut(),
        );

        let is_satisfied = check_satisfied(
            &shift_binop_csrrw_circuit,
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
            &shift_binop_csrrw_circuit.setup_layout,
            &twiddles,
            &lde_precomputations,
            LDE_FACTOR,
            TREE_CAP_SIZE,
            self.worker(),
        );

        let (_, proof) = self.run_prover(
            &shift_binop_csrrw_circuit,
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
            assert_eq!(
                proof.delegation_argument_accumulator.unwrap(),
                Mersenne31Quartic::ZERO
            );
        }

        dbg!(proof.delegation_argument_accumulator.unwrap());

        accumulators
            .delegation_argument_mut()
            .add_assign(&proof.delegation_argument_accumulator.unwrap());
        accumulators
            .permutation_argument_mut()
            .mul_assign(&proof.permutation_grand_product_accumulator);
    }
}
