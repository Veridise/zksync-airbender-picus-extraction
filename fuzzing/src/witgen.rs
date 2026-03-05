use prover::cs::cs::circuit::Circuit as _;
use prover::cs::cs::cs_reference::BasicAssembly;
use prover::cs::cs::oracle::ExecutorFamilyDecoderData;
use prover::cs::cs::witness_placer::cs_debug_evaluator::CSDebugWitnessEvaluator;
use prover::cs::definitions::Variable;
use prover::field::Mersenne31Field;
use prover::field::PrimeField;
use rand::rngs::SmallRng;
use rand::Rng;

use crate::witgen::oracles::rand::RngOracle;
use crate::witgen::oracles::rand::RngOracleConfig;
use crate::witgen::targets::Circuits;

mod oracles;
pub mod targets;

/// Entrypoint function for the witgen fuzzer.
pub fn run(target: Circuits) {
    match target {
        Circuits::AddSubLuiAuipcMop => targets::add_sub_lui_auipc_mop::run(),
    }
}

trait FuzzTarget<F: PrimeField> {
    fn synthesize(&self, cs: &mut BasicAssembly<F>);

    fn random_decoder_data(&self, rng: &mut SmallRng) -> ExecutorFamilyDecoderData;
}

/// Main loop called by the corresponding target entrypoint function.
fn run_fuzzer(target: &dyn FuzzTarget<Mersenne31Field>) {
    let mut cs = BasicAssembly::<Mersenne31Field>::new();
    cs.witness_placer = Some(populate_inputs(target));
    target.synthesize(&mut cs);
    let (circuit_output, wit_placer) = cs.finalize();

    let Some(wit_placer) = wit_placer else {
        unreachable!();
    };

    let vars = (0..circuit_output.num_of_variables as u64)
        .map(Variable)
        .collect::<Vec<_>>();
    for var in &vars {
        println!(" v{}: {:?}", var.0, wit_placer.get_value(*var));
    }
}

const MAX_TABLE_LEN: u32 = 100;

/// Populates the inputs used by the witness placer.
fn populate_inputs(
    target: &dyn FuzzTarget<Mersenne31Field>,
) -> CSDebugWitnessEvaluator<Mersenne31Field> {
    let mut rng: SmallRng = rand::make_rng();
    let table_len = rng.next_u32() % MAX_TABLE_LEN;

    let preprocessed_decoder_table: Vec<_> = (0..table_len)
        .map(|_| target.random_decoder_data(&mut rng))
        .collect();
    CSDebugWitnessEvaluator::new_with_oracle_and_preprocessed_decoder(
        RngOracle::new(
            rng,
            RngOracleConfig {
                pc_mod: preprocessed_decoder_table.len() as u32,
            },
        ),
        preprocessed_decoder_table,
    )
}
