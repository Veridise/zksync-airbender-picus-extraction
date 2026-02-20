use std::collections::HashMap;

use clap::{Parser, Subcommand};
use llzk_backend::builder::{Builder, OpsBuilder, StructBuilder};
use llzk_backend::VariableExtractor;
use prover::cs::definitions::Variable;
use prover::cs::machine::ops::unrolled::add_sub_lui_auipc_mop::{
    add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode, add_sub_lui_auipc_mop_table_addition_fn};
use prover::cs::one_row_compiler::OneRowCompiler;
use prover::{cs::cs::cs_reference::BasicAssembly, field::Mersenne31Field};
use prover::cs::cs::circuit::Circuit;
use add_sub_lui_auipc_mop;
use add_sub_lui_auipc_mop::{TRACE_LEN_LOG2, ROM_ADDRESS_SPACE_SECOND_WORD_BITS};
use anyhow::Result;
use llzk::prelude::StructDefOpLike;
use llzk_backend::AddConstraints;
use llzk::prelude::*;

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate LLZK IR from the circuits
    DumpLLZK {
        #[arg(short, long)]
        output: String,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .init();
    let cli = Cli::parse();
    match &cli.command {
        Commands::DumpLLZK { output } => {
            let mut cs = BasicAssembly::<Mersenne31Field>::new();

            add_sub_lui_auipc_mop_table_addition_fn(&mut cs);
            add_sub_lui_auipc_mop_circuit_with_preprocessed_bytecode(&mut cs);

            let (circuit_output, _maybe_wit_placer) = cs.finalize();

            // from [add_sub_lui_auipc_mop::get_circuit_for_rom_bound]
            let max_bytecode_size_in_words = (1 << (16 + ROM_ADDRESS_SPACE_SECOND_WORD_BITS)) / 4;
            // taken from add_sub_lui_auipc_mop::get_circuit:
            let compiler = OneRowCompiler::<Mersenne31Field>::default();
            // We don't want the compiled artifact as much as we want the constraints
            // that the compilation process adds.
            let _compiled = compiler.compile_executor_circuit_assuming_preprocessed_bytecode(
                circuit_output.clone(),
                max_bytecode_size_in_words,
                TRACE_LEN_LOG2 as usize,
            );

            // Generate an empty LLZK module
            let ctx = LlzkContext::new();
            let llzk_builder = Builder::new(&ctx);
            let module = llzk_module(llzk_builder.unknown_location());

            let mut struct_builder = StructBuilder::new(&ctx, "add_sub_lui_auipc_mop");

            // Also transform to PCL
            // TODO: Need to get the PCL pass exposed
            // llzk::passes::register_all_llzk_passes();
            // let pm = PassManager::new(&ctx);
            // pm.add_pass(llzk::passes::create_pcl_to_llzk_pass());
            // pm.run(&mut module).expect("failed to convert to PCL");

            println!("{:#?}", circuit_output);
            // println!("{:#?}", compiled);

            // Sanity check: all variables should be an input, output, or intermediate,
            // with the exception of the 2 variables used to encode the RAM write's
            // prior value (i.e., the read_value of one RAM query).
            let num_input_vars = circuit_output.get_inputs()?.iter().map(|x| x.num_vars()).sum::<usize>();
            let num_output_vars = circuit_output.get_outputs()?.iter().map(|x| x.num_vars()).sum::<usize>();
            let num_intermediate_vars = circuit_output.get_intermediates()?.iter().map(|x| x.num_vars()).sum::<usize>();
            let extracted = num_input_vars + num_output_vars + num_intermediate_vars;
            let expected = circuit_output.num_of_variables - 2;
            assert_eq!(extracted, expected);

            // Add inputs to struct
            // maps Variable to (input argument, optional index if array)
            let mut arg_map: HashMap<Variable, (usize, Option<i64>)> = HashMap::new();
            for (input_num, input) in circuit_output.get_inputs()?.iter().enumerate() {
                let arg_no = input_num + 1; // because of the self arg
                match input {
                    llzk_backend::ExtractedVariable::Register { low, high } => {
                        arg_map.insert(*low, (arg_no, Some(0i64)));
                        arg_map.insert(*high, (arg_no, Some(1i64)));
                        struct_builder.with_input(llzk_builder.register_type());
                    },
                    llzk_backend::ExtractedVariable::Scalar(variable) => {
                        arg_map.insert(*variable, (arg_no, None));
                        struct_builder.with_input(llzk_builder.felt_type());
                    },
                };
            }
            // Add outputs to struct
            // Maps CircuitOutput variable to (field name, index)
            let mut field_map: HashMap<Variable, (String, Option<i64>)> = HashMap::new();
            for (output_num, output) in circuit_output.get_outputs()?.iter().enumerate() {
                // TODO: better naming scheme
                match output {
                    llzk_backend::ExtractedVariable::Register { low, high } => {
                        let name = format!("out_reg_{output_num}");
                        field_map.insert(*low, (name.clone(), Some(0i64)));
                        field_map.insert(*high, (name.clone(), Some(1i64)));
                        struct_builder.with_member(name, llzk_builder.register_type(), true);
                    },
                    llzk_backend::ExtractedVariable::Scalar(variable) => {
                        let name = format!("out_var_{output_num}");
                        field_map.insert(*variable, (name.clone(), None));
                        struct_builder.with_member(name, llzk_builder.felt_type(), true);
                    }
                }
            }
            // Add intermediates to struct
            for (output_num, output) in circuit_output.get_intermediates()?.iter().enumerate() {
                // TODO: better naming scheme
                match output {
                    llzk_backend::ExtractedVariable::Register { low, high } => {
                        let name = format!("internal_reg_{output_num}");
                        field_map.insert(*low, (name.clone(), Some(0i64)));
                        field_map.insert(*high, (name.clone(), Some(1i64)));
                        struct_builder.with_member(name, llzk_builder.register_type(), false);
                    },
                    llzk_backend::ExtractedVariable::Scalar(variable) => {
                        let name = format!("internal_var_{output_num}");
                        field_map.insert(*variable, (name.clone(), None));
                        struct_builder.with_member(name, llzk_builder.felt_type(), false);
                    }
                }
            }

            let struct_op = struct_builder.build_in_module(&module)?;

            // TODO: add constraints
            /*
            (*struct_op).add_constraints(|builder| -> Result<()> {
                let get_input_val = |builder, var| -> Option<Value<>> {
                    match arg_map.get(var) {
                        None => None,
                        Some((arg_no, index)) => {
                            todo!("read")
                        }
                    }
                };
                // Add boolean constraints
                for bool_var in circuit_output.boolean_vars {

                }
                todo!("boolean constraints");
                // Add range constraints
                todo!();
                // Add all other constraints
                Ok(())
            })?;
            */

            println!("{struct_op}");
        }
    }
}
