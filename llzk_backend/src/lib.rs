use anyhow::anyhow;
use anyhow::Result;
use llzk::prelude::*;
use prover::cs::definitions::OpcodeFamilyCircuitState;
use prover::{cs::{cs::circuit::CircuitOutput, definitions::Variable}, field::{Field, PrimeField}};

use crate::builder::OpsBuilder;

pub mod builder;
pub mod codegen;

