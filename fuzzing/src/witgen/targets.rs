use clap::ValueEnum;

pub mod add_sub_lui_auipc_mop;

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum Circuits {
    AddSubLuiAuipcMop,
}
