use std::fs;

use fuzzing::rv32im::run_on_airbender;

#[test]
fn probe_unmodified_compliance_binary_airbender_only() {
    let binary =
        fs::read("tests/compliance-tests-programs/M-divu-00.bin").expect("read compliance binary");
    let text = fs::read("tests/compliance-tests-programs/M-divu-00.text").expect("read text");

    let airbender = run_on_airbender::<false>(&binary, Some(&text));
    println!("airbender={airbender:?}");
}
