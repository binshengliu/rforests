extern crate rforests;

#[test]
fn test_generate() {
    rforests::generate_statistics("tests/train.txt");
    println!("test-----------------------", );
    println!("{}", "a#b".split('#').next().unwrap());
    assert!(false);
}
