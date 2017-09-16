extern crate num;

pub struct Feature<T: num::Unsigned> {
    id: u32,
    // All the values that this feature may be
    values: Vec<i32>,

    // lines[0] means the index in values
    lines: Vec<T>,
}
