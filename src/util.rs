use std;
pub type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

/// Type for feature id.
pub type Id = usize;

/// Type for labels, feature values.
pub type Value = f64;
