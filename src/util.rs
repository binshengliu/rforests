pub type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

/// Type for feature id.
pub type Id = usize;

/// Type for labels, feature values.
pub type Value = f64;

use scoped_threadpool::Pool;
use std::sync::Mutex;
lazy_static! {
    pub static ref POOL: Mutex<Pool> = Mutex::new(Pool::new(4));
}
