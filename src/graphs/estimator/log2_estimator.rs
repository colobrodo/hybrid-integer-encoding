use std::convert::Infallible;

use webgraph::prelude::*;

pub struct Log2Estimator;

impl Encode for Log2Estimator {
    type Error = Infallible;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(u64::ilog2(value + 2) as usize)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn end_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }
}
