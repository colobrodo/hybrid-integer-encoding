use std::convert::Infallible;

use webgraph::prelude::*;

#[derive(Default)]
pub struct FixedEstimator;

impl Encode for FixedEstimator {
    type Error = Infallible;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn write_outdegree(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_reference_offset(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_block_count(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_block(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_interval_count(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_interval_start(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_interval_len(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_first_residual(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn write_residual(&mut self, _value: u64) -> Result<usize, Self::Error> {
        Ok(1)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn end_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }
}
