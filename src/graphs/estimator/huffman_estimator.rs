use std::convert::Infallible;

use anyhow::Result;

use webgraph::prelude::Encode;

use crate::graphs::{BvGraphComponent, ContextModel};
use crate::huffman::{CostModel, EncodeParams, IntegerHistogram};

pub struct HuffmanEstimator<EP: EncodeParams, S: ContextModel> {
    cost_model: CostModel<EP>,
    context_model: S,
    marker_: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams, S: ContextModel> HuffmanEstimator<EP, S> {
    pub fn new(data: IntegerHistogram<EP>, context_model: S) -> Self {
        Self {
            cost_model: data.cost(),
            context_model,
            marker_: core::marker::PhantomData,
        }
    }

    pub fn estimate(&mut self, component: BvGraphComponent, value: u64) -> usize {
        let ctx = self.context_model.choose_context(component);
        self.context_model.update(component, value);
        self.cost_model.cost(ctx, value)
    }
}

impl<EP: EncodeParams, S: ContextModel> Encode for HuffmanEstimator<EP, S> {
    type Error = Infallible;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        self.context_model.reset();
        Ok(0)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::Outdegree, value))
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::ReferenceOffset, value))
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::BlockCount, value))
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::Blocks, value))
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::IntervalCount, value))
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::IntervalStart, value))
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::IntervalLen, value))
    }

    fn num_of_residuals(&mut self, total_residuals: usize) {
        self.context_model.num_of_residuals(total_residuals);
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::FirstResidual, value))
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        Ok(self.estimate(BvGraphComponent::Residual, value))
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn end_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }
}
