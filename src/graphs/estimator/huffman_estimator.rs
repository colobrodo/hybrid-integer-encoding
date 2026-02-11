use std::borrow::Borrow;
use std::convert::Infallible;

use anyhow::Result;

use webgraph::prelude::Encode;

use crate::graphs::{BvGraphComponent, ContextModel};
use crate::huffman::{CostModel, EncodeParams};

pub struct HuffmanEstimator<EP: EncodeParams, M: Borrow<CostModel<EP>>, S: ContextModel> {
    cost_model: M,
    context_model: S,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams, S: ContextModel, M: Borrow<CostModel<EP>>> HuffmanEstimator<EP, M, S> {
    pub fn new(cost_model: M, context_model: S) -> Self {
        Self {
            cost_model,
            context_model,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn estimate(&mut self, component: BvGraphComponent, value: u64) -> usize {
        let ctx = self.context_model.choose_context(component);
        self.context_model.update(component, value);
        self.cost_model.borrow().cost(ctx, value)
    }
}

impl<EP: EncodeParams, S: ContextModel> Copy for HuffmanEstimator<EP, &CostModel<EP>, S> where
    S: Copy
{
}

impl<EP: EncodeParams, S: ContextModel> Clone for HuffmanEstimator<EP, &CostModel<EP>, S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            cost_model: self.cost_model,
            context_model: self.context_model.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<EP: EncodeParams, M: Borrow<CostModel<EP>>, S: ContextModel> Encode
    for HuffmanEstimator<EP, M, S>
{
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
