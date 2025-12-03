use std::convert::Infallible;

use crate::huffman::{DefaultEncodeParams, EncodeParams, HuffmanEncoder, IntegerHistogram};

use anyhow::{Context, Result};
use dsi_bitstream::traits::{BitWrite, LE};
use webgraph::prelude::*;

use super::{BvGraphComponent, ContextModel, HuffmanEstimator};

/// Encoder to compress a graph using Hybrid integer huffman encoding.
/// This encoder can be constructed using an `HuffmanGraphEncoderBuilder`
pub struct HuffmanGraphEncoder<'a, EP: EncodeParams, E: Encode, W: BitWrite<LE>, S: ContextModel> {
    context_model: S,
    encoder: HuffmanEncoder<EP>,
    estimator: E,
    writer: &'a mut W,
}
impl<'a, EP: EncodeParams, E: Encode, W: BitWrite<LE>, S: ContextModel>
    HuffmanGraphEncoder<'a, EP, E, W, S>
{
    pub(crate) fn new(
        encoder: HuffmanEncoder<EP>,
        estimator: E,
        context_model: S,
        writer: &'a mut W,
    ) -> Self {
        Self {
            encoder,
            estimator,
            context_model,
            writer,
        }
    }

    pub fn write_header(&mut self) -> Result<usize> {
        self.encoder
            .write_header(self.writer)
            .with_context(|| "Failed to write Huffman encoder header")
    }

    fn write(&mut self, component: BvGraphComponent, value: u64) -> Result<usize> {
        let ctx = self.context_model.choose_context(component);
        let (token_bits, trailing_bits) = self
            .encoder
            .write(ctx, value, self.writer)
            .with_context(|| {
                format!(
                    "Failed to write {:?} component with value {}",
                    component, value
                )
            })?;
        self.context_model.update(component, value);
        Ok(token_bits + trailing_bits)
    }
}

impl<EP: EncodeParams, E: Encode, W: BitWrite<LE>, S: ContextModel> Encode
    for HuffmanGraphEncoder<'_, EP, E, W, S>
{
    type Error = Infallible;

    fn start_node(&mut self, _node_id: usize) -> Result<usize, Self::Error> {
        self.context_model.reset();
        Ok(0)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::Outdegree, value).unwrap();
        Ok(result)
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self
            .write(BvGraphComponent::ReferenceOffset, value)
            .unwrap();
        Ok(result)
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::BlockCount, value).unwrap();
        Ok(result)
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::Blocks, value).unwrap();
        Ok(result)
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::IntervalCount, value).unwrap();
        Ok(result)
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::IntervalStart, value).unwrap();
        Ok(result)
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::IntervalLen, value).unwrap();
        Ok(result)
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::FirstResidual, value).unwrap();
        Ok(result)
    }

    fn num_of_residuals(&mut self, total_residuals: usize) {
        self.context_model.num_of_residuals(total_residuals);
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::Residual, value).unwrap();
        Ok(result)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.writer.flush().expect("Failed to flush encoder writer");
        Ok(0)
    }

    fn end_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

impl<EP: EncodeParams, E: Encode, W: BitWrite<LE>, S: ContextModel> EncodeAndEstimate
    for HuffmanGraphEncoder<'_, EP, E, W, S>
{
    type Estimator<'b>
        = &'b mut E
    where
        Self: 'b;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        &mut self.estimator
    }
}

/// Encoder to construct a `HuffmanGraphEncoder` or a `HuffmanEstimator` from the frequencies
/// of the symbols encountered during encoding.
/// It requires an estimator that implements the `Encode` trait and is used for estimating the
/// adjacency list size of each node during the process of reference selection.
pub struct HuffmanGraphEncoderBuilder<
    E: Encode,
    C: ContextModel,
    EP: EncodeParams = DefaultEncodeParams,
> {
    estimator: E,
    context_model: C,
    data: IntegerHistogram<EP>,
}

impl<EP: EncodeParams, E: Encode, C: ContextModel> HuffmanGraphEncoderBuilder<E, C, EP> {
    pub fn new(num_symbols: usize, estimator: E, context_model: C) -> Self {
        let contexts = C::num_contexts();
        Self {
            estimator,
            context_model,
            data: IntegerHistogram::new(contexts, num_symbols),
        }
    }

    pub fn build<W: BitWrite<LE>>(
        self,
        writer: &'_ mut W,
        max_bits: usize,
    ) -> HuffmanGraphEncoder<'_, EP, E, W, C> {
        let encoder = HuffmanEncoder::<EP>::new(self.data, max_bits);
        HuffmanGraphEncoder::new(encoder, self.estimator, self.context_model, writer)
    }

    pub fn build_estimator(self) -> HuffmanEstimator<EP, C> {
        HuffmanEstimator::new(self.data.cost(), self.context_model)
    }

    fn add_data(&mut self, component: BvGraphComponent, value: u64) {
        let context = self.context_model.choose_context(component);
        self.data.add(context, value as u32);
        self.context_model.update(component, value);
    }
}

impl<EP: EncodeParams, E: Encode, S: ContextModel> Encode for HuffmanGraphEncoderBuilder<E, S, EP> {
    type Error = E::Error;

    fn start_node(&mut self, node: usize) -> Result<usize, Self::Error> {
        let result = self.estimator.start_node(node)?;
        self.context_model.reset();
        Ok(result)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_outdegree(value)?;
        self.add_data(BvGraphComponent::Outdegree, value);
        Ok(result)
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_reference_offset(value)?;
        self.add_data(BvGraphComponent::ReferenceOffset, value);
        Ok(result)
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_block_count(value)?;
        self.add_data(BvGraphComponent::BlockCount, value);
        Ok(result)
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_block(value)?;
        self.add_data(BvGraphComponent::Blocks, value);
        Ok(result)
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_count(value)?;
        self.add_data(BvGraphComponent::IntervalCount, value);
        Ok(result)
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_start(value)?;
        self.add_data(BvGraphComponent::IntervalStart, value);
        Ok(result)
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_len(value)?;
        self.add_data(BvGraphComponent::IntervalLen, value);
        Ok(result)
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_first_residual(value)?;
        self.add_data(BvGraphComponent::FirstResidual, value);
        Ok(result)
    }

    fn num_of_residuals(&mut self, total_residuals: usize) {
        self.context_model.num_of_residuals(total_residuals);
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_residual(value)?;
        self.add_data(BvGraphComponent::Residual, value);
        Ok(result)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.estimator.flush()
    }

    fn end_node(&mut self, node: usize) -> Result<usize, Self::Error> {
        self.estimator.end_node(node)
    }
}

impl<EP: EncodeParams, E: Encode, S: ContextModel> EncodeAndEstimate
    for HuffmanGraphEncoderBuilder<E, S, EP>
{
    type Estimator<'a>
        = &'a mut E
    where
        Self: 'a;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        &mut self.estimator
    }
}
