use std::{convert::Infallible, marker::PhantomData};

use crate::huffman::{EncodeParams, HuffmanEncoder};

use anyhow::Result;
use dsi_bitstream::traits::{BitWrite, Endianness};
use webgraph::prelude::*;

use super::{BvGraphComponent, ContextChoiceStrategy};

/// Encoder to compress a graph using Hybrid integer huffman encoding.
/// This encoder can be constructed using an `HuffmanGraphEncoderBuilder`
pub struct HuffmanGraphEncoder<
    'a,
    EP: EncodeParams,
    E: Endianness,
    EE: Encode,
    W: BitWrite<E>,
    S: ContextChoiceStrategy,
> {
    encoder: HuffmanEncoder<EP>,
    estimator: EE,
    context_strategy: S,
    writer: &'a mut W,
    _marker: PhantomData<E>,
}
impl<'a, EP: EncodeParams, E: Endianness, EE: Encode, W: BitWrite<E>, S: ContextChoiceStrategy>
    HuffmanGraphEncoder<'a, EP, E, EE, W, S>
{
    pub(crate) fn new(
        encoder: HuffmanEncoder<EP>,
        estimator: EE,
        context_strategy: S,
        writer: &'a mut W,
    ) -> Self {
        Self {
            encoder,
            estimator,
            context_strategy,
            writer,
            _marker: PhantomData,
        }
    }

    pub fn write_header(&mut self) -> Result<usize> {
        self.encoder.write_header(self.writer)
    }

    fn write(&mut self, component: BvGraphComponent, value: u64) -> Result<usize> {
        let ctx = self.context_strategy.choose_context(component);
        let (token_bits, trailing_bits) = self.encoder.write(ctx, value as u32, self.writer)?;
        self.context_strategy.update(component, value);
        Ok(token_bits + trailing_bits)
    }
}

impl<'a, EP: EncodeParams, E: Endianness, EE: Encode, W: BitWrite<E>, S: ContextChoiceStrategy>
    Encode for HuffmanGraphEncoder<'a, EP, E, EE, W, S>
{
    type Error = Infallible;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
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

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(BvGraphComponent::Residual, value).unwrap();
        Ok(result)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.writer.flush().unwrap();
        Ok(0)
    }

    fn end_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

impl<'a, EP: EncodeParams, E: Endianness, EE: Encode, W: BitWrite<E>, S: ContextChoiceStrategy>
    EncodeAndEstimate for HuffmanGraphEncoder<'a, EP, E, EE, W, S>
{
    type Estimator<'b> = &'b mut EE where Self:'b;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        &mut self.estimator
    }
}
