use dsi_bitstream::traits::{BitWrite, Endianness};
use webgraph::prelude::*;

use crate::huffman::{EncodeParams, HuffmanEncoder, IntegerHistogram};

use crate::graphs::{estimator::HuffmanEstimator, BvGraphComponent, HuffmanGraphEncoder};

/// Builder to construct a HuffmanGraphEncoder, it requires an estimator that implements the Encode
/// trait and is used for estimating the adjacency list size of each node during the process of
/// reference selection.
pub struct HuffmanGraphEncoderBuilder<EP: EncodeParams, EE: Encode> {
    estimator: EE,
    data: IntegerHistogram<EP>,
}

impl<EP: EncodeParams, EE: Encode> HuffmanGraphEncoderBuilder<EP, EE> {
    // TODO: for now num_contexts is the number of components
    pub fn new(num_symbols: usize, estimator: EE) -> Self {
        Self {
            estimator,
            data: IntegerHistogram::new(BvGraphComponent::COMPONENTS, num_symbols),
        }
    }

    pub fn build<E: Endianness, W: BitWrite<E>>(
        self,
        writer: &'_ mut W,
        max_bits: usize,
    ) -> HuffmanGraphEncoder<'_, EP, E, EE, W> {
        let encoder = HuffmanEncoder::<EP>::new(self.data, max_bits);
        HuffmanGraphEncoder::new(encoder, self.estimator, writer)
    }

    pub fn build_estimator(self) -> HuffmanEstimator<EP> {
        HuffmanEstimator::new(self.data)
    }
}

impl<EP: EncodeParams, EE: Encode> Encode for HuffmanGraphEncoderBuilder<EP, EE> {
    type Error = EE::Error;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_outdegree(value)?;
        self.data
            .add(BvGraphComponent::Outdegree as u8, value as u32);
        Ok(result)
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_reference_offset(value)?;
        self.data
            .add(BvGraphComponent::ReferenceOffset as u8, value as u32);
        Ok(result)
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_block_count(value)?;
        self.data
            .add(BvGraphComponent::BlockCount as u8, value as u32);
        Ok(result)
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_block(value)?;
        self.data.add(BvGraphComponent::Blocks as u8, value as u32);
        Ok(result)
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_count(value)?;
        self.data
            .add(BvGraphComponent::IntervalCount as u8, value as u32);
        Ok(result)
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_start(value)?;
        self.data
            .add(BvGraphComponent::IntervalStart as u8, value as u32);
        Ok(result)
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_interval_len(value)?;
        self.data
            .add(BvGraphComponent::IntervalLen as u8, value as u32);
        Ok(result)
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_first_residual(value)?;
        self.data
            .add(BvGraphComponent::FirstResidual as u8, value as u32);
        Ok(result)
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.estimator.write_residual(value)?;
        self.data
            .add(BvGraphComponent::Residual as u8, value as u32);
        Ok(result)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.estimator.flush()
    }

    fn end_node(&mut self, node: usize) -> Result<usize, Self::Error> {
        self.estimator.end_node(node)
    }
}

impl<EP: EncodeParams, EE: Encode> EncodeAndEstimate for HuffmanGraphEncoderBuilder<EP, EE> {
    type Estimator<'a> = &'a mut EE where Self:'a;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        &mut self.estimator
    }
}
