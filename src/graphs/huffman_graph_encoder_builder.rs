use dsi_bitstream::traits::{BitWrite, Endianness};
use webgraph::prelude::*;

use crate::huffman::{EncodeParams, HuffmanEncoder, IntegerData};

use crate::graphs::{BvGraphComponent, HuffmanGraphEncoder};

/// Builder to construct a HuffmanGraphEncoder, it requires an estimator that implements the Encode
/// trait and is used for estimating the adjacnecy list size of each node during the process of
/// reference selection.
pub struct HuffmanGraphEncoderBuilder<EE: Encode> {
    estimator: EE,
    data: IntegerData,
}

impl<EE: Encode> HuffmanGraphEncoderBuilder<EE> {
    // TODO: for now num_contexts is only 1
    pub fn new(estimator: EE) -> Self {
        Self {
            estimator,
            data: IntegerData::new(BvGraphComponent::COMPONENTS),
        }
    }

    pub fn build<EP: EncodeParams, E: Endianness, W: BitWrite<E>>(
        self,
        writer: &'_ mut W,
        max_bits: usize,
    ) -> HuffmanGraphEncoder<'_, EP, E, EE, W> {
        let encoder = HuffmanEncoder::<EP>::new(&self.data, max_bits);
        HuffmanGraphEncoder::new(encoder, self.estimator, writer)
    }
}

impl<EE: Encode> Encode for HuffmanGraphEncoderBuilder<EE> {
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

impl<EE: Encode> EncodeAndEstimate for HuffmanGraphEncoderBuilder<EE> {
    type Estimator<'a> = &'a mut EE where Self:'a;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        &mut self.estimator
    }
}
