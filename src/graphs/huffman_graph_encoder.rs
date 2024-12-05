use std::{convert::Infallible, marker::PhantomData};

use crate::huffman::{EncodeParams, HuffmanEncoder};

use anyhow::Result;
use dsi_bitstream::traits::{BitWrite, Endianness};
use webgraph::prelude::*;

/// Encoder to compress a graph using Hybrid integer huffman encoding.
/// This encoder can be constructed using a HuffmanGraphEncoderBuilder,
/// and can also be used as an estimator if a MockBitWriter is used as writer
pub struct HuffmanGraphEncoder<'a, EP: EncodeParams, E: Endianness, W: BitWrite<E>> {
    encoder: HuffmanEncoder<EP>,
    writer: &'a mut W,
    _marker: PhantomData<E>,
}
impl<'a, EP: EncodeParams, E: Endianness, W: BitWrite<E>> HuffmanGraphEncoder<'a, EP, E, W> {
    pub(crate) fn new(encoder: HuffmanEncoder<EP>, writer: &'a mut W) -> Self {
        Self {
            encoder,
            writer,
            _marker: PhantomData,
        }
    }

    fn write(&mut self, value: u64) -> Result<usize> {
        let (token_bits, trailing_bits) = self.encoder.write(0, value as u32, self.writer)?;
        Ok(token_bits + trailing_bits)
    }
}

impl<'a, EP: EncodeParams, E: Endianness, W: BitWrite<E>> Encode
    for HuffmanGraphEncoder<'a, EP, E, W>
{
    type Error = Infallible;

    fn start_node(&mut self, _node: usize) -> Result<usize, Self::Error> {
        Ok(0)
    }

    fn write_outdegree(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_reference_offset(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_block_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_block(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_interval_count(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_interval_start(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_interval_len(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_first_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
        Ok(result)
    }

    fn write_residual(&mut self, value: u64) -> Result<usize, Self::Error> {
        let result = self.write(value).unwrap();
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

impl<'a, EP: EncodeParams, E: Endianness, W: BitWrite<E>> EncodeAndEstimate
    for HuffmanGraphEncoder<'a, EP, E, W>
{
    type Estimator<'b> = &'b mut Self where Self:'b;

    fn estimator(&mut self) -> Self::Estimator<'_> {
        self
    }
}
