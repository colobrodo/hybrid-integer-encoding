use dsi_bitstream::traits::{BitRead, Endianness};
use webgraph::prelude::*;

use crate::huffman::{EncodeParams, EntropyCoder, HuffmanReader};

use super::{BvGraphComponent, ContextChoiceStrategy};

pub struct HuffmanGraphDecoder<
    EP: EncodeParams,
    E: Endianness,
    R: BitRead<E>,
    S: ContextChoiceStrategy,
> {
    reader: HuffmanReader<E, R>,
    context_strategy: S,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams, E: Endianness, R: BitRead<E>, S: ContextChoiceStrategy>
    HuffmanGraphDecoder<EP, E, R, S>
{
    pub fn new(reader: HuffmanReader<E, R>, strategy: S) -> Self {
        HuffmanGraphDecoder {
            reader,
            context_strategy: strategy,
            _marker: std::marker::PhantomData,
        }
    }

    fn read(&mut self, component: BvGraphComponent) -> u64 {
        let context = self.context_strategy.choose_context(component);
        let symbol = self
            .reader
            .read::<EP>(context as usize)
            .expect("Reading symbol from huffman reader during graph decoding")
            as u64;
        self.context_strategy.update(component, symbol);
        symbol
    }
}

impl<EP: EncodeParams, E: Endianness, R: BitRead<E>, S: ContextChoiceStrategy> Decode
    for HuffmanGraphDecoder<EP, E, R, S>
{
    fn read_outdegree(&mut self) -> u64 {
        self.read(BvGraphComponent::Outdegree)
    }

    fn read_reference_offset(&mut self) -> u64 {
        self.read(BvGraphComponent::ReferenceOffset)
    }

    fn read_block_count(&mut self) -> u64 {
        self.read(BvGraphComponent::BlockCount)
    }

    fn read_block(&mut self) -> u64 {
        self.read(BvGraphComponent::Blocks)
    }

    fn read_interval_count(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalCount)
    }

    fn read_interval_start(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalStart)
    }

    fn read_interval_len(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalLen)
    }

    fn read_first_residual(&mut self) -> u64 {
        self.read(BvGraphComponent::FirstResidual)
    }

    fn read_residual(&mut self) -> u64 {
        self.read(BvGraphComponent::Residual)
    }
}
