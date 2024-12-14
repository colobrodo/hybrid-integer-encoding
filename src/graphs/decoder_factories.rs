use dsi_bitstream::traits::{BitRead, Endianness};
use webgraph::prelude::*;

use crate::huffman::{EncodeParams, HuffmanReader};

use super::{ContextChoiceStrategy, HuffmanGraphDecoder, SimpleChoiceStrategy};

pub struct HuffmanGraphDecoderFactory<
    EP: EncodeParams,
    E: Endianness,
    F: BitReaderFactory<E>,
    S: ContextChoiceStrategy,
> {
    _marker: core::marker::PhantomData<(EP, E, F, S)>,
    factory: F,
    max_bits: usize,
    num_contexts: usize,
}

// TODO: make this work for any context choice strategy: maybe pass a ContextChoiceStrategyFactory and implement this trait for lambdas that return
//       ContextChoiceStrategy to pass SimpleChoiceStartegy::default and do not create other useless objects
impl<EP: EncodeParams, E: Endianness, F: BitReaderFactory<E>> SequentialDecoderFactory
    for HuffmanGraphDecoderFactory<EP, E, F, SimpleChoiceStrategy>
where
    for<'a> <F as BitReaderFactory<E>>::BitReader<'a>: BitRead<E>,
{
    type Decoder<'a> = HuffmanGraphDecoder<EP, E, <F as BitReaderFactory<E>>::BitReader<'a>, SimpleChoiceStrategy>
    where Self:'a;

    fn new_decoder(&self) -> anyhow::Result<Self::Decoder<'_>> {
        let reader = self.factory.new_reader();
        let huffman_reader = HuffmanReader::new(reader, self.max_bits, self.num_contexts)?;
        Ok(HuffmanGraphDecoder::new(
            huffman_reader,
            SimpleChoiceStrategy,
        ))
    }
}
