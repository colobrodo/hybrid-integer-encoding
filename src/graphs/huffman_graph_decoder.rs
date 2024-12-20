use dsi_bitstream::traits::{BitRead, BitSeek, Endianness};
use epserde::deser::MemCase;
use sux::traits::IndexedSeq;
use webgraph::prelude::*;

use crate::huffman::{DefaultEncodeParams, EncodeParams, EntropyCoder, HuffmanReader};

use super::{BvGraphComponent, ContextChoiceStrategy, SimpleChoiceStrategy};

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

impl<EP: EncodeParams, E: Endianness, R: BitRead<E> + BitSeek, S: ContextChoiceStrategy> BitSeek
    for HuffmanGraphDecoder<EP, E, R, S>
{
    type Error = <R as BitSeek>::Error;

    fn bit_pos(&mut self) -> Result<u64, Self::Error> {
        self.reader.bit_pos()
    }

    fn set_bit_pos(&mut self, bit_pos: u64) -> Result<(), Self::Error> {
        self.reader.set_bit_pos(bit_pos)
    }
}

pub struct SequentialHuffmanDecoderFactory<
    EP: EncodeParams,
    E: Endianness,
    F: BitReaderFactory<E>,
    S: ContextChoiceStrategy,
> {
    _marker: core::marker::PhantomData<(EP, E, F, S)>,
    factory: F,
    max_bits: usize,
}

impl<EP: EncodeParams, E: Endianness, F: BitReaderFactory<E>>
    SequentialHuffmanDecoderFactory<EP, E, F, SimpleChoiceStrategy>
{
    pub fn new(factory: F, max_bits: usize) -> Self {
        SequentialHuffmanDecoderFactory {
            factory,
            max_bits,
            _marker: std::marker::PhantomData,
        }
    }
}

// TODO: make this work for any context choice strategy: maybe pass a ContextChoiceStrategyFactory and implement this trait for lambdas that return
//       ContextChoiceStrategy to pass SimpleChoiceStartegy::default and do not create other useless objects
impl<EP: EncodeParams, E: Endianness, F: BitReaderFactory<E>> SequentialDecoderFactory
    for SequentialHuffmanDecoderFactory<EP, E, F, SimpleChoiceStrategy>
where
    for<'a> <F as BitReaderFactory<E>>::BitReader<'a>: BitRead<E>,
{
    type Decoder<'a> = HuffmanGraphDecoder<EP, E, <F as BitReaderFactory<E>>::BitReader<'a>, SimpleChoiceStrategy>
    where Self:'a;

    fn new_decoder(&self) -> anyhow::Result<Self::Decoder<'_>> {
        let reader = self.factory.new_reader();
        let strategy = SimpleChoiceStrategy;
        let huffman_reader =
            HuffmanReader::from_bitreader(reader, self.max_bits, strategy.num_contexts())?;
        Ok(HuffmanGraphDecoder::new(
            huffman_reader,
            SimpleChoiceStrategy,
        ))
    }
}

pub struct RandomAccessHuffmanDecoderFactory<
    E: Endianness,
    F: BitReaderFactory<E>,
    OFF: IndexedSeq<Input = usize, Output = usize>,
    S: ContextChoiceStrategy,
    EP: EncodeParams = DefaultEncodeParams,
> {
    _marker: core::marker::PhantomData<(EP, E, F, S)>,
    factory: F,
    /// The offsets into the data.
    offsets: MemCase<OFF>,
    max_bits: usize,
}

impl<
        E: Endianness,
        OFF: IndexedSeq<Input = usize, Output = usize>,
        F: BitReaderFactory<E>,
        EP: EncodeParams,
    > RandomAccessHuffmanDecoderFactory<E, F, OFF, SimpleChoiceStrategy, EP>
{
    pub fn new(factory: F, offsets: MemCase<OFF>, max_bits: usize) -> Self {
        RandomAccessHuffmanDecoderFactory {
            offsets,
            factory,
            max_bits,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        E: Endianness,
        OFF: IndexedSeq<Input = usize, Output = usize>,
        F: BitReaderFactory<E>,
        EP: EncodeParams,
    > RandomAccessDecoderFactory
    for RandomAccessHuffmanDecoderFactory<E, F, OFF, SimpleChoiceStrategy, EP>
where
    for<'a> <F as BitReaderFactory<E>>::BitReader<'a>: BitRead<E> + BitSeek,
{
    type Decoder<'a> = HuffmanGraphDecoder<EP, E, <F as BitReaderFactory<E>>::BitReader<'a>, SimpleChoiceStrategy>
    where Self:'a;

    fn new_decoder(&self, node: usize) -> anyhow::Result<Self::Decoder<'_>> {
        let mut reader = self.factory.new_reader();
        let strategy = SimpleChoiceStrategy;
        // TODO: FIXME: now we are decoding the table at each random access (bleah) (this function is called each time successors is called)
        //              we should decode it only when the factory is created and then use HuffmanReader::new
        //              to do so is better if we use a strategy factory that can tell how many contexts you can use in advance (you know)
        let table =
            HuffmanReader::decode_table(&mut reader, self.max_bits, strategy.num_contexts())?;
        reader.set_bit_pos(self.offsets.get(node) as u64)?;
        let huffman_reader = HuffmanReader::new(table, reader);
        Ok(HuffmanGraphDecoder::new(huffman_reader, strategy))
    }
}
