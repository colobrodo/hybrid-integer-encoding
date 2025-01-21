use dsi_bitstream::traits::{BitRead, BitSeek, LE};
use epserde::deser::MemCase;
use sux::traits::IndexedSeq;
use webgraph::prelude::*;

use crate::huffman::{
    DefaultEncodeParams, EncodeParams, EntropyCoder, HuffmanReader, HuffmanTable,
};

use super::{BvGraphComponent, ContextChoiceStrategy, SimpleChoiceStrategy};

pub struct HuffmanGraphDecoder<EP: EncodeParams, R: BitRead<LE>, S: ContextChoiceStrategy> {
    reader: HuffmanReader<R>,
    context_strategy: S,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams, R: BitRead<LE>, S: ContextChoiceStrategy> HuffmanGraphDecoder<EP, R, S> {
    pub fn new(reader: HuffmanReader<R>, strategy: S) -> Self {
        HuffmanGraphDecoder {
            reader,
            context_strategy: strategy,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
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

impl<EP: EncodeParams, R: BitRead<LE>, S: ContextChoiceStrategy> Decode
    for HuffmanGraphDecoder<EP, R, S>
{
    #[inline(always)]
    fn read_outdegree(&mut self) -> u64 {
        self.read(BvGraphComponent::Outdegree)
    }

    #[inline(always)]
    fn read_reference_offset(&mut self) -> u64 {
        self.read(BvGraphComponent::ReferenceOffset)
    }

    #[inline(always)]
    fn read_block_count(&mut self) -> u64 {
        self.read(BvGraphComponent::BlockCount)
    }

    #[inline(always)]
    fn read_block(&mut self) -> u64 {
        self.read(BvGraphComponent::Blocks)
    }

    #[inline(always)]
    fn read_interval_count(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalCount)
    }

    #[inline(always)]
    fn read_interval_start(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalStart)
    }

    #[inline(always)]
    fn read_interval_len(&mut self) -> u64 {
        self.read(BvGraphComponent::IntervalLen)
    }

    #[inline(always)]
    fn read_first_residual(&mut self) -> u64 {
        self.read(BvGraphComponent::FirstResidual)
    }

    #[inline(always)]
    fn read_residual(&mut self) -> u64 {
        self.read(BvGraphComponent::Residual)
    }
}

impl<EP: EncodeParams, R: BitRead<LE> + BitSeek, S: ContextChoiceStrategy> BitSeek
    for HuffmanGraphDecoder<EP, R, S>
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
    F: BitReaderFactory<LE>,
    S: ContextChoiceStrategy,
> {
    _marker: core::marker::PhantomData<(EP, S)>,
    factory: F,
    max_bits: usize,
}

impl<EP: EncodeParams, F: BitReaderFactory<LE>>
    SequentialHuffmanDecoderFactory<EP, F, SimpleChoiceStrategy>
{
    pub fn new(factory: F, max_bits: usize) -> Self {
        SequentialHuffmanDecoderFactory {
            factory,
            max_bits,
            _marker: std::marker::PhantomData,
        }
    }
}

// TODO: make this work for any context choice strategy: we should pass something that creates a strategy
//       each time with a fixed amount of contexts know in advance to be able to create the huffman header
//       and reuse it at each random access like now
impl<EP: EncodeParams, F: BitReaderFactory<LE>> SequentialDecoderFactory
    for SequentialHuffmanDecoderFactory<EP, F, SimpleChoiceStrategy>
where
    for<'a> <F as BitReaderFactory<LE>>::BitReader<'a>: BitRead<LE>,
{
    type Decoder<'a>
        = HuffmanGraphDecoder<EP, <F as BitReaderFactory<LE>>::BitReader<'a>, SimpleChoiceStrategy>
    where
        Self: 'a;

    fn new_decoder(&self) -> anyhow::Result<Self::Decoder<'_>> {
        let reader = self.factory.new_reader();
        let strategy = SimpleChoiceStrategy;
        let huffman_reader =
            HuffmanReader::from_bitreader(reader, self.max_bits, strategy.num_contexts())?;
        Ok(HuffmanGraphDecoder::new(huffman_reader, strategy))
    }
}

pub struct RandomAccessHuffmanDecoderFactory<
    F: BitReaderFactory<LE>,
    OFF: IndexedSeq<Input = usize, Output = usize>,
    S: ContextChoiceStrategy + Clone,
    EP: EncodeParams = DefaultEncodeParams,
> {
    _marker: core::marker::PhantomData<(EP, S)>,
    factory: F,
    /// The offsets into the data.
    offsets: MemCase<OFF>,
    //TODO: for now we support only stateless context choice strategy
    strategy: S,
    table: HuffmanTable,
}

impl<
        OFF: IndexedSeq<Input = usize, Output = usize>,
        F: BitReaderFactory<LE>,
        S: ContextChoiceStrategy + Clone,
        EP: EncodeParams,
    > RandomAccessHuffmanDecoderFactory<F, OFF, S, EP>
where
    for<'a> <F as BitReaderFactory<LE>>::BitReader<'a>: BitRead<LE> + BitSeek,
{
    pub fn new(
        factory: F,
        strategy: S,
        offsets: MemCase<OFF>,
        max_bits: usize,
    ) -> anyhow::Result<Self> {
        let mut reader = factory.new_reader();
        let table = HuffmanReader::decode_table(&mut reader, max_bits, strategy.num_contexts())?;
        drop(reader);
        Ok(RandomAccessHuffmanDecoderFactory {
            offsets,
            factory,
            table,
            strategy,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<
        OFF: IndexedSeq<Input = usize, Output = usize>,
        F: BitReaderFactory<LE>,
        EP: EncodeParams,
        S: ContextChoiceStrategy + Copy,
    > RandomAccessDecoderFactory for RandomAccessHuffmanDecoderFactory<F, OFF, S, EP>
where
    for<'a> <F as BitReaderFactory<LE>>::BitReader<'a>: BitRead<LE> + BitSeek,
{
    type Decoder<'a>
        = HuffmanGraphDecoder<EP, <F as BitReaderFactory<LE>>::BitReader<'a>, S>
    where
        Self: 'a;

    fn new_decoder(&self, node: usize) -> anyhow::Result<Self::Decoder<'_>> {
        let mut reader = self.factory.new_reader();
        reader.set_bit_pos(self.offsets.get(node) as u64)?;
        // TODO: remove the clone of the whole table
        let huffman_reader = HuffmanReader::new(self.table.clone(), reader);
        Ok(HuffmanGraphDecoder::new(huffman_reader, self.strategy))
    }
}
