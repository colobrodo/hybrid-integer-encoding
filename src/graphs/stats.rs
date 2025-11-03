use std::sync::Mutex;

use dsi_bitstream::traits::BitSeek;
use webgraph::prelude::*;

/// Represent the overall usage of bits in the encoding of the graph.
/// As StatsDecoder it differs from the webgraph counterpart due to the
/// fact that this keeps track of the number of bit read (and so by any encoder)
/// while the other simultaneously count the space of the stream encoded with all the
/// instantaneous codes.  
#[derive(Default)]
pub struct GraphStats {
    pub total: u64,
    pub outdegrees: u64,
    pub reference_offsets: u64,
    pub block_counts: u64,
    pub blocks: u64,
    pub interval_counts: u64,
    pub interval_starts: u64,
    pub interval_lens: u64,
    pub first_residuals: u64,
    pub residuals: u64,
}

impl GraphStats {
    fn update(&mut self, other: &Self) {
        self.total += other.total;
        self.outdegrees += other.outdegrees;
        self.reference_offsets += other.reference_offsets;
        self.block_counts += other.block_counts;
        self.blocks += other.blocks;
        self.interval_counts += other.interval_counts;
        self.interval_starts += other.interval_starts;
        self.interval_lens += other.interval_lens;
        self.first_residuals += other.first_residuals;
        self.residuals += other.residuals;
    }
}

/// A wrapper on adjacency list decoders to count the number of bits read for each component.
/// The wrapped decoder should be seekable to know the number of bits read before and
/// after each call.  
/// This is different from webgraph-rs's StatsDecoder because it aim to find the
/// best instantaneous code to compress each part.
pub struct StatsDecoder<'a, D: Decode + BitSeek, F: SequentialDecoderFactory> {
    decoder: D,
    stats: GraphStats,
    factory: &'a StatsDecoderFactory<F>,
}

impl<'a, D: Decode + BitSeek, F: SequentialDecoderFactory> StatsDecoder<'a, D, F> {
    /// Perform an operation on the decoder and returns the result along with the
    /// number of bits consumed.
    fn measure_bits<FN, R>(&mut self, f: FN) -> (u64, R)
    where
        FN: FnOnce(&mut D) -> R,
    {
        let start_bit_pos = self
            .decoder
            .bit_pos()
            .expect("Cannot seek into graph stat decoder bitstream.");
        let result = f(&mut self.decoder);
        let end_bit_pos = self
            .decoder
            .bit_pos()
            .expect("Cannot seek into graph stat decoder bitstream.");
        let bit_used = end_bit_pos - start_bit_pos;
        (bit_used, result)
    }
}

impl<D: Decode + BitSeek, F: SequentialDecoderFactory> Drop for StatsDecoder<'_, D, F> {
    fn drop(&mut self) {
        self.factory.glob_stats.lock().unwrap().update(&self.stats);
    }
}

impl<'a, D: Decode + BitSeek, F: SequentialDecoderFactory> StatsDecoder<'a, D, F> {
    fn new(factory: &'a StatsDecoderFactory<F>, decoder: D) -> StatsDecoder<'a, D, F> {
        StatsDecoder {
            decoder,
            stats: GraphStats::default(),
            factory,
        }
    }
}

impl<D: Decode + BitSeek, F: SequentialDecoderFactory> Decode for StatsDecoder<'_, D, F> {
    fn read_outdegree(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_outdegree());
        self.stats.total += bit_used;
        self.stats.outdegrees += bit_used;
        result
    }

    fn read_reference_offset(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_reference_offset());
        self.stats.total += bit_used;
        self.stats.reference_offsets += bit_used;
        result
    }

    fn read_block_count(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_block_count());
        self.stats.total += bit_used;
        self.stats.block_counts += bit_used;
        result
    }

    fn read_block(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_block());
        self.stats.total += bit_used;
        self.stats.blocks += bit_used;
        result
    }

    fn read_interval_count(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_interval_count());
        self.stats.total += bit_used;
        self.stats.interval_counts += bit_used;
        result
    }

    fn read_interval_start(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_interval_start());
        self.stats.total += bit_used;
        self.stats.interval_starts += bit_used;
        result
    }

    fn read_interval_len(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_interval_len());
        self.stats.total += bit_used;
        self.stats.interval_lens += bit_used;
        result
    }

    fn read_first_residual(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_first_residual());
        self.stats.total += bit_used;
        self.stats.first_residuals += bit_used;
        result
    }

    fn read_residual(&mut self) -> u64 {
        let (bit_used, result) = self.measure_bits(|d| d.read_residual());
        self.stats.total += bit_used;
        self.stats.residuals += bit_used;
        result
    }
}

/// A wrapper that keeps track of how much bits each piece would take using
/// different codes for compressions for a [`SequentialDecoderFactory`]
/// implementation and returns the stats.
pub struct StatsDecoderFactory<F: SequentialDecoderFactory> {
    factory: F,
    glob_stats: Mutex<GraphStats>,
}

impl<F> StatsDecoderFactory<F>
where
    F: SequentialDecoderFactory,
{
    pub fn new(factory: F) -> Self {
        Self {
            factory,
            glob_stats: Mutex::new(GraphStats::default()),
        }
    }

    /// Consume self and return the stats.
    pub fn stats(self) -> GraphStats {
        self.glob_stats.into_inner().unwrap()
    }
}

impl<F> From<F> for StatsDecoderFactory<F>
where
    F: SequentialDecoderFactory,
{
    #[inline(always)]
    fn from(value: F) -> Self {
        Self::new(value)
    }
}

impl<F> SequentialDecoderFactory for StatsDecoderFactory<F>
where
    F: SequentialDecoderFactory,
    for<'a> F::Decoder<'a>: BitSeek, // Enforce BitSeek on the decoder type
{
    type Decoder<'a>
        = StatsDecoder<'a, F::Decoder<'a>, F>
    where
        Self: 'a;

    #[inline(always)]
    fn new_decoder(&self) -> anyhow::Result<Self::Decoder<'_>> {
        let inner_decoder = self.factory.new_decoder()?;
        let stats = StatsDecoder::new(self, inner_decoder);
        Ok(stats)
    }
}
