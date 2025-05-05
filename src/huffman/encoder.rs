use std::mem;

use dsi_bitstream::traits::{BitWrite, Endianness, LE};

use crate::huffman::compute_symbol_bits;

use super::common::{
    compute_symbol_len_bits, encode, DefaultEncodeParams, EncodeParams, HuffmanSymbolInfo,
};

use anyhow::Result;

// Instead of calculating the costs (in bit) to encode each symbol in the estimation round
// (with Huffman estimators), we precalculate the length of each token and avoid the
// expensive divisions

/// Define the cost of encoding each token according to a distribution
/// defined by `IntegerHistogram`.
pub struct CostModel<EP: EncodeParams = DefaultEncodeParams> {
    costs: Vec<Vec<usize>>,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams> CostModel<EP> {
    /// Returns the estimated cost of encoding the value in the given context.
    pub fn cost(&self, context: u8, value: u64) -> usize {
        let (token, n_bits, _) = encode::<EP>(value);
        self.costs[context as usize][token] + n_bits
    }
}

pub struct HuffmanEncoder<EP: EncodeParams = DefaultEncodeParams> {
    // maximum number of bits per code
    max_bits: usize,
    // A table of huffman symbol infos for each context, of dimension NUM_SYMBOLS * NUM_CONTEXT
    info_: Vec<Vec<HuffmanSymbolInfo>>,
    _marker: core::marker::PhantomData<EP>,
}

/// Compute the histogram of the each token frequency for each context.
/// An histogram is a vector of length `num_symbols` where each index represents a symbol,
/// and the value at each index represents the frequency of the symbol.
pub struct IntegerHistograms<EP: EncodeParams> {
    ctx_histograms: Vec<Vec<usize>>,
    /// Total number of symbols for each context.
    totals: Vec<usize>,
    num_contexts: usize,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams> IntegerHistograms<EP> {
    pub fn new(num_contexts: usize, num_symbols: usize) -> Self {
        let mut histograms = Vec::with_capacity(num_contexts);
        histograms.resize(num_contexts, vec![0; num_symbols]);
        Self {
            ctx_histograms: histograms,
            totals: vec![0; num_contexts],
            num_contexts,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn context_count(&self, context: u8) -> usize {
        self.totals[context as usize]
    }

    pub fn count(&self) -> usize {
        self.totals.iter().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn add(&mut self, context: u8, value: u32) {
        debug_assert!(
            (context as usize) < self.num_contexts,
            "Context out of bounds trying to add symbol {} on context {}, but only {} contexts are available",
            value, context, self.num_contexts
        );
        let (token, _, _) = encode::<EP>(value.into());
        self.ctx_histograms[context as usize][token] += 1;
        self.totals[context as usize] += 1;
    }

    /// Returns the histogram as his underlying vector.
    pub fn as_vec(self) -> Vec<Vec<usize>> {
        self.ctx_histograms
    }

    /// Returns the cost model of encoding each value according to the distributions
    /// defined by this histogram
    pub fn cost(&self) -> CostModel<EP> {
        let mut costs = self
            .ctx_histograms
            .iter()
            .map(|histogram| Vec::with_capacity(histogram.len()))
            .collect::<Vec<_>>();
        for (ctx, ctx_histogram) in self.ctx_histograms.iter().enumerate() {
            let total_symbols = self.totals[ctx];
            for &freq in ctx_histogram.iter() {
                let cnt = f64::max(freq as f64, 0.1);
                let inv_freq = (total_symbols as f64 / cnt) as u64;
                let token_cost = inv_freq.max(2).ilog2() as usize;
                costs[ctx].push(token_cost);
            }
        }
        CostModel {
            costs,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn number_of_contexts(&self) -> usize {
        self.num_contexts
    }
}

type Bag = (usize, Vec<u8>);

// Compute the optimal number of bits for each symbol given the input
// distribution. Uses a (quadratic version) of the package-merge/coin-collector
// algorithm.
fn compute_symbol_num_bits(histogram: &[usize], max_bits: usize, infos: &mut [HuffmanSymbolInfo]) {
    assert!(infos.len() == 1 << max_bits);
    // Mark the present/missing symbols.
    let mut non_zero_symbols = 0;
    for (i, freq) in histogram.iter().enumerate() {
        if *freq == 0 {
            continue;
        }
        infos[i].present = 1;
        non_zero_symbols += 1;
    }
    if non_zero_symbols <= 1 {
        for info in infos.iter_mut() {
            if info.present != 0 {
                info.n_bits = 1;
            }
        }
        return;
    }

    // Create a list of symbols for any given cost.
    let mut bags = vec![Vec::<Bag>::default(); max_bits];
    for bag in bags.iter_mut() {
        for (s, info) in infos.iter().enumerate() {
            if info.present == 0 {
                continue;
            }
            let sym = vec![s as u8];
            bag.push((histogram[s], sym));
        }
    }

    // Pair up symbols (or groups of symbols) of a given bit-length to create
    // symbols of the following bit-length, creating pairs by merging (groups of)
    // symbols consecutively in increasing order of cost.
    for i in 0..(max_bits - 1) {
        bags[i].sort();
        for j in (0..bags[i].len() - 1).step_by(2) {
            let nf = bags[i][j].0 + bags[i][j + 1].0;
            let mut n_sym = mem::take(&mut bags[i][j].1);
            n_sym.append(&mut bags[i][j + 1].1);
            bags[i + 1].push((nf, n_sym));
        }
    }
    bags[max_bits - 1].sort();

    // In the groups of symbols for the highest bit length we need to select the
    // last 2*num_symbols-2 groups, and assign to each symbol one bit of cost for
    // each of its occurrences in these groups.
    for i in 0..2 * non_zero_symbols - 2 {
        let b = &bags[max_bits - 1][i];
        for &x in b.1.iter() {
            infos[x as usize].n_bits += 1;
        }
    }
}

impl<EP: EncodeParams> HuffmanEncoder<EP> {
    pub fn new(data: IntegerHistograms<EP>, max_bits: usize) -> Self {
        let num_symbols = 1 << max_bits;
        let histograms = data.as_vec();
        let mut info = Vec::new();
        for histogram in &histograms {
            let mut ctx_info = vec![HuffmanSymbolInfo::default(); num_symbols];
            compute_symbol_num_bits(histogram, max_bits, &mut ctx_info);
            compute_symbol_bits(max_bits, &mut ctx_info);
            info.push(ctx_info);
        }
        Self {
            info_: info,
            max_bits,
            _marker: core::marker::PhantomData,
        }
    }

    /// Write the value into the bit stream, if successful returns a tuple with the
    /// number of bits written for the symbol and for the trailing bits.
    pub fn write(
        &self,
        ctx: u8,
        value: u32,
        writer: &mut impl BitWrite<LE>,
    ) -> Result<(usize, usize)> {
        let (token, n_bits, bits) = encode::<EP>(value as u64);
        debug_assert!(
            self.info_[ctx as usize][token].present == 1,
            "Unknown value {value} in context {ctx}"
        );
        let n_bits_per_token = self.info_[ctx as usize][token].n_bits as usize;
        writer.write_bits(
            self.info_[ctx as usize][token].bits as u64,
            n_bits_per_token,
        )?;
        writer.write_bits(bits, n_bits)?;
        Ok((n_bits_per_token, n_bits))
    }

    // Very simple encoding: number of symbols (16 bits) followed by, for each
    // symbol, 1 bit for presence/absence, and 4 bits for symbol length if present.
    // TODO: short  encoding for empty ctxs, RLE for missing symbols.
    pub fn write_header<E: Endianness>(&self, writer: &mut impl BitWrite<E>) -> Result<usize> {
        // number of bits needed to represent the length of each symbol in the header
        let symbol_len_bits = compute_symbol_len_bits(self.max_bits as u32);
        let mut total_bits_written = 0;
        for info in self.info_.iter() {
            let mut ms = 0;
            for (i, sym_info) in info.iter().enumerate() {
                if sym_info.present != 0 {
                    ms = i;
                }
            }

            total_bits_written += writer.write_bits(ms as u64, self.max_bits)?;
            for sym_info in info.iter().take(ms + 1) {
                if sym_info.present != 0 {
                    writer.write_bits(1, 1)?;
                    writer.write_bits(sym_info.n_bits as u64 - 1, symbol_len_bits as usize)?;
                    total_bits_written += symbol_len_bits as usize + 1;
                } else {
                    writer.write_bits(0, 1)?;
                    total_bits_written += 1;
                }
            }
        }
        Ok(total_bits_written)
    }
}
