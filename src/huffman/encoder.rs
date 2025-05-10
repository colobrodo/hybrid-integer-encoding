use core::f32;
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

pub struct Histogram {
    frequencies: Vec<usize>,
    /// Total number of symbols
    total: usize,
}

impl Histogram {
    fn from_vec(freqs: Vec<usize>) -> Self {
        let total = freqs.iter().sum();
        Histogram {
            frequencies: freqs,
            total,
        }
    }

    fn empty() -> Self {
        Histogram {
            frequencies: Vec::new(),
            total: 0,
        }
    }

    fn empty_with_known_size(number_of_elements: usize) -> Self {
        Histogram {
            frequencies: vec![0; number_of_elements],
            total: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.total == 0
    }

    fn push(&mut self, symbol: usize) {
        if self.frequencies.len() < symbol {
            self.frequencies.resize(symbol + 1, 0);
        }
        self.frequencies[symbol] += 1;
        self.total += 1;
    }

    fn symbols(&self) -> usize {
        self.frequencies.len()
    }

    fn entropy(&self) -> f32 {
        let mut entropy = 0.0;
        for &freq in self.frequencies.iter() {
            if freq == 0 {
                continue;
            }
            entropy += freq as f32 * f32::log2(self.total as f32 / freq as f32)
        }
        entropy
    }

    fn add(&self, other: &Histogram) -> Histogram {
        let size = self.symbols();
        let other_size = other.symbols();
        let new_size = size.max(other_size);
        let mut new_freqs = vec![0; new_size];
        for i in 0..new_size {
            let freq = *self.frequencies.get(i).unwrap_or(&0);
            let other_freq = *other.frequencies.get(i).unwrap_or(&0);
            new_freqs[i] = freq + other_freq;
        }
        Histogram::from_vec(new_freqs)
    }

    fn distance(&self, other: &Histogram) -> f32 {
        let union = self.add(other);
        union.entropy() - self.entropy() - other.entropy()
    }
}

/// Compute the histogram of the each token frequency for each context.
/// An histogram is a vector of length `num_symbols` where each index represents a symbol,
/// and the value at each index represents the frequency of the symbol.
pub struct IntegerHistograms<EP: EncodeParams> {
    ctx_histograms: Vec<Histogram>,
    num_contexts: usize,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams> IntegerHistograms<EP> {
    pub fn new(num_contexts: usize, num_symbols: usize) -> Self {
        let mut histograms = Vec::with_capacity(num_contexts);
        histograms.resize_with(num_contexts, || {
            Histogram::empty_with_known_size(num_symbols)
        });
        Self {
            ctx_histograms: histograms,
            num_contexts,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn context_count(&self, context: u8) -> usize {
        self.ctx_histograms[context as usize].total
    }

    pub fn count(&self) -> usize {
        self.ctx_histograms.iter().map(|h| h.total).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ctx_histograms.iter().all(|h| h.is_empty())
    }

    pub fn add(&mut self, context: u8, value: u32) {
        debug_assert!(
            (context as usize) < self.num_contexts,
            "Context out of bounds trying to add symbol {} on context {}, but only {} contexts are available",
            value, context, self.num_contexts
        );
        let (token, _, _) = encode::<EP>(value.into());
        self.ctx_histograms[context as usize].push(token);
    }

    /// Returns the cost model of encoding each value according to the distributions
    /// defined by this histogram
    pub fn cost(&self) -> CostModel<EP> {
        let mut costs = self
            .ctx_histograms
            .iter()
            .map(|histogram| Vec::with_capacity(histogram.symbols()))
            .collect::<Vec<_>>();
        for (ctx, ctx_histogram) in self.ctx_histograms.iter().enumerate() {
            let total_symbols = self.ctx_histograms[ctx].total;
            for &freq in ctx_histogram.frequencies.iter() {
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

    pub fn print_entropies(&self) {
        for (ctx, histogram) in self.ctx_histograms.iter().enumerate() {
            let entropy = histogram.entropy();
            println!("Entropy for distribution in context {}: {}", ctx, entropy);
        }
        let clusters_map = self.cluster(20);
        // print all the clusters
        for (index, &cluster) in clusters_map.iter().enumerate() {
            println!(
                "cluster {} -> {} (type: {}, cluster type: {})",
                index,
                cluster,
                index % 9,
                cluster % 9
            );
        }
    }

    pub fn cluster(&self, num_clusters: usize) -> Vec<usize> {
        // find the histogram with maximal entropy
        let (max_entropy_index, _) = self
            .ctx_histograms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.entropy().partial_cmp(&b.entropy()).unwrap())
            .unwrap();
        let num_distributions = self.number_of_contexts();
        let mut clusters = Vec::new();
        let mut last_cluster_index = max_entropy_index;
        let mut chosen = vec![false; num_distributions];
        // contains the distance to the closer center
        let mut distances = vec![f32::INFINITY; num_distributions];
        while clusters.len() < num_clusters.min(self.ctx_histograms.len()) {
            clusters.push(last_cluster_index);
            chosen[last_cluster_index] = true;
            let mut farthest_histogram_distance = 0.0;
            let mut farthest_histogram_index = 0;
            let center_histogram = &self.ctx_histograms[last_cluster_index];
            // find the next cluster: the one farthest away from all the existing clusters
            for (index, histogram) in self.ctx_histograms.iter().enumerate() {
                let distance_to_last_center = histogram.distance(center_histogram);
                // avoid consider elements that are already centers
                if chosen[index] {
                    continue;
                }
                if distance_to_last_center < distances[index] {
                    distances[index] = distance_to_last_center;
                }
                if distances[index] > farthest_histogram_distance {
                    farthest_histogram_distance = distances[index];
                    farthest_histogram_index = index;
                }
            }
            last_cluster_index = farthest_histogram_index;
        }
        let mut clusters_map = vec![0; num_distributions];
        // choose the center of each point
        for (index, histogram) in self.ctx_histograms.iter().enumerate() {
            if chosen[index] {
                continue;
            }
            let mut closer_center_index = 0;
            let mut closer_center_distance = f32::INFINITY;
            for &center in clusters.iter() {
                let distance_to_center = self.ctx_histograms[center].distance(&histogram);
                if distance_to_center < closer_center_distance {
                    closer_center_distance = distance_to_center;
                    closer_center_index = center;
                }
            }
            clusters_map[index] = closer_center_index;
        }

        clusters_map
    }

    pub fn number_of_contexts(&self) -> usize {
        self.num_contexts
    }
}

type Bag = (usize, Vec<u8>);

// Compute the optimal number of bits for each symbol given the input
// distribution. Uses a (quadratic version) of the package-merge/coin-collector
// algorithm.
fn compute_symbol_num_bits(
    histogram: &Histogram,
    max_bits: usize,
    infos: &mut [HuffmanSymbolInfo],
) {
    assert!(infos.len() == 1 << max_bits);
    // Mark the present/missing symbols.
    let mut non_zero_symbols = 0;
    for (i, freq) in histogram.frequencies.iter().enumerate() {
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
            bag.push((histogram.frequencies[s], sym));
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
    pub fn new(histograms: IntegerHistograms<EP>, max_bits: usize) -> Self {
        let num_symbols = 1 << max_bits;
        let mut info = Vec::new();
        for histogram in &histograms.ctx_histograms {
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
