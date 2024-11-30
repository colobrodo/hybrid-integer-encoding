use std::mem;

use dsi_bitstream::traits::{BitWrite, Endianness};

use crate::huffman::compute_symbol_bits;

use super::{
    common::{
        compute_symbol_len_bits, encode, DefaultEncodeParams, EncodeParams, HuffmanSymbolInfo,
    },
    DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_CONTEXT, DEFAULT_NUM_SYMBOLS,
};

use anyhow::Result;

pub struct HuffmanEncoder<
    EP: EncodeParams = DefaultEncodeParams,
    const NUM_CONTEXT: usize = DEFAULT_NUM_CONTEXT,
    const MAX_BITS: usize = DEFAULT_MAX_HUFFMAN_BITS,
    const NUM_SYMBOLS: usize = DEFAULT_NUM_SYMBOLS,
> {
    _marker: core::marker::PhantomData<EP>,
    info_: [[HuffmanSymbolInfo; NUM_SYMBOLS]; NUM_CONTEXT],
}

/// Data structure to hold the integer values and their contexts.
pub struct IntegerData {
    values: Vec<u32>,
    contexts: Vec<u8>,
}

impl Default for IntegerData {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegerData {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            contexts: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn add(&mut self, context: u8, value: u32) {
        self.values.push(value);
        self.contexts.push(context);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u8, &u32)> + '_ {
        self.contexts.iter().zip(self.values.iter())
    }

    /// Compute the histogram of the each token frequency for each context.
    fn compute_histograms<EP: EncodeParams, const NUM_SYMBOLS: usize>(
        &self,
    ) -> Vec<[usize; NUM_SYMBOLS]> {
        let mut histograms = Vec::new();
        for (&value, &ctx) in self.values.iter().zip(self.contexts.iter()) {
            let (token, _, _) = encode::<EP>(value.into());
            if histograms.len() <= ctx as usize {
                histograms.resize(ctx as usize + 1, [0; NUM_SYMBOLS]);
            }
            histograms[ctx as usize][token] += 1;
        }
        histograms
    }

    pub fn context(&self, i: usize) -> u8 {
        self.contexts[i]
    }

    pub fn value(&self, i: usize) -> u32 {
        self.values[i]
    }
}

type Bag = (usize, Vec<u8>);

// Compute the optimal number of bits for each symbol given the input
// distribution. Uses a (quadratic version) of the package-merge/coin-collector
// algorithm.
fn compute_symbol_num_bits<const MAX_BITS: usize, const NUM_SYMBOLS: usize>(
    histogram: &[usize],
    infos: &mut [HuffmanSymbolInfo],
) {
    assert!(infos.len() == NUM_SYMBOLS);
    // Mark the present/missing symbols.
    let mut nzsym = 0;
    for (i, freq) in histogram.iter().enumerate() {
        if *freq == 0 {
            continue;
        }
        infos[i].present = 1;
        nzsym += 1;
    }
    if nzsym <= 1 {
        for info in infos.iter_mut() {
            if info.present != 0 {
                info.nbits = 1;
            }
        }
        return;
    }

    // Create a list of symbols for any given cost.
    let mut bags = vec![Vec::<Bag>::default(); MAX_BITS];
    for bag in bags.iter_mut() {
        for s in 0..NUM_SYMBOLS {
            if infos[s].present == 0 {
                continue;
            }
            let sym = vec![s as u8];
            bag.push((histogram[s], sym));
        }
    }

    // Pair up symbols (or groups of symbols) of a given bit-length to create
    // symbols of the following bit-length, creating pairs by merging (groups of)
    // symbols consecutively in increasing order of cost.
    for i in 0..(MAX_BITS - 1) {
        bags[i].sort();
        for j in (0..bags[i].len() - 1).step_by(2) {
            let nf = bags[i][j].0 + bags[i][j + 1].0;
            let mut nsym = mem::take(&mut bags[i][j].1);
            nsym.append(&mut bags[i][j + 1].1);
            bags[i + 1].push((nf, nsym));
        }
    }
    bags[MAX_BITS - 1].sort();

    // In the groups of symbols for the highest bit length we need to select the
    // last 2*num_symbols-2 groups, and assign to each symbol one bit of cost for
    // each of its occurrences in these groups.
    for i in 0..2 * nzsym - 2 {
        let b = &bags[MAX_BITS - 1][i];
        for &x in b.1.iter() {
            infos[x as usize].nbits += 1;
        }
    }
}

impl<
        EP: EncodeParams,
        const NUM_CONTEXT: usize,
        const MAX_BITS: usize,
        const NUM_SYMBOLS: usize,
    > HuffmanEncoder<EP, NUM_CONTEXT, MAX_BITS, NUM_SYMBOLS>
{
    pub fn new(data: &IntegerData) -> Self {
        assert!(
            NUM_SYMBOLS <= 1 << MAX_BITS,
            "NUM_SYMBOLS must be less than or equal to 2^MAX_BITS"
        );
        let histograms = data.compute_histograms::<EP, NUM_SYMBOLS>();
        let mut info = [[HuffmanSymbolInfo::default(); NUM_SYMBOLS]; NUM_CONTEXT];
        for (ctx, histogram) in histograms.iter().enumerate() {
            compute_symbol_num_bits::<MAX_BITS, NUM_SYMBOLS>(histogram, &mut info[ctx]);
            compute_symbol_bits::<MAX_BITS, NUM_SYMBOLS>(&mut info[ctx]);
        }
        Self {
            info_: info,
            _marker: core::marker::PhantomData,
        }
    }

    /// Write the value into the bit stream, if successfull returns a tuple with the
    /// number of bits written for the symbol and for the trailing bits.
    pub fn write<E: Endianness>(
        &self,
        ctx: u8,
        value: u32,
        writer: &mut impl BitWrite<E>,
    ) -> Result<(usize, usize)> {
        let (token, nbits, bits) = encode::<EP>(value as u64);
        debug_assert!(
            self.info_[ctx as usize][token].present == 1,
            "Unknown value {value}"
        );
        let nbits_per_token = self.info_[ctx as usize][token].nbits as usize;
        writer.write_bits(self.info_[ctx as usize][token].bits as u64, nbits_per_token)?;
        writer.write_bits(bits, nbits)?;
        Ok((nbits, nbits_per_token))
    }

    // Very simple encoding: number of symbols (16 bits) followed by, for each
    // symbol, 1 bit for presence/absence, and 4 bits for symbol length if present.
    // TODO: short  encoding for empty ctxs, RLE for missing symbols.
    pub fn write_header<E: Endianness>(&self, writer: &mut impl BitWrite<E>) -> Result<()> {
        // number of bits needed to represent the length of each symbol in the header
        let symbol_len_bits = compute_symbol_len_bits(MAX_BITS as u32);
        for info in self.info_.iter() {
            let mut ms = 0;
            for (i, sym_info) in info.iter().enumerate() {
                if sym_info.present != 0 {
                    ms = i;
                }
            }

            writer.write_bits(ms as u64, MAX_BITS)?;
            for sym_info in info.iter().take(ms + 1) {
                if sym_info.present != 0 {
                    writer.write_bits(1, 1)?;
                    writer.write_bits(sym_info.nbits as u64 - 1, symbol_len_bits as usize)?;
                } else {
                    writer.write_bits(0, 1)?;
                }
            }
        }
        Ok(())
    }
}
