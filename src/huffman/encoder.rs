use std::mem;

use dsi_bitstream::traits::{BitWrite, Endianness};

use crate::huffman::compute_symbol_bits;

use super::{
    common::{encode, DefaultEncodeParams, EncodeParams, HuffmanSymbolInfo},
    MAX_HUFFMAN_BITS, NUM_SYMBOLS, SYM_LEN_BITS,
};

use anyhow::Result;

pub struct HuffmanEncoder<EP: EncodeParams = DefaultEncodeParams> {
    info_: [HuffmanSymbolInfo; 1 << MAX_HUFFMAN_BITS],
    _marker: core::marker::PhantomData<EP>,
}

type Bag = (usize, Vec<u8>);

fn compute_histogram<EP: EncodeParams>(data: &[u32]) -> [usize; NUM_SYMBOLS] {
    let mut histogram = [0; NUM_SYMBOLS];
    for value in data {
        let (token, _, _) = encode::<EP>((*value).into());
        histogram[token] += 1;
    }
    histogram
}

// Compute the optimal number of bits for each symbol given the input
// distribution. Uses a (quadratic version) of the package-merge/coin-collector
// algorithm.
fn compute_symbol_num_bits(histogram: &[usize], infos: &mut [HuffmanSymbolInfo; NUM_SYMBOLS]) {
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
    let mut bags = vec![Vec::<Bag>::default(); MAX_HUFFMAN_BITS];
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
    for i in 0..(MAX_HUFFMAN_BITS - 1) {
        bags[i].sort();
        for j in (0..bags[i].len() - 1).step_by(2) {
            let nf = bags[i][j].0 + bags[i][j + 1].0;
            let mut nsym = mem::take(&mut bags[i][j].1);
            nsym.append(&mut bags[i][j + 1].1);
            bags[i + 1].push((nf, nsym));
        }
    }
    bags[MAX_HUFFMAN_BITS - 1].sort();

    // In the groups of symbols for the highest bit length we need to select the
    // last 2*num_symbols-2 groups, and assign to each symbol one bit of cost for
    // each of its occurrences in these groups.
    for i in 0..2 * nzsym - 2 {
        let b = &bags[MAX_HUFFMAN_BITS - 1][i];
        for &x in b.1.iter() {
            infos[x as usize].nbits += 1;
        }
    }
}

impl<EP: EncodeParams> HuffmanEncoder<EP> {
    pub fn new(data: &[u32]) -> Self {
        let histogram = compute_histogram::<EP>(data);
        let mut info = [HuffmanSymbolInfo::default(); NUM_SYMBOLS];
        compute_symbol_num_bits(&histogram, &mut info);
        compute_symbol_bits(&mut info);
        Self {
            info_: info,
            _marker: core::marker::PhantomData,
        }
    }

    /// Write the value into the bit stream, if successfull returns a tuple with the
    /// number of bits written for the symbol and for the trailing bits.
    pub fn write<E: Endianness>(
        &self,
        value: u32,
        writer: &mut impl BitWrite<E>,
    ) -> Result<(usize, usize)> {
        let (token, nbits, bits) = encode::<EP>(value as u64);
        debug_assert!(self.info_[token].present == 1, "Unknown value {value}");
        let nbits_per_token = self.info_[token].nbits as usize;
        writer.write_bits(self.info_[token].bits as u64, nbits_per_token)?;
        writer.write_bits(bits, nbits)?;
        Ok((nbits, nbits_per_token))
    }

    // Very simple encoding: number of symbols (16 bits) followed by, for each
    // symbol, 1 bit for presence/absence, and 4 bits for symbol length if present.
    // TODO: short encoding for empty ctxs, RLE for missing symbols.
    pub fn write_header<E: Endianness>(&self, writer: &mut impl BitWrite<E>) -> Result<()> {
        let mut ms = 0;
        for (i, info) in self.info_.iter().enumerate() {
            if info.present != 0 {
                ms = i;
            }
        }

        writer.write_bits(ms as u64, MAX_HUFFMAN_BITS)?;
        for info in self.info_.iter().take(ms + 1) {
            if info.present != 0 {
                writer.write_bits(1, 1)?;
                writer.write_bits(info.nbits as u64 - 1, SYM_LEN_BITS as usize)?;
            } else {
                writer.write_bits(0, 1)?;
            }
        }
        Ok(())
    }
}
