use std::{error::Error, mem};

use dsi_bitstream::traits::{BitWrite, Endianness};

// TODO: remove pub from struct and fields: only for debugging
#[derive(Clone, Copy, Default, Debug)]
pub struct HuffmanSymbolInfo {
    pub present: u8,
    pub nbits: u8,
    pub bits: u8,
}

const MAX_HUFFMAN_BITS: usize = 8;
const NUM_SYMBOLS: usize = 1 << MAX_HUFFMAN_BITS;

// Variable integer encoding scheme that puts bits either in an entropy-coded
// symbol or as raw bits, depending on the specified configuration.
// log2 of the number of explicit tokens (referred as k in the paper)
const LOG2_NUM_EXPLICIT: u32 = 4;
// number of less significant bit in token (referred as i in the paper)
const MSB_IN_TOKEN: u32 = 2;
// number of most significant bit in token (referred as j in the paper)
const LSB_IN_TOKEN: u32 = 1;

fn encode(value: u64) -> (usize, usize, u64) {
    let split_token = 1 << LOG2_NUM_EXPLICIT;
    if value < split_token {
        (value as usize, 0, 0)
    } else {
        let n = usize::BITS - 1 - value.leading_zeros();
        let m = value - (1 << n);
        let token = split_token
            + ((n - LOG2_NUM_EXPLICIT) << (MSB_IN_TOKEN + LSB_IN_TOKEN)) as u64
            + ((m >> (n - MSB_IN_TOKEN)) << LSB_IN_TOKEN)
            + (m & ((1 << LSB_IN_TOKEN) - 1));
        let nbits = n - MSB_IN_TOKEN - LSB_IN_TOKEN;
        let bits = (value >> LSB_IN_TOKEN) & ((1 << nbits) - 1);
        (token as usize, nbits as usize, bits)
    }
}

// TODO: remove pub from info_ field used only for encoding
pub struct HuffmanEncoder {
    pub info_: [HuffmanSymbolInfo; 1 << MAX_HUFFMAN_BITS],
}

fn compute_histogram(data: &[u64]) -> [usize; NUM_SYMBOLS] {
    let mut histogram = [0; NUM_SYMBOLS];
    for value in data {
        let (token, _, _) = encode(*value);
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
    let mut bags = vec![Vec::<(usize, Vec<u8>)>::default(); MAX_HUFFMAN_BITS];
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

// For a given array of HuffmanSymbolInfo, where only the `present` and `nbits`
// fields are set, fill up the `bits` field by building a Canonical Huffman code
// (https://en.wikipedia.org/wiki/Canonical_Huffman_code).
fn compute_symbol_bits(infos: &mut [HuffmanSymbolInfo; NUM_SYMBOLS]) -> bool {
    let mut syms = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        if info.present == 0 {
            continue;
        }
        syms.push((info.nbits, i as u8));
    }
    syms.sort();
    let present_symbols = syms.len();
    let mut x: u8 = 0;
    for (s, sym) in syms.iter().enumerate() {
        infos[sym.1 as usize].bits =
            u8::reverse_bits(x) >> (MAX_HUFFMAN_BITS as u8 - infos[sym.1 as usize].nbits);
        x += 1;
        if s + 1 != present_symbols {
            x <<= syms[s + 1].0 - sym.0;
        }
    }
    true
}

impl HuffmanEncoder {
    pub fn new(data: &[u64]) -> Self {
        let histogram = compute_histogram(data);
        let mut info = [HuffmanSymbolInfo::default(); NUM_SYMBOLS];
        compute_symbol_num_bits(&histogram, &mut info);
        debug_assert!(compute_symbol_bits(&mut info));
        Self { info_: info }
    }

    /// Write the value into the bit stream, if successfull returns a tuple with the
    /// number of bits written for the symbol and for the trailing bits.
    pub fn write<E: Endianness>(
        &self,
        value: u64,
        writer: &mut impl BitWrite<E>,
    ) -> Result<(usize, usize), Box<dyn Error>> {
        let (token, nbits, bits) = encode(value);
        debug_assert!(self.info_[token].present == 1, "Unknown value {value}");
        let nbits_per_token = self.info_[token].nbits as usize;
        writer.write_bits(self.info_[token].bits as u64, nbits_per_token)?;
        writer.write_bits(bits, nbits)?;
        Ok((nbits, nbits_per_token))
    }

    // Very simple encoding: number of symbols (8 bits) followed by, for each
    // symbol, 1 bit for presence/absence, and 3 bits for symbol length if present.
    // TODO: short encoding for empty ctxs, RLE for missing symbols.
    pub fn write_header<E: Endianness>(
        &self,
        writer: &mut impl BitWrite<E>,
    ) -> Result<(), Box<dyn Error>> {
        let mut ms = 0;
        for (i, info) in self.info_.iter().enumerate() {
            if info.present != 0 {
                ms = i;
            }
        }

        writer.write_bits(ms as u64, 8)?;
        for info in self.info_.iter().take(ms + 1) {
            if info.present != 0 {
                writer.write_bits(1, 1)?;
                writer.write_bits(info.nbits as u64 - 1, 3)?;
            } else {
                writer.write_bits(0, 1)?;
            }
        }
        Ok(())
    }
}
