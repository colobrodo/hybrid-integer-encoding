// maximum number of bits for huffman code
pub const DEFAULT_MAX_HUFFMAN_BITS: usize = 8;
// maximum number of codes, directly derived from MAX_HUFFMAN_BITS
pub const DEFAULT_NUM_SYMBOLS: usize = 1 << DEFAULT_MAX_HUFFMAN_BITS;
// maximum number of contexts
// NOTE: the number of context are indexed by u8, so the maximum number of context is 256
pub const DEFAULT_NUM_CONTEXT: usize = 256;

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct HuffmanSymbolInfo {
    pub(crate) present: u8,
    pub(crate) n_bits: u8,
    pub(crate) bits: u16,
}

/// Parameters for the variable integer encoding scheme.
pub trait EncodeParams {
    // log2 of the number of explicit tokens (referred as k in the paper)
    const LOG2_NUM_EXPLICIT: u64;
    // number of less significant bit in token (referred as i in the paper)
    const MSB_IN_TOKEN: u64;
    // number of most significant bit in token (referred as j in the paper)
    const LSB_IN_TOKEN: u64;
}

#[derive(Default, Clone, Copy)]
pub struct DefaultEncodeParams;

impl EncodeParams for DefaultEncodeParams {
    const LOG2_NUM_EXPLICIT: u64 = 4;
    const MSB_IN_TOKEN: u64 = 2;
    const LSB_IN_TOKEN: u64 = 1;
}

// Variable integer encoding scheme that puts bits either in an entropy-coded
// symbol or as raw bits, depending on the specified configuration.
#[inline]
pub fn encode<EP: EncodeParams>(value: u64) -> (usize, usize, u64) {
    let split_token = 1 << EP::LOG2_NUM_EXPLICIT;
    if value < split_token {
        (value as usize, 0, 0)
    } else {
        let n = value.ilog2() as u64;
        let m = value & !(1 << n);
        let token = (1 << EP::LOG2_NUM_EXPLICIT)
            + ((n - EP::LOG2_NUM_EXPLICIT) << (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN))
            + ((m >> (n - EP::MSB_IN_TOKEN)) << EP::LSB_IN_TOKEN)
            + (m & ((1 << EP::LSB_IN_TOKEN) - 1));
        let n_bits = n - EP::MSB_IN_TOKEN - EP::LSB_IN_TOKEN;
        let bits = (value >> EP::LSB_IN_TOKEN) & ((1 << n_bits) - 1);
        (token as usize, n_bits as usize, bits)
    }
}

// For a given array of HuffmanSymbolInfo, where only the `present` and `n_bits`
// fields are set, fill up the `bits` field by building a Canonical Huffman code
// (https://en.wikipedia.org/wiki/Canonical_Huffman_code).
pub(crate) fn compute_symbol_bits(max_bits: usize, infos: &mut [HuffmanSymbolInfo]) {
    debug_assert!(infos.len() == 1 << max_bits);
    let mut symbols = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        if info.present == 0 {
            continue;
        }
        symbols.push((info.n_bits, i));
    }
    symbols.sort();
    let present_symbols = symbols.len();
    let mut x: usize = 0;
    for (s, sym) in symbols.iter().enumerate() {
        infos[sym.1].bits = u16::reverse_bits(x as u16)
            >> (u16::BITS as usize - max_bits)
            >> (max_bits as u8 - infos[sym.1].n_bits);
        x += 1;
        if s + 1 != present_symbols {
            x <<= symbols[s + 1].0 - sym.0;
        }
    }
}

/// Returns the number of bits needed to represent the length of each symbol in
/// the header.
pub const fn compute_symbol_len_bits(max_bits: u32) -> u32 {
    usize::BITS - (max_bits - 1).leading_zeros()
}
