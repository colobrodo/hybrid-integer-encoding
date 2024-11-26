pub const MAX_HUFFMAN_BITS: usize = 10;
pub const NUM_SYMBOLS: usize = 1 << MAX_HUFFMAN_BITS;
// number of bits needed to represent the length of each symbol in the header
pub const SYM_LEN_BITS: u32 = usize::BITS - (MAX_HUFFMAN_BITS - 1).leading_zeros();

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct HuffmanSymbolInfo {
    pub(crate) present: u8,
    pub(crate) nbits: u8,
    pub(crate) bits: u16,
}

pub trait EncodeParams {
    // log2 of the number of explicit tokens (referred as k in the paper)
    const LOG2_NUM_EXPLICIT: u32;
    // number of less significant bit in token (referred as i in the paper)
    const MSB_IN_TOKEN: u32;
    // number of most significant bit in token (referred as j in the paper)
    const LSB_IN_TOKEN: u32;
}

pub struct DefaultEncodeParams;

impl EncodeParams for DefaultEncodeParams {
    const LOG2_NUM_EXPLICIT: u32 = 4;
    const MSB_IN_TOKEN: u32 = 2;
    const LSB_IN_TOKEN: u32 = 1;
}

// Variable integer encoding scheme that puts bits either in an entropy-coded
// symbol or as raw bits, depending on the specified configuration.
pub(crate) fn encode<EP: EncodeParams>(value: u64) -> (usize, usize, u64) {
    let split_token: u32 = 1 << EP::LOG2_NUM_EXPLICIT;
    if value < split_token as u64 {
        (value as usize, 0, 0)
    } else {
        let n = usize::BITS - 1 - value.leading_zeros();
        let m = (value - (1 << n)) as u32;
        let token = split_token
            + ((n - EP::LOG2_NUM_EXPLICIT) << (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN))
            + ((m >> (n - EP::MSB_IN_TOKEN)) << EP::LSB_IN_TOKEN)
            + (m & ((1 << EP::LSB_IN_TOKEN) - 1));
        let nbits = n - EP::MSB_IN_TOKEN - EP::LSB_IN_TOKEN;
        let bits = (value >> EP::LSB_IN_TOKEN) & ((1 << nbits) - 1);
        (token as usize, nbits as usize, bits)
    }
}

// For a given array of HuffmanSymbolInfo, where only the `present` and `nbits`
// fields are set, fill up the `bits` field by building a Canonical Huffman code
// (https://en.wikipedia.org/wiki/Canonical_Huffman_code).
pub(crate) fn compute_symbol_bits(infos: &mut [HuffmanSymbolInfo; NUM_SYMBOLS]) {
    let mut syms = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        if info.present == 0 {
            continue;
        }
        syms.push((info.nbits, i));
    }
    syms.sort();
    let present_symbols = syms.len();
    let mut x: usize = 0;
    for (s, sym) in syms.iter().enumerate() {
        infos[sym.1].bits = u16::reverse_bits(x as u16)
            >> (u16::BITS as usize - MAX_HUFFMAN_BITS)
            >> (MAX_HUFFMAN_BITS as u8 - infos[sym.1].nbits);
        x += 1;
        if s + 1 != present_symbols {
            x <<= syms[s + 1].0 - sym.0;
        }
    }
}
