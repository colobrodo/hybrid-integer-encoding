pub const MAX_HUFFMAN_BITS: usize = 8;
pub const NUM_SYMBOLS: usize = 1 << MAX_HUFFMAN_BITS;

// TODO: remove pub from struct and fields: only for debugging
#[derive(Clone, Copy, Default, Debug)]
pub struct HuffmanSymbolInfo {
    pub present: u8,
    pub nbits: u8,
    pub bits: u8,
}

pub trait EncodeParams {
    // log2 of the number of explicit tokens (referred as k in the paper)
    const LOG2_NUM_EXPLICIT: u32;
    // number of less significant bit in token (referred as i in the paper)
    const MSB_IN_TOKEN: u32;
    // number of most significant bit in token (referred as j in the paper)
    const LSB_IN_TOKEN: u32;
}

pub struct DefaultEncodeParams {}

impl EncodeParams for DefaultEncodeParams {
    const LOG2_NUM_EXPLICIT: u32 = 4;
    const MSB_IN_TOKEN: u32 = 2;
    const LSB_IN_TOKEN: u32 = 1;
}

// Variable integer encoding scheme that puts bits either in an entropy-coded
// symbol or as raw bits, depending on the specified configuration.
pub(crate) fn encode<EP: EncodeParams>(value: u64) -> (usize, usize, u64) {
    let split_token = 1 << EP::LOG2_NUM_EXPLICIT;
    if value < split_token {
        (value as usize, 0, 0)
    } else {
        let n = usize::BITS - 1 - value.leading_zeros();
        let m = value - (1 << n);
        let token = split_token
            + ((n - EP::LOG2_NUM_EXPLICIT) << (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN)) as u64
            + ((m >> (n - EP::MSB_IN_TOKEN)) << EP::LSB_IN_TOKEN)
            + (m & ((1 << EP::LSB_IN_TOKEN) - 1));
        let nbits = n - EP::MSB_IN_TOKEN - EP::LSB_IN_TOKEN;
        let bits = (value >> EP::LSB_IN_TOKEN) & ((1 << nbits) - 1);
        (token as usize, nbits as usize, bits)
    }
}
