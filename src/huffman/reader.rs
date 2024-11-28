use dsi_bitstream::traits::{BitRead, Endianness};

use super::{
    compute_symbol_bits, compute_symbol_len_bits, EncodeParams, HuffmanSymbolInfo,
    DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_CONTEXT, DEFAULT_NUM_SYMBOLS,
};
use common_traits::*;

use anyhow::{anyhow, Result};

pub trait EntropyCoder {
    fn read_token(&mut self, context: usize) -> Result<usize>;

    fn read_bits(&mut self, n: usize) -> Result<u64>;

    #[inline(always)]
    fn read<EP: EncodeParams>(&mut self, context: usize) -> Result<usize> {
        let split_token = 1 << EP::LOG2_NUM_EXPLICIT;
        let mut token = self.read_token(context)?;
        if token < split_token {
            return Ok(token);
        }
        let nbits = EP::LOG2_NUM_EXPLICIT - (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN)
            + ((token - split_token) as u32 >> (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN));
        let low = token & ((1 << EP::LSB_IN_TOKEN) - 1);
        token >>= EP::LSB_IN_TOKEN;
        let bits = self.read_bits(nbits as usize)? as usize;
        let ret = (((((1 << EP::MSB_IN_TOKEN) | (token & ((1 << EP::MSB_IN_TOKEN) - 1)))
            << nbits)
            | bits)
            << EP::LSB_IN_TOKEN)
            | low;
        Ok(ret)
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct HuffmanDecoderInfo {
    nbits: u8,
    symbol: u8,
}

pub struct HuffmanReader<
    E: Endianness,
    R: BitRead<E>,
    const NUM_CONTEXT: usize = DEFAULT_NUM_CONTEXT,
    const MAX_BITS: usize = DEFAULT_MAX_HUFFMAN_BITS,
    const NUM_SYMBOLS: usize = DEFAULT_NUM_SYMBOLS,
> {
    reader: R,
    info_: [[HuffmanDecoderInfo; NUM_SYMBOLS]; NUM_CONTEXT],
    _marker: core::marker::PhantomData<E>,
}

fn decode_symbol_num_bits<
    const MAX_BITS: usize,
    const NUM_SYMBOLS: usize,
    E: Endianness,
    R: BitRead<E>,
>(
    infos: &mut [HuffmanSymbolInfo; NUM_SYMBOLS],
    reader: &mut R,
) -> Result<()> {
    // number of bits needed to represent the length of each symbol in the header
    let symbol_len_bits: u32 = compute_symbol_len_bits(MAX_BITS as u32);
    let ms = reader.read_bits(MAX_BITS)? as usize;
    for info in infos.iter_mut().take(ms + 1) {
        info.present = reader.read_bits(1)? as u8;
        if info.present != 0 {
            info.nbits = reader.read_bits(symbol_len_bits as usize)? as u8 + 1;
        }
    }
    for info in infos.iter_mut().skip(ms + 1) {
        info.present = 0;
    }
    Ok(())
}

/// Computes the lookup table from bitstream bits to decoded symbol for the
/// decoder.
/// The computed table is stored in the `infos` array.
fn compute_decoder_table<const MAX_BITS: usize, const NUM_SYMBOLS: usize>(
    sym_infos: &[HuffmanSymbolInfo; NUM_SYMBOLS],
    infos: &mut [HuffmanDecoderInfo; NUM_SYMBOLS],
) -> Result<()> {
    let cnt = sym_infos.iter().filter(|sym| sym.present != 0).count();
    let s = sym_infos
        .iter()
        .enumerate()
        .filter(|(_, sym)| sym.present != 0)
        .last()
        .map_or(0, |(i, _)| i);
    if cnt <= 1 {
        for (info, sym_info) in infos.iter_mut().zip(sym_infos) {
            info.nbits = sym_info.nbits;
            info.symbol = s as u8;
        }
        return Ok(());
    }

    for i in 0..1 << MAX_BITS {
        let mut s = NUM_SYMBOLS;
        for (sym, sym_info) in sym_infos.iter().enumerate() {
            if sym_info.present == 0 {
                continue;
            }
            if (i & ((1 << sym_info.nbits) - 1)) as u16 == sym_info.bits {
                s = sym;
                break;
            }
        }
        if s == NUM_SYMBOLS {
            return Err(anyhow!("Invalid table"));
        }
        infos[i as usize].nbits = sym_infos[s].nbits;
        infos[i as usize].symbol = s as u8;
    }
    Ok(())
}

impl<
        E: Endianness,
        R: BitRead<E>,
        const NUM_CONTEXT: usize,
        const MAX_BITS: usize,
        const NUM_SYMBOLS: usize,
    > HuffmanReader<E, R, NUM_CONTEXT, MAX_BITS, NUM_SYMBOLS>
{
    pub fn new(reader: R) -> Result<Self> {
        let mut reader = reader;
        let mut info = [[HuffmanDecoderInfo::default(); NUM_SYMBOLS]; NUM_CONTEXT];
        for ctx in 0..NUM_CONTEXT {
            let mut symbol_info = [HuffmanSymbolInfo::default(); NUM_SYMBOLS];
            decode_symbol_num_bits::<MAX_BITS, NUM_SYMBOLS, _, _>(&mut symbol_info, &mut reader)?;
            compute_symbol_bits::<MAX_BITS, NUM_SYMBOLS>(&mut symbol_info);
            compute_decoder_table::<MAX_BITS, NUM_SYMBOLS>(&symbol_info, &mut info[ctx])?;
        }
        Ok(Self {
            reader,
            info_: info,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<
        E: Endianness,
        R: BitRead<E>,
        const NUM_CONTEXT: usize,
        const MAX_BITS: usize,
        const NUM_SYMBOLS: usize,
    > EntropyCoder for HuffmanReader<E, R, NUM_CONTEXT, MAX_BITS, NUM_SYMBOLS>
{
    fn read_token(&mut self, context: usize) -> Result<usize> {
        let bits: u64 = self.reader.peek_bits(MAX_BITS)?.cast();
        let info = self.info_[context][bits as usize];
        self.reader
            .skip_bits_after_table_lookup(info.nbits as usize);
        Ok(info.symbol as usize)
    }

    fn read_bits(&mut self, n: usize) -> Result<u64> {
        let bits = self.reader.read_bits(n)?;
        Ok(bits)
    }
}
