use dsi_bitstream::traits::{BitRead, Endianness};

use super::{compute_symbol_bits, EncodeParams, HuffmanSymbolInfo, MAX_HUFFMAN_BITS, NUM_SYMBOLS};
use common_traits::*;

use anyhow::{anyhow, Result};

pub trait EntropyCoder {
    fn read_token(&mut self) -> Result<u8>;

    fn read_bits(&mut self, n: usize) -> Result<u64>;

    fn read<EP: EncodeParams>(&mut self) -> Result<u8> {
        let split_token = 1 << EP::LOG2_NUM_EXPLICIT;
        let mut token = self.read_token()?;
        if token < split_token {
            return Ok(token);
        }
        let nbits = EP::LOG2_NUM_EXPLICIT - (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN)
            + ((token - split_token) as u32 >> (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN));
        let low = token & ((1 << EP::LSB_IN_TOKEN) - 1);
        token >>= EP::LSB_IN_TOKEN;
        let bits = self.read_bits(nbits as usize)? as u8;
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

pub struct HuffmanReader<E: Endianness, R: BitRead<E>> {
    reader: R,
    info_: [HuffmanDecoderInfo; 1 << MAX_HUFFMAN_BITS],
    _marker: core::marker::PhantomData<E>,
}

fn decode_symbol_num_bits<E: Endianness, R: BitRead<E>>(
    infos: &mut [HuffmanSymbolInfo; NUM_SYMBOLS],
    reader: &mut R,
) -> Result<()> {
    let ms = reader.read_bits(8)? as usize;
    for info in infos.iter_mut().take(ms + 1) {
        info.present = reader.read_bits(1)? as u8;
        if info.present != 0 {
            info.nbits = reader.read_bits(3)? as u8 + 1;
        }
    }
    for info in infos.iter_mut().skip(ms + 1) {
        info.present = 0;
    }
    Ok(())
}

// Computes the lookup table from bitstream bits to decoded symbol for the
// decoder.
fn compute_decoder_table(
    sym_infos: &[HuffmanSymbolInfo; NUM_SYMBOLS],
) -> Result<[HuffmanDecoderInfo; NUM_SYMBOLS]> {
    let cnt = sym_infos.iter().filter(|sym| sym.present != 0).count();
    let s = sym_infos
        .iter()
        .enumerate()
        .filter(|(_, sym)| sym.present != 0)
        .last()
        .map_or(0, |(i, _)| i);
    let mut decoder_infos = [HuffmanDecoderInfo::default(); NUM_SYMBOLS];
    if cnt <= 1 {
        for (info, sym_info) in decoder_infos.iter_mut().zip(sym_infos) {
            info.nbits = sym_info.nbits;
            info.symbol = s as u8;
        }
        return Ok(decoder_infos);
    }

    for i in 0..1 << MAX_HUFFMAN_BITS {
        let mut s = NUM_SYMBOLS;
        for (sym, sym_info) in sym_infos.iter().enumerate() {
            if sym_info.present == 0 {
                continue;
            }
            if (i & ((1 << sym_info.nbits) - 1)) as u8 == sym_info.bits {
                s = sym;
                break;
            }
        }
        if s == NUM_SYMBOLS {
            return Err(anyhow!("Invalid table"));
        }
        decoder_infos[i as usize].nbits = sym_infos[s].nbits;
        decoder_infos[i as usize].symbol = s as u8;
    }
    Ok(decoder_infos)
}

impl<E: Endianness, R: BitRead<E>> HuffmanReader<E, R> {
    pub fn new(reader: R) -> Result<Self> {
        let mut reader = reader;
        let mut symbol_info = [HuffmanSymbolInfo::default(); NUM_SYMBOLS];
        decode_symbol_num_bits(&mut symbol_info, &mut reader)?;
        compute_symbol_bits(&mut symbol_info);
        let info = compute_decoder_table(&symbol_info)?;
        Ok(Self {
            reader,
            info_: info,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<E: Endianness, R: BitRead<E>> EntropyCoder for HuffmanReader<E, R> {
    fn read_token(&mut self) -> Result<u8> {
        let bits: u64 = self.reader.peek_bits(MAX_HUFFMAN_BITS)?.cast();
        let info = self.info_[bits as usize];
        self.reader
            .skip_bits_after_table_lookup(info.nbits as usize);
        Ok(info.symbol)
    }

    fn read_bits(&mut self, n: usize) -> Result<u64> {
        let bits = self.reader.read_bits(n)?;
        Ok(bits)
    }
}
