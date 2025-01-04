use std::rc::Rc;

use dsi_bitstream::traits::{BitRead, BitSeek, Endianness};

use super::{compute_symbol_bits, compute_symbol_len_bits, EncodeParams, HuffmanSymbolInfo};
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

pub struct HuffmanReader<E: Endianness, R: BitRead<E>> {
    reader: R,
    max_bits: usize,
    info_: Rc<[Box<[HuffmanDecoderInfo]>]>,
    _marker: core::marker::PhantomData<E>,
}

#[derive(Clone)]
pub struct HuffmanTable {
    max_bits: usize,
    info_: Rc<[Box<[HuffmanDecoderInfo]>]>,
}

fn decode_symbol_num_bits<E: Endianness, R: BitRead<E>>(
    max_bits: usize,
    infos: &mut [HuffmanSymbolInfo],
    reader: &mut R,
) -> Result<()> {
    assert!(infos.len() == 1 << max_bits);
    // number of bits needed to represent the length of each symbol in the header
    let symbol_len_bits: u32 = compute_symbol_len_bits(max_bits as u32);
    let ms = reader.read_bits(max_bits)? as usize;
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
fn compute_decoder_table(
    max_bits: usize,
    sym_infos: &[HuffmanSymbolInfo],
    infos: &mut [HuffmanDecoderInfo],
) -> Result<()> {
    let num_symbols = 1 << max_bits;
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

    for (i, info) in infos.iter_mut().enumerate() {
        let mut s = num_symbols;
        for (sym, sym_info) in sym_infos.iter().enumerate() {
            if sym_info.present == 0 {
                continue;
            }
            if (i & ((1 << sym_info.nbits) - 1)) as u16 == sym_info.bits {
                s = sym;
                break;
            }
        }
        if s == num_symbols {
            return Err(anyhow!("Invalid table"));
        }
        info.nbits = sym_infos[s].nbits;
        info.symbol = s as u8;
    }
    Ok(())
}

impl HuffmanTable {
    fn new<R: BitRead<E>, E: Endianness>(
        reader: &mut R,
        max_bits: usize,
        num_contexts: usize,
    ) -> Result<Self> {
        let num_symbols = 1 << max_bits;
        let mut info = Rc::new_uninit_slice(num_contexts);
        let data = Rc::get_mut(&mut info).unwrap();
        //let mut info = Vec::with_capacity(num_contexts);
        for i in 0..num_contexts {
            let mut symbol_info = vec![HuffmanSymbolInfo::default(); num_symbols];
            decode_symbol_num_bits(max_bits, &mut symbol_info, reader)?;
            compute_symbol_bits(max_bits, &mut symbol_info);
            let mut ctx_info = vec![HuffmanDecoderInfo::default(); num_symbols];
            compute_decoder_table(max_bits, &symbol_info, &mut ctx_info)?;
            data[i].write(ctx_info.into_boxed_slice());
        }
        let info = unsafe { info.assume_init() };
        Ok(Self {
            max_bits,
            info_: info,
        })
    }
}

impl<E: Endianness, R: BitRead<E>> HuffmanReader<E, R> {
    /// Constructs a `HuffmanReader` by consuming the provided `BitRead` implementation.
    /// Reads the Huffman table from the start and fully owns the bitstream.
    pub fn from_bitreader(reader: R, max_bits: usize, num_contexts: usize) -> Result<Self> {
        let mut reader = reader;
        let table_decoder = HuffmanTable::new(&mut reader, max_bits, num_contexts)?;
        Ok(HuffmanReader::new(table_decoder, reader))
    }

    /// Decodes the Huffman table from the provided BitReader without consuming it.
    /// Returns a HuffmanReaderLoader for further operations while allowing reuse of the BitReader.
    pub fn decode_table(
        reader: &mut R,
        max_bits: usize,
        num_contexts: usize,
    ) -> Result<HuffmanTable> {
        HuffmanTable::new(reader, max_bits, num_contexts)
    }

    pub fn new(table: HuffmanTable, reader: R) -> Self {
        Self {
            reader,
            max_bits: table.max_bits,
            info_: table.info_,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Endianness, R: BitRead<E>> EntropyCoder for HuffmanReader<E, R> {
    fn read_token(&mut self, context: usize) -> Result<usize> {
        let bits: u64 = self.reader.peek_bits(self.max_bits)?.cast();
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

impl<E: Endianness, R: BitRead<E> + BitSeek> BitSeek for HuffmanReader<E, R> {
    type Error = <R as BitSeek>::Error;

    fn bit_pos(&mut self) -> Result<u64, Self::Error> {
        self.reader.bit_pos()
    }

    fn set_bit_pos(&mut self, bit_pos: u64) -> Result<(), Self::Error> {
        self.reader.set_bit_pos(bit_pos)
    }
}
