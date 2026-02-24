use std::rc::Rc;

use dsi_bitstream::traits::{BitRead, BitSeek, LE};

use super::{compute_symbol_bits, compute_symbol_len_bits, EncodeParams, HuffmanSymbolInfo};
use common_traits::*;

use anyhow::{anyhow, Result};

/// Structure holding information about the Huffman lookup table.
/// Contains the number of bits for the Huffman code, the symbol, and pre-computed
/// trailing bits count for faster decoding.
#[derive(Clone, Copy, Default, Debug)]
struct HuffmanDecoderInfo {
    /// Number of bits in the Huffman code for this symbol
    nbits: u8,
    /// Pre-computed number of trailing bits to read after the token.
    /// This is 0 for explicit tokens (symbol < split_token).
    trailing_bits: u8,
    /// The decoded symbol (token)
    symbol: u16,
}

/// Struct representing a Huffman decoder.
/// It contains a reader and a Huffman table for decoding.
pub struct HuffmanDecoder<R: BitRead<LE>, EP: EncodeParams> {
    table: HuffmanTable<EP>,
    reader: R,
}

/// The Huffman decoder lookup table computed from the header of the file and the maximum
/// number of bits per token known in advance. The table pre-computes the number of trailing
/// bits for each symbol based on the encoding parameters.
pub struct HuffmanTable<EP: EncodeParams> {
    max_bits: usize,
    info_: Rc<[Box<[HuffmanDecoderInfo]>]>,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams> Clone for HuffmanTable<EP> {
    fn clone(&self) -> Self {
        Self {
            max_bits: self.max_bits,
            info_: self.info_.clone(),
            _marker: core::marker::PhantomData,
        }
    }
}

/// Decodes the number of bits for symbols from the bitstream.
/// Updates the provided `infos` array with the decoded information.
fn decode_symbol_num_bits<R: BitRead<LE>>(
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
            info.n_bits = reader.read_bits(symbol_len_bits as usize)? as u8 + 1;
        }
    }
    for info in infos.iter_mut().skip(ms + 1) {
        info.present = 0;
    }
    Ok(())
}

/// Computes the number of trailing bits for a given symbol based on the encoding parameters.
/// Returns 0 for explicit tokens (symbols less than split_token).
#[inline]
fn compute_trailing_bits<EP: EncodeParams>(symbol: usize) -> u8 {
    let split_token = 1usize << EP::LOG2_NUM_EXPLICIT;
    if symbol < split_token {
        0
    } else {
        let nbits = EP::LOG2_NUM_EXPLICIT - (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN)
            + ((symbol - split_token) as u64 >> (EP::MSB_IN_TOKEN + EP::LSB_IN_TOKEN));
        nbits as u8
    }
}

/// Computes the lookup table from bitstream bits to decoded symbol for the
/// decoder. The computed table is stored in the `infos` array.
/// Also pre-computes the trailing_bits for each symbol based on EncodeParams.
fn compute_decoder_table<EP: EncodeParams>(
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
        .next_back()
        .map_or(0, |(i, _)| i);
    if cnt <= 1 {
        let trailing_bits = compute_trailing_bits::<EP>(s);
        for info in infos.iter_mut() {
            info.nbits = sym_infos[s].n_bits;
            info.symbol = s as u16;
            info.trailing_bits = trailing_bits;
        }
        return Ok(());
    }

    for (i, info) in infos.iter_mut().enumerate() {
        let mut s = num_symbols;
        for (sym, sym_info) in sym_infos.iter().enumerate() {
            if sym_info.present == 0 {
                continue;
            }
            // NOTE: in case we want to support both big endian and little endian we should
            // change this condition to create the table looking at the low/top part of the
            // next 'max_bits' bits
            if (i & ((1 << sym_info.n_bits) - 1)) as u16 == sym_info.bits {
                s = sym;
                break;
            }
        }
        if s == num_symbols {
            return Err(anyhow!("Invalid table: cannot found a symbol for {}", i));
        }
        assert!(sym_infos[s].n_bits > 0);
        info.nbits = sym_infos[s].n_bits;
        info.symbol = s as u16;
        info.trailing_bits = compute_trailing_bits::<EP>(s);
    }
    Ok(())
}

impl<EP: EncodeParams> HuffmanTable<EP> {
    /// Creates a new HuffmanTable from the provided BitReader.
    /// Reads the symbols and constructs the table based on the maximum bits and contexts.
    /// Pre-computes trailing_bits for each symbol based on the EncodeParams.
    fn new<R: BitRead<LE>>(reader: &mut R, max_bits: usize, num_contexts: usize) -> Result<Self> {
        let num_symbols = 1 << max_bits;
        let mut info = Rc::new_uninit_slice(num_contexts);
        let data = Rc::get_mut(&mut info).unwrap();
        for ctx_cell in data.iter_mut() {
            let mut symbol_info = vec![HuffmanSymbolInfo::default(); num_symbols];
            decode_symbol_num_bits(max_bits, &mut symbol_info, reader)?;
            compute_symbol_bits(max_bits, &mut symbol_info);
            let mut ctx_info = vec![HuffmanDecoderInfo::default(); num_symbols];
            compute_decoder_table::<EP>(max_bits, &symbol_info, &mut ctx_info)?;
            ctx_cell.write(ctx_info.into_boxed_slice());
        }
        let info = unsafe { info.assume_init() };
        Ok(Self {
            max_bits,
            info_: info,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<R: BitRead<LE>, EP: EncodeParams> HuffmanDecoder<R, EP> {
    /// Constructs a `HuffmanDecoder` by consuming the provided `BitRead` implementation.
    /// Reads the Huffman table from the start and fully owns the bitstream.
    pub fn from_bitreader(reader: R, max_bits: usize, num_contexts: usize) -> Result<Self> {
        let mut reader = reader;
        let table = HuffmanTable::new(&mut reader, max_bits, num_contexts)?;
        Ok(HuffmanDecoder::new(table, reader))
    }

    /// Decodes the Huffman table from the provided BitReader without consuming it.
    /// Returns a HuffmanTable for further operations while allowing reuse of the BitReader.
    pub fn decode_table(
        reader: &mut R,
        max_bits: usize,
        num_contexts: usize,
    ) -> Result<HuffmanTable<EP>> {
        HuffmanTable::new(reader, max_bits, num_contexts)
    }

    /// Creates a new HuffmanDecoder with the provided table and reader.
    pub fn new(table: HuffmanTable<EP>, reader: R) -> Self {
        Self { reader, table }
    }

    /// Reads a token from the bitstream for the given context.
    /// Returns a tuple of (symbol, trailing_bits) where trailing_bits is the
    /// pre-computed number of additional bits to read after the token.
    #[inline(always)]
    fn read_token(&mut self, context: usize) -> Result<(usize, u8)> {
        let bits: u64 = self.reader.peek_bits(self.table.max_bits)?.cast();
        let info = self.table.info_[context][bits as usize];
        self.reader.skip_bits_after_peek(info.nbits as usize);
        Ok((info.symbol as usize, info.trailing_bits))
    }

    /// Reads a specified number of bits from the stream.
    /// Returns the bits as a u64 value.
    #[inline(always)]
    fn read_bits(&mut self, n: usize) -> Result<u64> {
        let bits = self.reader.read_bits(n)?;
        Ok(bits)
    }

    /// Reads a value based on the encoding parameters and context.
    /// Returns the decoded value as a usize.
    ///
    /// This method uses pre-computed trailing_bits from the lookup table,
    /// avoiding runtime computation of the number of extra bits to read.
    #[inline(always)]
    pub fn read(&mut self, context: usize) -> Result<usize> {
        let (mut token, trailing_bits) = self.read_token(context)?;

        // Fast path for explicit tokens (no trailing bits)
        if trailing_bits == 0 {
            return Ok(token);
        }

        let low = token & ((1 << EP::LSB_IN_TOKEN) - 1);
        token >>= EP::LSB_IN_TOKEN;
        let bits = self.read_bits(trailing_bits as usize)? as usize;
        let ret = (((((1 << EP::MSB_IN_TOKEN) | (token & ((1 << EP::MSB_IN_TOKEN) - 1)))
            << trailing_bits)
            | bits)
            << EP::LSB_IN_TOKEN)
            | low;
        Ok(ret)
    }
}

impl<R: BitRead<LE> + BitSeek, EP: EncodeParams> BitSeek for HuffmanDecoder<R, EP> {
    type Error = <R as BitSeek>::Error;

    fn bit_pos(&mut self) -> Result<u64, Self::Error> {
        self.reader.bit_pos()
    }

    fn set_bit_pos(&mut self, bit_pos: u64) -> Result<(), Self::Error> {
        self.reader.set_bit_pos(bit_pos)
    }
}
