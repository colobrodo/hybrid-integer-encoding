use std::convert::Infallible;

use dsi_bitstream::traits::{BitWrite, Endianness};

pub struct MockBitWriter;

impl<E: Endianness> BitWrite<E> for MockBitWriter {
    type Error = Infallible;

    fn write_bits(&mut self, _value: u64, n: usize) -> std::result::Result<usize, Self::Error> {
        Ok(n)
    }

    fn write_unary(&mut self, value: u64) -> std::result::Result<usize, Self::Error> {
        Ok(value as usize + 1)
    }

    fn flush(&mut self) -> std::result::Result<usize, Self::Error> {
        Ok(0)
    }
}
