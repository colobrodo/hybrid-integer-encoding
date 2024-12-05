use std::marker::PhantomData;

use dsi_bitstream::traits::{BitWrite, Endianness};

use super::MockBitWriter;

pub struct StatBitWriter<E: Endianness, W: BitWrite<E>> {
    writer: W,
    pub written_bits: usize,
    _marker: PhantomData<E>,
}

impl<E: Endianness> StatBitWriter<E, MockBitWriter> {
    pub fn empty() -> Self {
        Self::new(MockBitWriter)
    }
}

impl<E: Endianness, W: BitWrite<E>> BitWrite<E> for StatBitWriter<E, W> {
    type Error = W::Error;

    fn write_bits(&mut self, value: u64, n: usize) -> Result<usize, Self::Error> {
        let written = self.writer.write_bits(value, n)?;
        self.written_bits += n;
        Ok(written)
    }

    fn write_unary(&mut self, value: u64) -> Result<usize, Self::Error> {
        let written = self.writer.write_unary(value)?;
        self.written_bits += value as usize + 1;
        Ok(written)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.writer.flush()
    }
}

impl<E: Endianness, W: BitWrite<E>> StatBitWriter<E, W> {
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<E: Endianness, W: BitWrite<E>> StatBitWriter<E, W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            written_bits: 0,
            _marker: core::marker::PhantomData::<E>,
        }
    }
}
