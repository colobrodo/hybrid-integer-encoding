#[cfg(test)]
mod tests {
    use anyhow::Result;
    use dsi_bitstream::{
        impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec},
        traits::{BitWrite, LE},
    };
    use hybrid_integer_encoding::huffman::{
        encode, DefaultEncodeParams, EntropyCoder, HuffmanDecoder, HuffmanEncoder,
        IntegerHistogram, DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS,
    };
    use rand::prelude::*;
    use rand_distr::Zipf;

    fn encode_and_decode<
        const NUM_CONTEXT: usize,
        const MAX_BITS: usize,
        const NUM_SYMBOLS: usize,
    >(
        seed: u64,
    ) -> Result<()> {
        let nsamples = 100_000;
        let mut rng = SmallRng::seed_from_u64(seed);
        let zipf = Zipf::new(1000_000_000.0, 1.5)?;

        let mut data = IntegerHistogram::new(NUM_CONTEXT, NUM_SYMBOLS);
        let default_context = 0;
        let mut integers = Vec::with_capacity(nsamples);
        for _ in 0..nsamples {
            let sample = zipf.sample(&mut rng) as u32;
            data.add(default_context, sample);
            integers.push((default_context, sample));
        }

        let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(data, MAX_BITS);
        let word_write = MemWordWriterVec::new(Vec::<u64>::new());
        let mut writer = BufBitWriter::<LE, _>::new(word_write);

        encoder.write_header(&mut writer)?;
        for &(ctx, value) in integers.iter() {
            encoder.write(ctx, value, &mut writer)?;
        }
        writer.flush()?;

        let binary_data = writer.into_inner()?.into_inner();
        let binary_data = unsafe {
            core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, nsamples * 2)
        };

        let mut reader = HuffmanDecoder::from_bitreader(
            BufBitReader::<LE, _>::new(MemWordReader::new(binary_data)),
            MAX_BITS,
            NUM_CONTEXT,
        )?;

        for &(ctx, original) in integers.iter() {
            let value = reader.read::<DefaultEncodeParams>(ctx as usize)?;
            assert_eq!(value, original as usize);
        }
        Ok(())
    }

    #[test]
    fn test_encode_and_decode_with_default_params() -> anyhow::Result<()> {
        encode_and_decode::<1, DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS>(0)?;
        Ok(())
    }

    #[test]
    fn test_encode_and_decode2() -> anyhow::Result<()> {
        encode_and_decode::<1, DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS>(31415)?;
        Ok(())
    }

    #[test]
    fn test_encode_and_decode_with_custom_params() -> anyhow::Result<()> {
        const MAX_BITS: usize = 10;
        encode_and_decode::<1, MAX_BITS, { 1 << MAX_BITS }>(0)?;
        Ok(())
    }

    #[test]
    fn test_encode_and_decode_with_multiple_contexts() -> anyhow::Result<()> {
        const MAX_BITS: usize = 10;
        encode_and_decode::<4, MAX_BITS, { 1 << MAX_BITS }>(0)?;
        Ok(())
    }

    #[test]
    fn test_encode_big_num() -> Result<()> {
        let (token, n_bits, _) = encode::<DefaultEncodeParams>(17179902313);
        assert_eq!(n_bits, 31);
        assert_eq!(token, 257);
        Ok(())
    }
}
