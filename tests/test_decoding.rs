#[cfg(test)]
mod tests {
    use dsi_bitstream::{
        impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec},
        traits::{BitWrite, LE},
    };
    use hybrid_integer_encoding::huffman::{
        DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader, IntegerHistogram,
        DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS,
    };
    use rand::{prelude::Distribution, rngs::SmallRng, SeedableRng};

    fn encode_and_decode<
        const NUM_CONTEXT: usize,
        const MAX_BITS: usize,
        const NUM_SYMBOLS: usize,
    >(
        seed: u64,
    ) {
        let nsamples = 1000;
        let mut rng = SmallRng::seed_from_u64(seed);
        let zipf = zipf::ZipfDistribution::new(1000000000, 1.5).unwrap();

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

        encoder.write_header(&mut writer).unwrap();
        for &(ctx, value) in integers.iter() {
            encoder.write(ctx, value, &mut writer).unwrap();
        }
        writer.flush().unwrap();

        let binary_data = writer.into_inner().unwrap().into_inner();
        let binary_data = unsafe {
            core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, nsamples * 2)
        };

        let mut reader = HuffmanReader::<LE, _>::from_bitreader(
            BufBitReader::<LE, _>::new(MemWordReader::new(binary_data)),
            MAX_BITS,
            NUM_CONTEXT,
        )
        .unwrap();

        for &(ctx, original) in integers.iter() {
            let value = reader.read::<DefaultEncodeParams>(ctx as usize).unwrap();
            assert_eq!(value, original as usize);
        }
    }

    #[test]
    fn encode_and_decode_with_default_params() {
        encode_and_decode::<1, DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS>(0);
    }

    #[test]
    fn encode_and_decode2() {
        encode_and_decode::<1, DEFAULT_MAX_HUFFMAN_BITS, DEFAULT_NUM_SYMBOLS>(31415);
    }

    #[test]
    fn encode_and_decode_with_custom_params() {
        const MAX_BITS: usize = 10;
        encode_and_decode::<1, MAX_BITS, { 1 << MAX_BITS }>(0);
    }

    #[test]
    fn encode_and_decode_with_multiple_contexts() {
        const MAX_BITS: usize = 10;
        encode_and_decode::<4, MAX_BITS, { 1 << MAX_BITS }>(0);
    }
}
