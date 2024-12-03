#[cfg(test)]
mod tests {
    use dsi_bitstream::{
        impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec},
        traits::{BitWrite, LE},
    };
    use hybrid_integer_encoding::huffman::{
        DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader, IntegerData,
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

        let mut data = IntegerData::new(NUM_CONTEXT);
        let default_context = 0;
        for _ in 0..nsamples {
            data.add(default_context, zipf.sample(&mut rng) as u32);
        }

        let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&data, MAX_BITS);
        let word_write = MemWordWriterVec::new(Vec::<u64>::new());
        let mut writer = BufBitWriter::<LE, _>::new(word_write);

        encoder.write_header(&mut writer).unwrap();
        for (&ctx, &value) in data.iter() {
            encoder.write(ctx, value, &mut writer).unwrap();
        }
        writer.flush().unwrap();

        let binary_data = writer.into_inner().unwrap().into_inner();
        let binary_data = unsafe {
            core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, data.len() * 2)
        };

        let reader = BufBitReader::<LE, _>::new(MemWordReader::new(binary_data));
        let mut reader = HuffmanReader::<LE, _>::new(reader, MAX_BITS, 1).unwrap();

        for (&ctx, &original) in data.iter() {
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

    #[test]
    fn encode_and_decode_with_11_max_bits_and_16_contexts() {
        const MAX_BITS: usize = 11;
        encode_and_decode::<16, MAX_BITS, { 1 << MAX_BITS }>(0);
    }
    #[test]
    fn encode_and_decode_with_12_max_bits_and_8_contexts() {
        const MAX_BITS: usize = 12;
        encode_and_decode::<8, MAX_BITS, { 1 << MAX_BITS }>(0);
    }
}
