use std::error::Error;

use dsi_bitstream::{
    impls::{BufBitWriter, MemWordWriterVec},
    traits::{BitWrite, LE},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use hybrid_integer_encoding::huffman::HuffmanEncoder;

fn generate_random_data(low: u64, high: u64, nsamples: usize) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut samples = Vec::new();
    for _ in 0..nsamples {
        let x1 = rng.gen_range(0.0..1.0);
        let k = x1 * x1 * (high - low) as f64;
        samples.push(k.ceil() as u64 + low);
    }
    samples
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = generate_random_data(0, 100, 1000);
    let encoder = HuffmanEncoder::new(&data);
    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let mut writer = BufBitWriter::<LE, _>::new(word_write);
    for (i, symbol) in encoder.info_.iter().enumerate() {
        if symbol.present == 0 {
            debug_assert!(symbol.nbits == 0);
            continue;
        }
        println!("{}: {}, {:#b}", i, symbol.nbits, symbol.bits)
    }

    encoder.write_header(&mut writer)?;
    for value in data {
        encoder.write(value, &mut writer)?;
    }
    writer.flush()?;
    let binary_data = writer.into_inner()?.into_inner();
    for blob in binary_data {
        println!("{:#b}", blob);
    }

    Ok(())
}
