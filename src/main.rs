use std::{
    fs::File,
    hint::black_box,
    io::{BufRead, BufReader},
    marker::PhantomData,
    mem,
    path::PathBuf,
};

use dsi_bitstream::{
    impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec, WordAdapter},
    traits::{BitWrite, Endianness, LE},
};

use hybrid_integer_encoding::huffman::{
    DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use rand::{prelude::Distribution, rngs::SmallRng, SeedableRng};

#[derive(Parser, Debug)]
#[clap(name = "hybrid-integer-encoding", version)]
struct App {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Encode a file in ASCII format containing a set of integers.
    Encode {
        /// the input path to encode
        input_path: PathBuf,
        output_path: PathBuf,
    },

    /// Reads a compressed file and outputs the content to stdout.
    Decode {
        /// The number of integer inside the file (HACK)
        lenght: u64,
        /// The path of the compressed file
        path: PathBuf,
    },

    /// Measure the time taken to decode an encoded sample of random numbers
    Bench {
        /// Number of samples to decode
        #[arg(short = 'r', long, default_value = "10000")]
        samples: u64,
        /// The number of time to repeat the tests
        #[arg(short = 'R', long, default_value = "10")]
        repeats: usize,
        /// Seed to reproduce the experiment
        #[arg(short = 's', long, default_value = "0")]
        seed: u64,
    },
}

struct StatBitWriter<E: Endianness, W: BitWrite<E>> {
    writer: W,
    written_bits: usize,
    _marker: PhantomData<E>,
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
        self.written_bits += value as usize;
        Ok(written)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.writer.flush()
    }
}

impl<E: Endianness, W: BitWrite<E>> StatBitWriter<E, W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            written_bits: 0,
            _marker: core::marker::PhantomData::<E>,
        }
    }
}

fn encode_file(input_path: PathBuf, output_path: PathBuf) -> Result<()> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut numbers = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        // Split the line by whitespace and parse each number as u8
        for num in line.split_whitespace() {
            match num.parse::<u32>() {
                Ok(n) => numbers.push(n),
                Err(_) => println!("Skipping invalid number: {}", num),
            }
        }
    }

    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(outfile));
    let mut writer = StatBitWriter::new(writer);
    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&numbers);

    encoder.write_header(&mut writer)?;
    println!("Header took {} bits", writer.written_bits);
    for number in numbers {
        encoder.write(number, &mut writer)?;
    }

    writer.flush()?;

    Ok(())
}

fn decode_file(path: PathBuf, lenght: u64) -> Result<()> {
    let file = File::open(path)?;
    let reader = BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(file)));
    let mut reader = HuffmanReader::new(reader)?;
    let mut i = 0;
    while let Ok(value) = reader.read::<DefaultEncodeParams>() {
        // TODO: HACK: reading from mem word, read a 0 at the end of the bitstream but the lenght of the encoded file is not know
        if i == lenght {
            break;
        }
        println!("{}", value);
        i += 1;
    }

    Ok(())
}

fn bench(repeats: usize, nsamples: u64, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let zipf = zipf::ZipfDistribution::new(1000000000, 1.5).unwrap();

    let data = (0..nsamples)
        .map(|_| zipf.sample(&mut rng) as u32)
        .collect::<Vec<_>>();

    let overall_start = std::time::Instant::now();

    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&data);
    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let mut writer = BufBitWriter::<LE, _>::new(word_write);

    encoder.write_header(&mut writer).unwrap();
    for value in &data {
        encoder.write(*value, &mut writer).unwrap();
    }
    writer.flush().unwrap();

    let binary_data = writer.into_inner().unwrap().into_inner();
    let binary_data = mem::ManuallyDrop::new(binary_data);
    let binary_data = unsafe {
        let ptr = binary_data.as_ptr() as *mut u32;
        Vec::from_raw_parts(ptr, binary_data.len() * 2, binary_data.capacity() * 2)
    };

    for _ in 0..repeats {
        let reader = BufBitReader::<LE, _>::new(MemWordReader::new(&binary_data));
        let mut reader = HuffmanReader::new(reader).unwrap();

        let start = std::time::Instant::now();

        for _ in &data {
            let _value = black_box(reader.read::<DefaultEncodeParams>().unwrap());
        }
        println!(
            "Decode:    {:>20} ns/sample",
            (start.elapsed().as_secs_f64() / nsamples as f64) * 1e9
        );
    }

    println!(
        "Executed bench in {:>20} s",
        overall_start.elapsed().as_secs_f64()
    )
}

fn main() -> Result<()> {
    let args = App::parse();
    match args.command {
        Command::Encode {
            input_path,
            output_path,
        } => {
            encode_file(input_path, output_path)?;
        }
        Command::Decode { path, lenght } => {
            decode_file(path, lenght)?;
        }
        Command::Bench {
            samples,
            repeats,
            seed,
        } => bench(repeats, samples, seed),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use dsi_bitstream::{
        impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec},
        traits::{BitWrite, LE},
    };
    use hybrid_integer_encoding::huffman::{
        DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
    };
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn generate_random_data(low: u32, high: u32, nsamples: usize) -> Vec<u32> {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut samples = Vec::new();
        for _ in 0..nsamples {
            let x1 = rng.gen_range(0.0..1.0);
            let k = x1 * x1 * (high - low) as f32;
            samples.push(k.ceil() as u32 + low);
        }
        samples
    }

    #[test]
    fn encode_and_decode() {
        let nsamples = 1000;
        let data = generate_random_data(0, 100, nsamples);
        let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&data);
        let word_write = MemWordWriterVec::new(Vec::<u64>::new());
        let mut writer = BufBitWriter::<LE, _>::new(word_write);

        encoder.write_header(&mut writer).unwrap();
        for value in &data {
            encoder.write(*value, &mut writer).unwrap();
        }
        writer.flush().unwrap();

        let binary_data = writer.into_inner().unwrap().into_inner();
        let binary_data = unsafe {
            core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, data.len() * 2)
        };

        let reader = BufBitReader::<LE, _>::new(MemWordReader::new(binary_data));
        let mut reader = HuffmanReader::new(reader).unwrap();

        for original in &data {
            let value = reader.read::<DefaultEncodeParams>().unwrap();
            assert_eq!(value, *original as usize);
        }
    }
}
