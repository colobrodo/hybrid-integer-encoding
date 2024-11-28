use std::{
    fs::File,
    hint::black_box,
    io::{BufRead, BufReader},
    marker::PhantomData,
    path::PathBuf,
};

use dsi_bitstream::{
    impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec, WordAdapter},
    traits::{BitWrite, Endianness, LE},
};

use hybrid_integer_encoding::huffman::{
    DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader, IntegerData,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use rand::{prelude::Distribution, rngs::SmallRng, SeedableRng};

#[derive(Parser, Debug)]
#[clap(name = "hybrid-integer-encoding", version)]
struct App {
    #[clap(subcommand)]
    command: Command,
    #[arg(short, long, default_value = "false")]
    silent: bool,
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
        /// If true, use 4 contexts and max 8 bits per symbol.
        /// otherwise, use 1 context and max 10 bits per symbol.
        #[arg(short = 'c', long, default_value = "false")]
        use_context: bool,
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
        self.written_bits += value as usize + 1;
        Ok(written)
    }

    fn flush(&mut self) -> Result<usize, Self::Error> {
        self.writer.flush()
    }
}

impl<E: Endianness, W: BitWrite<E>> StatBitWriter<E, W> {
    fn into_inner(self) -> W {
        self.writer
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

fn encode_file(input_path: PathBuf, output_path: PathBuf, verbose: bool) -> Result<()> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let default_context = 0;

    let mut integers = IntegerData::new();
    for line in reader.lines().map_while(Result::ok) {
        // Split the line by whitespace and parse each number as u8
        for num in line.split_whitespace() {
            match num.parse::<u32>() {
                Ok(n) => integers.add(default_context, n),
                Err(_) => println!("Skipping invalid number: {}", num),
            }
        }
    }

    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(outfile));
    let mut writer = StatBitWriter::new(writer);
    let encoder = HuffmanEncoder::<DefaultEncodeParams, 1>::new(&integers);

    encoder.write_header(&mut writer)?;
    if verbose {
        println!("Header took {} bits", writer.written_bits);
    }
    for (&ctx, &number) in integers.iter() {
        encoder.write(ctx, number, &mut writer)?;
    }

    writer.flush()?;

    if verbose {
        println!("Written whole file using {} bits", writer.written_bits);
    }
    Ok(())
}

fn decode_file(path: PathBuf, lenght: u64) -> Result<()> {
    let file = File::open(path)?;
    let reader = BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(file)));
    let mut reader = HuffmanReader::<LE, _>::new(reader)?;
    let mut i = 0;
    while let Ok(value) = reader.read::<DefaultEncodeParams>(0) {
        // TODO: HACK: reading from mem word, read a 0 at the end of the bitstream but the lenght of the encoded file is not know
        if i == lenght {
            break;
        }
        println!("{}", value);
        i += 1;
    }

    Ok(())
}

fn bench<const NUM_CONTEXT: usize, const MAX_BITS: usize, const NUM_SYMBOLS: usize>(
    repeats: usize,
    nsamples: u64,
    seed: u64,
    verbose: bool,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let zipf = zipf::ZipfDistribution::new(1000000000, 1.5).unwrap();

    let mut data = IntegerData::new();
    let default_context = 0;
    for _ in 0..nsamples {
        data.add(default_context, zipf.sample(&mut rng) as u32);
    }

    let overall_start = std::time::Instant::now();

    let encoder =
        HuffmanEncoder::<DefaultEncodeParams, NUM_CONTEXT, MAX_BITS, NUM_SYMBOLS>::new(&data);
    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let writer = BufBitWriter::<LE, _>::new(word_write);
    let mut writer = StatBitWriter::new(writer);

    encoder.write_header(&mut writer).unwrap();
    if verbose {
        println!("Header took {} bits", writer.written_bits);
    }

    for (&ctx, &value) in data.iter() {
        encoder.write(ctx, value, &mut writer).unwrap();
    }
    writer.flush().unwrap();

    if verbose {
        println!("Written whole file using {} bits", writer.written_bits);
    }

    let binary_data = writer.into_inner().into_inner().unwrap().into_inner();
    let binary_data =
        unsafe { core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, data.len() * 2) };

    let mut time_per_repeat = Vec::new();

    for _ in 0..repeats {
        let reader = BufBitReader::<LE, _>::new(MemWordReader::new(&binary_data));
        let mut reader =
            HuffmanReader::<LE, _, NUM_CONTEXT, MAX_BITS, NUM_SYMBOLS>::new(reader).unwrap();

        let start = std::time::Instant::now();

        for _ in data.iter() {
            let _value = black_box(
                reader
                    .read::<DefaultEncodeParams>(default_context as usize)
                    .unwrap(),
            );
        }
        let elapsed_time = (start.elapsed().as_secs_f64() / nsamples as f64) * 1e9;
        println!("Decode:    {:>20} ns/read", elapsed_time);
        time_per_repeat.push(elapsed_time);
    }

    time_per_repeat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = time_per_repeat[time_per_repeat.len() / 2];
    println!("Median time of repeats {:>20} ns/read", median_time);
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
            encode_file(input_path, output_path, !args.silent)?;
        }
        Command::Decode { path, lenght } => {
            decode_file(path, lenght)?;
        }
        Command::Bench {
            samples,
            repeats,
            seed,
            use_context,
        } => {
            if use_context {
                bench::<1, 10, 1024>(repeats, samples, seed, !args.silent)
            } else {
                bench::<4, 8, 256>(repeats, samples, seed, !args.silent)
            }
        }
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

        let mut data = IntegerData::new();
        let default_context = 0;
        for _ in 0..nsamples {
            data.add(default_context, zipf.sample(&mut rng) as u32);
        }

        let encoder = HuffmanEncoder::<DefaultEncodeParams, 1, MAX_BITS, NUM_SYMBOLS>::new(&data);
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
        let mut reader = HuffmanReader::<LE, _, 1, MAX_BITS, NUM_SYMBOLS>::new(reader).unwrap();

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
}
