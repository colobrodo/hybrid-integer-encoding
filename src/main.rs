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

use epserde::prelude::*;
use hybrid_integer_encoding::huffman::{
    encode, DefaultEncodeParams, EncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
    IntegerData,
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
        #[clap(flatten)]
        huffman_arguments: HuffmanArguments,
        /// the input path to encode
        input_path: PathBuf,
        output_path: PathBuf,
    },

    /// Reads a compressed file and outputs the content to stdout.
    Decode {
        #[clap(flatten)]
        huffman_arguments: HuffmanArguments,
        /// The number of integer inside the file (HACK)
        lenght: u64,
        /// The path of the compressed file
        path: PathBuf,
    },

    /// Measure the time taken to decode an encoded sample of numbers either from a file or sampled from a Zipf distribution.
    Bench {
        #[clap(subcommand)]
        command: BenchCommand,
    },
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Measure the time taken to decode an encoded sample of random numbers
    Random {
        #[clap(flatten)]
        huffman_arguments: HuffmanArguments,
        #[clap(flatten)]
        bench_arguments: BenchArguments,
        /// Number of samples to decode
        #[arg(short = 'r', long, default_value = "10000")]
        samples: u64,
        /// Seed to reproduce the experiment
        #[arg(short = 's', long, default_value = "0")]
        seed: u64,
    },
    /// Measure the time taken to read an encoded sample of numbers from a epserde serialized file.
    File {
        #[clap(flatten)]
        huffman_arguments: HuffmanArguments,
        #[clap(flatten)]
        bench_arguments: BenchArguments,
        /// The path of the encoded file
        path: PathBuf,
    },
}

#[derive(Debug, clap::Args)]
struct BenchArguments {
    /// The number of time to repeat the tests
    #[arg(short = 'R', long, default_value = "10")]
    repeats: usize,
}

#[derive(Debug, clap::Args)]
struct HuffmanArguments {
    /// The number of contexts used, choosen based on the previous encoded symbol
    #[arg(short = 'c', long, default_value = "1")]
    contexts: usize,
    /// The maximum number of bits used for each code
    #[arg(short = 'b', long, default_value = "8")]
    max_bits: usize,
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

// TODO: add the ability to read both from ascii and from epserde serialized file
fn encode_file(
    input_path: PathBuf,
    output_path: PathBuf,
    max_bits: usize,
    num_contexts: usize,
    verbose: bool,
) -> Result<()> {
    let file = File::open(input_path)?;
    let file_size = file.metadata()?.len() * 8;
    let reader = BufReader::new(file);

    let mut integers = IntegerData::new(num_contexts);
    let mut last_sample = 0;
    for line in reader.lines().map_while(Result::ok) {
        // Split the line by whitespace and parse each number as u8
        for num in line.split_whitespace() {
            match num.parse::<u32>() {
                Ok(n) => {
                    let context = choose_context::<DefaultEncodeParams>(last_sample, num_contexts);
                    integers.add(context, n);
                    last_sample = n as u64;
                }
                Err(_) => println!("Skipping invalid number: {}", num),
            }
        }
    }

    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(outfile));
    let mut writer = StatBitWriter::new(writer);
    // TODO: accept from cli arguments for max bits and for number of contexts
    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&integers, max_bits);

    encoder.write_header(&mut writer)?;
    let header_size = writer.written_bits;
    for (&ctx, &number) in integers.iter() {
        encoder.write(ctx, number, &mut writer)?;
    }

    writer.flush()?;

    if verbose {
        println!("Written whole file using {} bits", writer.written_bits);
        println!("Header took {} bits", header_size);
        println!(
            "Compression ration: {:.3}",
            writer.written_bits as f64 / file_size as f64
        );
    }
    Ok(())
}

fn decode_file(path: PathBuf, lenght: u64, max_bits: usize, num_context: usize) -> Result<()> {
    let file = File::open(path)?;
    let reader = BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(file)));
    let mut reader = HuffmanReader::<LE, _>::new(reader, max_bits, num_context)?;
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

#[inline]
fn choose_context<EP: EncodeParams>(last_sample: u64, num_contexts: usize) -> u8 {
    let (token, _, _) = encode::<EP>(last_sample);
    (token.min(num_contexts - 1)) as u8
}

fn bench_file(
    path: PathBuf,
    repeats: usize,
    max_bits: usize,
    num_contexts: usize,
    verbose: bool,
) -> Result<()> {
    // Load the serialized form in a buffer
    let buffer = std::fs::read(&path)?;
    let integers = <Vec<u64>>::deserialize_eps(buffer.as_ref())?;
    let mut data = IntegerData::new(num_contexts);
    let mut last_integer = 0;
    for &n in integers {
        let context = choose_context::<DefaultEncodeParams>(last_integer, num_contexts);
        data.add(context, n as u32);
        last_integer = n;
    }

    bench(data, max_bits, repeats, verbose);
    Ok(())
}

fn bench_random(
    repeats: usize,
    nsamples: u64,
    max_bits: usize,
    num_contexts: usize,
    seed: u64,
    verbose: bool,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let zipf = zipf::ZipfDistribution::new(1000000000, 1.5).unwrap();

    let mut data = IntegerData::new(num_contexts);
    let mut last_sample = 0;
    for _ in 0..nsamples {
        let sample = zipf.sample(&mut rng) as u32;
        let context = choose_context::<DefaultEncodeParams>(last_sample as u64, num_contexts);
        data.add(context, sample);
        last_sample = sample;
    }

    bench(data, max_bits, repeats, verbose);
}

fn bench(data: IntegerData, max_bits: usize, repeats: usize, verbose: bool) {
    let overall_start = std::time::Instant::now();

    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&data, max_bits);
    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let writer = BufBitWriter::<LE, _>::new(word_write);
    let mut writer = StatBitWriter::new(writer);

    encoder.write_header(&mut writer).unwrap();
    let header_size = writer.written_bits;
    if verbose {
        println!("Header took {} bits", header_size);
    }

    for (&ctx, &value) in data.iter() {
        encoder.write(ctx, value, &mut writer).unwrap();
    }
    writer.flush().unwrap();
    let encoded_size = writer.written_bits;
    if verbose {
        println!("Written whole file using {} bits", encoded_size);
        println!("  with payload {} bits", encoded_size - header_size);
    }

    let binary_data = writer.into_inner().into_inner().unwrap().into_inner();
    let binary_data =
        unsafe { core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, data.len() * 2) };

    let mut time_per_repeat = Vec::new();

    for _ in 0..repeats {
        let reader = BufBitReader::<LE, _>::new(MemWordReader::new(&binary_data));
        let mut reader =
            HuffmanReader::<LE, _>::new(reader, max_bits, data.number_of_contexts()).unwrap();

        let start = std::time::Instant::now();

        for (&ctx, _original) in data.iter() {
            let _value = black_box(reader.read::<DefaultEncodeParams>(ctx as usize).unwrap());
        }
        let elapsed_time = (start.elapsed().as_secs_f64() / data.len() as f64) * 1e9;
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
            huffman_arguments,
        } => {
            encode_file(
                input_path,
                output_path,
                huffman_arguments.max_bits,
                huffman_arguments.contexts,
                !args.silent,
            )?;
        }
        Command::Decode {
            path,
            lenght,
            huffman_arguments,
        } => {
            decode_file(
                path,
                lenght,
                huffman_arguments.max_bits,
                huffman_arguments.contexts,
            )?;
        }
        Command::Bench { command } => match command {
            BenchCommand::Random {
                bench_arguments,
                samples,
                seed,
                huffman_arguments,
            } => bench_random(
                bench_arguments.repeats,
                samples,
                huffman_arguments.max_bits,
                huffman_arguments.contexts,
                seed,
                !args.silent,
            ),
            BenchCommand::File {
                bench_arguments,
                path,
                huffman_arguments,
            } => bench_file(
                path,
                bench_arguments.repeats,
                huffman_arguments.max_bits,
                huffman_arguments.contexts,
                !args.silent,
            )?,
        },
    }

    Ok(())
}
