use core::num;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read},
    path::PathBuf,
};

use dsi_bitstream::{
    impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec, WordAdapter},
    traits::{BitWrite, LE},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use hybrid_integer_encoding::huffman::{
    DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
};

use clap::{Args, Parser, Subcommand};

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
        /// The path of the compressed file
        path: PathBuf,
    },
}

fn encode_file(input_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut numbers = Vec::new();
    for line in reader.lines() {
        if let Ok(num_str) = line {
            // Split the line by whitespace and parse each number as u8
            for num in num_str.split_whitespace() {
                match num.parse::<u64>() {
                    Ok(n) => numbers.push(n),
                    Err(_) => println!("Skipping invalid number: {}", num),
                }
            }
        }
    }

    let outfile = File::create(output_path)?;
    let mut writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(outfile));
    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(&numbers);

    encoder.write_header(&mut writer)?;
    for number in numbers {
        encoder.write(number, &mut writer)?;
    }

    writer.flush()?;

    Ok(())
}

fn decode_file(path: PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(file)));
    let mut reader = HuffmanReader::new(reader)?;
    while let Ok(value) = reader.read::<DefaultEncodeParams>() {
        if value == 0 {
            break;
        }
        println!("{}", value);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = App::parse();
    match args.command {
        Command::Encode {
            input_path,
            output_path,
        } => {
            encode_file(input_path, output_path)?;
        }
        Command::Decode { path } => {
            decode_file(path)?;
        }
    }

    Ok(())
}
