use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    marker::PhantomData,
    path::PathBuf,
};

use dsi_bitstream::{
    impls::{BufBitReader, BufBitWriter, WordAdapter},
    traits::{BitWrite, Endianness, LE},
};

use hybrid_integer_encoding::huffman::{
    DefaultEncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
};

use clap::{Parser, Subcommand};

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

fn encode_file(input_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut numbers = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        // Split the line by whitespace and parse each number as u8
        for num in line.split_whitespace() {
            match num.parse::<u64>() {
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
