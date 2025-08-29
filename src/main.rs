use std::{
    fmt::Debug,
    fs::File,
    hint::black_box,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::exit,
    str::FromStr,
};

use dsi_bitstream::{
    impls::{BufBitReader, BufBitWriter, MemWordReader, MemWordWriterVec, WordAdapter},
    traits::{BitSeek, BitWrite, LE},
    utils::CountBitWriter,
};
use epserde::prelude::*;

use anyhow::Result;
use clap::{Parser, Subcommand};
use lender::{for_, Lender};
use rand::prelude::*;
use rand_distr::Zipf;

use hybrid_integer_encoding::{
    graphs::{
        build_offsets, compare_graphs, measure_stats, ComparisonResult, CreateBvComp, CreateBvCompZ,
    },
    utils::IntegerData,
};
use hybrid_integer_encoding::{
    graphs::{
        compressors::CompressionParameters, convert_graph, load_graph, load_graph_seq,
        ConstantContextModel, SimpleContextModel, ZuckerliContextModel,
    },
    huffman::{
        encode, DefaultEncodeParams, EncodeParams, EntropyCoder, HuffmanEncoder, HuffmanReader,
        IntegerHistogram,
    },
    utils::StatBitWriter,
};
use webgraph::{
    prelude::*,
    traits::{RandomAccessGraph, SequentialGraph},
};

#[derive(Parser, Debug)]
#[clap(name = "hybrid-integer-encoding", version)]
struct App {
    #[clap(subcommand)]
    command: Command,
    /// If true avoid printing information about the overall space used by the encoder
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
        length: u64,
        /// The path of the compressed file
        path: PathBuf,
    },

    /// Measure the time taken to decode an encoded sample of numbers either from a file or sampled from a Zipf distribution.
    Bench {
        #[clap(subcommand)]
        command: BenchCommand,
    },

    /// Recompress a graph using the Huffman encoder
    Graph {
        #[clap(subcommand)]
        command: GraphCommand,
    },
}

#[derive(Debug, Subcommand)]
enum GraphCommand {
    /// Measure the time taken to decode an encoded sample of random numbers
    Convert {
        /// The basename of the graph to compress
        basename: PathBuf,
        /// The output path where the huffman compressed representation of the graph is saved
        output_basename: PathBuf,
        /// Compression window size
        #[arg(short = 'w', long, default_value = "7")]
        compression_window: usize,
        /// Maximum number of recursive references
        #[arg(short = 'r', long, default_value = "3")]
        max_ref_count: usize,
        #[arg(short = 'l', long, default_value = "4")]
        min_interval_length: usize,
        /// The maximum number of bits used for each word of the huffman code.
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// If specified uses the BVGraph's greedy reference selection algorithm.
        /// Otherwise it choose the reference using the Zuckerli approximated algorithm.
        #[arg(long, default_value = "false")]
        greedy_compressor: bool,
        /// The type of context model to be used to choose the distribution of the next encoded symbol.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
        /// Number of iteration of the graph compression using the huffman estimator for reference selection.
        #[arg(long, default_value = "1")]
        num_rounds: usize,
        /// Creates the offsets file at the end of the conversion.
        /// This is available only with greedy compressor, if the approximated one is employed, this parameter
        /// is ignored.
        #[arg(long, default_value = "false")]
        build_offsets: bool,
    },
    /// Compare a Huffman graph and a graph in BvGraph format and checks that all the successor lists are equal.
    /// It returns a 0 exit code if the graphs are the same, otherwise it prints the first node that is different
    /// and exits with a non-zero code.
    Eq {
        /// The basename of the Huffman graph to read and compare
        first_basename: PathBuf,
        /// The basename of the BvGraph to read and compare
        second_basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the first graph.
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// The type of context model used to encode the Huffman graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
    },
    /// Prints the edges of huffman-compressed graph in csv format
    Read {
        /// The basename of the graph to read
        basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the graph
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// The type of context model used to encode the graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
        /// csv character separator between the source node id and the destination node id (in order).
        #[arg(long, default_value_t = ',')]
        separator: char,
    },
    /// Bench the sequential access on a huffman compressed graph
    Bench {
        /// The basename of the graph to read
        basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the graph
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        #[arg(short = 'R', long, default_value = "10")]
        repeats: usize,
        /// The type of context model used to encode the graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
        /// If specified prints the results in tsv format
        /// Each line is composed by two fields: the iteration index and the running time
        /// (in nanoseconds)
        #[arg(long, default_value = "false")]
        tsv: bool,
    },
    /// Bench the random access on a huffman compressed graph
    BenchRandom {
        /// The basename of the graph to read
        basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the graph
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// The type of context model used to encode the graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
        /// The number of random sampled nodes to bench the read time of their adjacency list
        #[arg(short = 'r', long, default_value = "1000")]
        random: usize,
        /// The number of repetition performed on the test
        #[arg(short = 'R', long, default_value = "10")]
        repeats: usize,
        /// Seed to reproduce the experiment
        #[arg(short = 's', long, default_value = "0")]
        seed: u64,
        /// If specified prints the results in tsv format
        /// Each line is composed by two fields: the iteration index and the running time
        /// (in nanoseconds)
        #[arg(long, default_value = "false")]
        tsv: bool,
    },
    /// Create the offsets file from an existing huffman compressed graph
    Offsets {
        /// The basename of the graph to read
        basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the graph
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// The type of context model used to encode the graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
    },
    /// Read the graphs and print the total number of bits used to encode each component
    Stats {
        /// The basename of the graph to read
        basename: PathBuf,
        /// The maximum number of bits for each word of the huffman code used to compress the graph
        #[arg(short = 'b', long, default_value = "8")]
        max_bits: usize,
        /// The type of context model used to encode the graph.
        #[arg(long, default_value = "simple")]
        context_model: ContextModelArgument,
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
    /// The number of contexts used, chosen based on the previous encoded symbol
    #[arg(short = 'c', long, default_value = "1")]
    contexts: usize,
    /// The maximum number of bits used for each code
    #[arg(short = 'b', long, default_value = "8")]
    max_bits: usize,
}

#[derive(clap::ValueEnum, Debug, Clone)]
enum ContextModelArgument {
    /// Uses only one single context for each symbol
    Single,
    /// Uses one context for each component of the adjacency list
    Simple,
    /// Implements a partial version of context model described in zuckerli
    /// using only the information available in the current adjacency list
    Zuckerli,
}

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

    let mut integers = Vec::new();
    let mut integer_data = IntegerHistogram::new(num_contexts, 0);
    let mut last_sample = 0;
    for line in reader.lines().map_while(Result::ok) {
        // Split the line by whitespace and parse each number as u8
        for num in line.split_whitespace() {
            match num.parse::<u32>() {
                Ok(n) => {
                    let context = choose_context::<DefaultEncodeParams>(last_sample, num_contexts);
                    integers.push((context, n));
                    integer_data.add(context, n);
                    last_sample = n as u64;
                }
                Err(_) => println!("Skipping invalid number: {}", num),
            }
        }
    }

    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(outfile));
    let mut writer = CountBitWriter::<LE, _>::new(writer);
    let encoder = HuffmanEncoder::<DefaultEncodeParams>::new(integer_data, max_bits);

    encoder.write_header(&mut writer)?;
    let header_size = writer.bits_written;
    for (ctx, number) in integers {
        encoder.write(ctx, number, &mut writer)?;
    }

    writer.flush()?;

    if verbose {
        println!("Written whole file using {} bits", writer.bits_written);
        println!("Header took {} bits", header_size);
        println!(
            "Compression ration: {:.3}",
            writer.bits_written as f64 / file_size as f64
        );
    }
    Ok(())
}

fn decode_file(path: PathBuf, length: u64, max_bits: usize, num_context: usize) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = HuffmanReader::from_bitreader(
        BufBitReader::<LE, _>::new(WordAdapter::<u32, _>::new(BufReader::new(file))),
        max_bits,
        num_context,
    )?;
    let mut i = 0;
    while let Ok(value) = reader.read::<DefaultEncodeParams>(0) {
        // TODO: HACK: reading from mem word, read a 0 at the end of the bitstream but the length of the encoded file is not know
        if i == length {
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

fn bench_file<EP: EncodeParams>(
    path: PathBuf,
    repeats: usize,
    max_bits: usize,
    num_contexts: usize,
    verbose: bool,
) -> Result<()> {
    let num_symbols = 1 << max_bits;
    // Load the serialized form in a buffer
    let buffer = std::fs::read(&path)?;
    let items = <Vec<u64>>::deserialize_eps(buffer.as_ref())?;
    let mut last_integer = 0;
    let mut integer_data = IntegerData::<EP>::new(num_contexts, num_symbols);
    for &n in items {
        let context = choose_context::<EP>(last_integer, num_contexts);
        integer_data.add(context, n as u32);
        last_integer = n;
    }

    bench(integer_data, max_bits, repeats, verbose)?;
    Ok(())
}

fn bench_random_sample<EP: EncodeParams>(
    repeats: usize,
    n_samples: u64,
    max_bits: usize,
    num_contexts: usize,
    seed: u64,
    verbose: bool,
) -> Result<()> {
    let num_symbols = 1 << max_bits;
    let mut rng = SmallRng::seed_from_u64(seed);
    let zipf = Zipf::new(1000000000.0, 1.5)?;

    let mut integer_data = IntegerData::<EP>::new(num_contexts, num_symbols);
    let mut last_sample = 0;
    for _ in 0..n_samples {
        let sample = rng.sample(zipf) as u32;
        let context = choose_context::<EP>(last_sample as u64, num_contexts);
        integer_data.add(context, sample);
        last_sample = sample;
    }

    bench(integer_data, max_bits, repeats, verbose)?;
    Ok(())
}

fn bench<EP: EncodeParams>(
    integer_data: IntegerData<EP>,
    max_bits: usize,
    repeats: usize,
    verbose: bool,
) -> Result<()> {
    let overall_start = std::time::Instant::now();
    let integers = integer_data.iter().collect::<Vec<_>>();
    let num_values = integers.len();

    let histograms = integer_data.histograms();
    let num_contexts = histograms.number_of_contexts();
    let encoder = HuffmanEncoder::<EP>::new(histograms, max_bits);
    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let mut writer = StatBitWriter::<LE, _>::new(BufBitWriter::<LE, _>::new(word_write));

    encoder.write_header(&mut writer).unwrap();
    let header_size = writer.written_bits;
    if verbose {
        println!("Header took {} bits", header_size);
    }

    for &(ctx, value) in integers.iter() {
        encoder.write(ctx, value, &mut writer)?;
    }
    writer.flush()?;
    let encoded_size = writer.written_bits;
    if verbose {
        println!("Written whole file using {} bits", encoded_size);
        println!("  with payload {} bits", encoded_size - header_size);
    }

    let binary_data = writer.into_inner().into_inner()?.into_inner();
    let binary_data =
        unsafe { core::slice::from_raw_parts(binary_data.as_ptr() as *const u32, num_values * 2) };

    let mut time_per_repeat = Vec::new();

    for _ in 0..repeats {
        let mut reader = HuffmanReader::from_bitreader(
            BufBitReader::<LE, _>::new(MemWordReader::new(&binary_data)),
            max_bits,
            num_contexts,
        )?;

        let start = std::time::Instant::now();

        for (ctx, _original) in integers.iter() {
            let _value = black_box(reader.read::<DefaultEncodeParams>(*ctx as usize)?);
        }
        let elapsed_time = (start.elapsed().as_secs_f64() / num_values as f64) * 1e9;
        println!("Decode:    {:>20} ns/read", elapsed_time);
        time_per_repeat.push(elapsed_time);
    }

    time_per_repeat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = time_per_repeat[time_per_repeat.len() / 2];
    println!("Median time of repeats {:>20} ns/read", median_time);
    println!(
        "Executed bench in {:>20} s",
        overall_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let _ = env_logger::builder().try_init();

    let start = std::time::Instant::now();

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
            length,
            huffman_arguments,
        } => {
            decode_file(
                path,
                length,
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
            } => {
                bench_random_sample::<DefaultEncodeParams>(
                    bench_arguments.repeats,
                    samples,
                    huffman_arguments.max_bits,
                    huffman_arguments.contexts,
                    seed,
                    !args.silent,
                )?;
            }
            BenchCommand::File {
                bench_arguments,
                path,
                huffman_arguments,
            } => {
                bench_file::<DefaultEncodeParams>(
                    path,
                    bench_arguments.repeats,
                    huffman_arguments.max_bits,
                    huffman_arguments.contexts,
                    !args.silent,
                )?;
            }
        },
        Command::Graph { command } => match command {
            GraphCommand::Convert {
                basename,
                output_basename,
                compression_window,
                max_ref_count,
                min_interval_length,
                greedy_compressor,
                max_bits,
                num_rounds,
                build_offsets,
                context_model,
            } => {
                let compression_parameters = CompressionParameters {
                    compression_window,
                    max_ref_count,
                    min_interval_length,
                    num_rounds,
                };
                // the Zuckerli reference selection algorithm don't work in a streaming fashion like the BVGraph greedy one.
                // So, for now, it is not possible to know each time we push a node the amount of bits used to write that node
                // but only when each chunk is flushed.
                if build_offsets && !greedy_compressor {
                    log::warn!("Cannot build offsets during conversion of graph '{}', when compressing it with Zuckerli reference selection algorithm.", basename.display());
                }
                match (context_model, greedy_compressor) {
                    (ContextModelArgument::Single, false) => convert_graph::<ConstantContextModel>(
                        basename,
                        output_basename,
                        max_bits,
                        CreateBvCompZ::with_chunk_size(10000),
                        compression_parameters,
                    )?,
                    (ContextModelArgument::Single, true) => convert_graph::<ConstantContextModel>(
                        basename,
                        output_basename,
                        max_bits,
                        CreateBvComp,
                        compression_parameters,
                    )?,

                    (ContextModelArgument::Simple, false) => convert_graph::<SimpleContextModel>(
                        basename,
                        output_basename,
                        max_bits,
                        CreateBvCompZ::with_chunk_size(10000),
                        compression_parameters,
                    )?,
                    (ContextModelArgument::Simple, true) => convert_graph::<SimpleContextModel>(
                        basename,
                        output_basename,
                        max_bits,
                        CreateBvComp,
                        compression_parameters,
                    )?,
                    (ContextModelArgument::Zuckerli, false) => {
                        convert_graph::<ZuckerliContextModel<DefaultEncodeParams>>(
                            basename,
                            output_basename,
                            max_bits,
                            CreateBvCompZ::with_chunk_size(10000),
                            compression_parameters,
                        )?
                    }
                    (ContextModelArgument::Zuckerli, true) => {
                        convert_graph::<ZuckerliContextModel<DefaultEncodeParams>>(
                            basename,
                            output_basename,
                            max_bits,
                            CreateBvComp,
                            compression_parameters,
                        )?
                    }
                }
            }
            GraphCommand::Eq {
                first_basename,
                second_basename,
                max_bits,
                context_model,
            } => {
                let result = match context_model {
                    ContextModelArgument::Single => compare_graphs::<ConstantContextModel>(
                        first_basename,
                        second_basename,
                        max_bits,
                    )?,
                    ContextModelArgument::Simple => compare_graphs::<SimpleContextModel>(
                        first_basename,
                        second_basename,
                        max_bits,
                    )?,
                    ContextModelArgument::Zuckerli => compare_graphs::<ConstantContextModel>(
                        first_basename,
                        second_basename,
                        max_bits,
                    )?,
                };
                // print the result of a comparison between two graphs, and exit if it is not successful.
                match result {
                    ComparisonResult::Equal => {}
                    ComparisonResult::Different {
                        node_id,
                        left_succs: _,
                        right_succs: _,
                    } => {
                        eprintln!("Found different node {}", node_id);
                        exit(1);
                    }
                }
            }
            GraphCommand::Read {
                basename,
                max_bits,
                separator,
                context_model,
            } => match context_model {
                ContextModelArgument::Single => {
                    let graph = load_graph_seq::<ConstantContextModel>(basename, max_bits)?;
                    for_!((src, succ) in graph {
                        for dst in succ {
                            println!("{}{}{}", src, separator, dst);
                        }
                    });
                }
                ContextModelArgument::Simple => {
                    let graph = load_graph_seq::<SimpleContextModel>(basename, max_bits)?;
                    for_!((src, succ) in graph {
                        for dst in succ {
                            println!("{}{}{}", src, separator, dst);
                        }
                    });
                }
                ContextModelArgument::Zuckerli => {
                    let graph = load_graph_seq::<ZuckerliContextModel<DefaultEncodeParams>>(
                        basename, max_bits,
                    )?;
                    for_!((src, succ) in graph {
                        for dst in succ {
                            println!("{}{}{}", src, separator, dst);
                        }
                    });
                }
            },
            GraphCommand::Bench {
                basename,
                max_bits,
                repeats,
                context_model,
                tsv,
            } => {
                let bench_result = match context_model {
                    ContextModelArgument::Single => {
                        let graph = load_graph_seq::<ConstantContextModel>(basename, max_bits)?;
                        bench_seq(graph, repeats)
                    }
                    ContextModelArgument::Simple => {
                        let graph = load_graph_seq::<SimpleContextModel>(basename, max_bits)?;
                        bench_seq(graph, repeats)
                    }
                    ContextModelArgument::Zuckerli => {
                        let graph = load_graph_seq::<ZuckerliContextModel<DefaultEncodeParams>>(
                            basename, max_bits,
                        )?;
                        bench_seq(graph, repeats)
                    }
                };
                if tsv {
                    bench_result.print_as_tsv();
                } else {
                    bench_result.print_formatted();
                }
            }
            GraphCommand::BenchRandom {
                basename,
                max_bits,
                context_model,
                random,
                repeats,
                seed,
                tsv,
            } => {
                let bench_result = match context_model {
                    ContextModelArgument::Single => {
                        let graph = load_graph::<ConstantContextModel>(basename, max_bits)?;
                        bench_random_graph(graph, seed, random, repeats)
                    }
                    ContextModelArgument::Simple => {
                        let graph = load_graph::<SimpleContextModel>(basename, max_bits)?;
                        bench_random_graph(graph, seed, random, repeats)
                    }
                    ContextModelArgument::Zuckerli => {
                        let graph = load_graph::<ZuckerliContextModel<DefaultEncodeParams>>(
                            basename, max_bits,
                        )?;
                        bench_random_graph(graph, seed, random, repeats)
                    }
                };
                if tsv {
                    bench_result.print_as_tsv();
                } else {
                    bench_result.print_formatted();
                }
            }
            GraphCommand::Offsets {
                basename,
                max_bits,
                context_model,
            } => match context_model {
                ContextModelArgument::Single => {
                    let graph = load_graph_seq::<ConstantContextModel>(basename.clone(), max_bits)?;
                    build_offsets(graph, basename)?;
                }
                ContextModelArgument::Simple => {
                    let graph = load_graph_seq::<SimpleContextModel>(basename.clone(), max_bits)?;
                    build_offsets(graph, basename)?;
                }
                ContextModelArgument::Zuckerli => {
                    let graph = load_graph_seq::<ZuckerliContextModel<DefaultEncodeParams>>(
                        basename.clone(),
                        max_bits,
                    )?;
                    build_offsets(graph, basename)?;
                }
            },
            GraphCommand::Stats {
                basename,
                max_bits,
                context_model,
            } => match context_model {
                ContextModelArgument::Single => {
                    let graph = load_graph_seq::<ConstantContextModel>(basename.clone(), max_bits)?;
                    print_graph_stats(graph);
                }
                ContextModelArgument::Simple => {
                    let graph = load_graph_seq::<SimpleContextModel>(basename.clone(), max_bits)?;
                    print_graph_stats(graph);
                }
                ContextModelArgument::Zuckerli => {
                    let graph = load_graph_seq::<ZuckerliContextModel<DefaultEncodeParams>>(
                        basename.clone(),
                        max_bits,
                    )?;
                    print_graph_stats(graph);
                }
            },
        },
    }

    log::info!("The command took {}s", start.elapsed().as_secs_f64());

    Ok(())
}

struct BenchResult {
    samples: Vec<f64>,
    name: String,
    measure_token: String,
}

impl BenchResult {
    fn new(name: &str, measure_token: &str) -> BenchResult {
        BenchResult {
            samples: Vec::new(),
            name: String::from_str(name).unwrap(),
            measure_token: String::from_str(measure_token).unwrap(),
        }
    }

    fn add(&mut self, sample: f64) {
        self.samples.push(sample);
    }

    fn print_as_tsv(&self) {
        for (i, sample) in self.samples.iter().enumerate() {
            println!("{}\t{}", i + 1, sample);
        }
    }

    fn print_formatted(&self) {
        for sample in self.samples.iter() {
            println!("{}: {:>20} {}", self.name, sample, self.measure_token)
        }
    }
}

fn bench_seq(graph: impl SequentialGraph, repeats: usize) -> BenchResult {
    let mut bench_result = BenchResult::new("Sequential", "ns/arc");
    for _ in 0..repeats {
        let mut c: u64 = 0;

        let start = std::time::Instant::now();
        let mut iter = graph.iter();
        while let Some((_, succ)) = iter.next() {
            c += succ.into_iter().count() as u64;
        }
        assert_eq!(c, graph.num_arcs_hint().unwrap());

        bench_result.add((start.elapsed().as_secs_f64() / c as f64) * 1e9);
    }
    bench_result
}

fn bench_random_graph(
    graph: impl RandomAccessGraph,
    seed: u64,
    samples: usize,
    repeats: usize,
) -> BenchResult {
    let mut bench_result = BenchResult::new("Random", "ns/arc");
    // Random-access speed test
    for _ in 0..repeats {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut c: u64 = 0;
        let num_nodes = graph.num_nodes();
        let start = std::time::Instant::now();
        for _ in 0..samples {
            c += black_box(
                graph
                    .successors(rng.random_range(0..num_nodes))
                    .into_iter()
                    .count() as u64,
            );
        }

        bench_result.add((start.elapsed().as_secs_f64() / c as f64) * 1e9)
    }
    bench_result
}

fn print_graph_stats<F: SequentialDecoderFactory>(graph: BvGraphSeq<F>)
where
    for<'a> F::Decoder<'a>: Decode + BitSeek,
{
    let stats = measure_stats(graph);
    println!("total: {}", stats.total);
    println!(
        "outdegrees: {} ({:.2}%)",
        stats.outdegrees,
        stats.outdegrees as f64 / stats.total as f64 * 100.0
    );
    println!(
        "reference_offsets: {} ({:.2}%)",
        stats.reference_offsets,
        stats.reference_offsets as f64 / stats.total as f64 * 100.0
    );
    println!(
        "block_counts: {} ({:.2}%)",
        stats.block_counts,
        stats.block_counts as f64 / stats.total as f64 * 100.0
    );
    println!(
        "blocks: {} ({:.2}%)",
        stats.blocks,
        stats.blocks as f64 / stats.total as f64 * 100.0
    );
    println!(
        "interval_counts: {} ({:.2}%)",
        stats.interval_counts,
        stats.interval_counts as f64 / stats.total as f64 * 100.0
    );
    println!(
        "interval_starts: {} ({:.2}%)",
        stats.interval_starts,
        stats.interval_starts as f64 / stats.total as f64 * 100.0
    );
    println!(
        "interval_lens: {} ({:.2}%)",
        stats.interval_lens,
        stats.interval_lens as f64 / stats.total as f64 * 100.0
    );
    println!(
        "first_residuals: {} ({:.2}%)",
        stats.first_residuals,
        stats.first_residuals as f64 / stats.total as f64 * 100.0
    );
    println!(
        "residuals: {} ({:.2}%)",
        stats.residuals,
        stats.residuals as f64 / stats.total as f64 * 100.0
    );
}
