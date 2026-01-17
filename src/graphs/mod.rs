mod component;
mod context_model;
pub mod estimator;
mod huffman_graph_decoder;
mod huffman_graph_encoder;
pub mod parameters;
mod stats;

use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{concurrent_progress_logger, prelude::*};
use epserde::deser::{Deserialize, Owned};
use epserde::prelude::*;
use lender::*;
use mmap_rs::MmapFlags;
use rayon::current_num_threads;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Seek};
use std::path::Path;
use std::time::Duration;
use sux::prelude::*;
use webgraph::prelude::{SequentialLabeling, *};

use component::*;
pub use context_model::*;
use estimator::*;
pub use huffman_graph_decoder::*;
use huffman_graph_encoder::*;
pub use parameters::*;
pub use stats::*;

use crate::huffman::{CostModel, DefaultEncodeParams, EncodeParams, IntegerHistograms};

type HuffmanEstimatedEncoderBuilder<EP, C> =
    HuffmanGraphEncoderBuilder<HuffmanEstimator<EP, CostModel<EP>, C>, C, EP>;

/// A factory trait for creating thread-local estimators of a specific encoders.
///
/// This trait is used to instantiate estimators in parallel contexts where each thread
/// needs its own independent estimator instance. The estimator type `E` must implement
trait ThreadEstimatorFactory<'a, E: Encode + Send + Sync> {
    fn create_estimator(&self) -> E;
}

struct DefaultEstimatorFactory<E: Encode + Send + Sync + Default> {
    _marker: core::marker::PhantomData<E>,
}

impl<'a, E: Encode + Send + Sync + Default> ThreadEstimatorFactory<'a, E>
    for DefaultEstimatorFactory<E>
{
    fn create_estimator(&self) -> E {
        E::default()
    }
}

impl<E: Encode + Send + Sync + Default> Default for DefaultEstimatorFactory<E> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

/// Factory for creating `HuffmanEstimator` instances with a captured cost model reference.
/// This factory captures a reference to the cost model from a previous estimation round and creates
/// thread-local `HuffmanEstimator` instances for parallel compression.
struct HuffmanEstimatorFactory<'a, EP: EncodeParams, C: ContextModel> {
    cost_model: &'a CostModel<EP>,
    _marker: core::marker::PhantomData<C>,
}

impl<'a, EP: EncodeParams + Send + Sync, C: ContextModel + Default + Copy + Send + Sync>
    ThreadEstimatorFactory<'a, HuffmanEstimator<EP, &'a CostModel<EP>, C>>
    for HuffmanEstimatorFactory<'a, EP, C>
{
    fn create_estimator(&self) -> HuffmanEstimator<EP, &'a CostModel<EP>, C> {
        HuffmanEstimator::new(self.cost_model, C::default())
    }
}

/// Run one reference-selection pass: collect symbols with the given estimator.
/// Returns a builder seeded with frequency estimates for the next stage.
fn reference_selection_round<
    G: SequentialGraph,
    EP: EncodeParams,
    E: Encode,
    C: ContextModel + Default,
>(
    graph: &G,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    compression_parameters: &CompressionParameters,
    msg: impl AsRef<str>,
    pl: &mut ConcurrentWrapper,
) -> Result<HuffmanEstimatedEncoderBuilder<EP, C>> {
    let num_symbols = 1 << compression_parameters.max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<_, _, EP>::new(num_symbols, huffman_estimator, C::default());
    // discard all the offsets
    let offsets_writer = OffsetsWriter::from_write(io::empty(), true)?;
    pl.item_name("node")
        .expected_updates(Some(graph.num_nodes()));
    pl.start(msg);
    match compression_parameters.compressor {
        CompressorType::Approximated { chunk_size } => {
            let mut compressor = BvCompZ::new(
                &mut huffman_graph_encoder_builder,
                offsets_writer,
                compression_parameters.compression_window,
                chunk_size,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
        CompressorType::Greedy => {
            let mut compressor = BvComp::new(
                &mut huffman_graph_encoder_builder,
                offsets_writer,
                compression_parameters.compression_window,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
    }
    pl.done();
    Ok(huffman_graph_encoder_builder)
}

/// Generic helper for parallel compression: it iterates over a split lender and computes the symbols' frequencies
/// with the given estimator factory, and in the end, returns the merged histograms.
fn parallel_compression_round_helper<'a, EP, E, C, G, Factory>(
    graph: &G,
    compression_parameters: &CompressionParameters,
    factory: &Factory,
    msg: impl AsRef<str>,
    cpl: &mut ConcurrentWrapper,
) -> Result<IntegerHistograms<EP>>
where
    EP: EncodeParams + Send + Sync,
    E: Encode + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'b> SplitLabeling<SplitLender<'b>: ExactSizeLender + Send>,
    Factory: ThreadEstimatorFactory<'a, E> + Send + Sync,
{
    let mut iter = graph.iter();
    let num_threads = current_num_threads();
    let to_skip = graph.num_nodes().div_ceil(num_threads);
    for _ in 0..5 {
        iter.advance_by(to_skip).unwrap();
        cpl.info(format_args!("Skipping {} items", to_skip));
        let _ = iter.next();
    }

    cpl.info(format_args!("Started parallel compression helper"));
    let num_symbols = 1 << compression_parameters.max_bits;
    let num_threads = current_num_threads();
    cpl.info(format_args!(
        "Now we know that we have {} threads",
        num_threads
    ));
    let split_iter = graph.split_iter(num_threads);
    cpl.info(format_args!("We obtained the split iter"));
    let mut split_iter = split_iter.into_iter();
    cpl.info(format_args!(
        "Called into_iter() on iter with length {}",
        split_iter.len()
    ));
    let mut i = 0;
    while let Some(_thread_lender) = split_iter.next() {
        cpl.info(format_args!("iterated {} items", i));
        i += 1;
    }
    let split_iter = split_iter.collect::<Vec<_>>();
    cpl.info(format_args!("now collected the iterator in a vec"));

    cpl.start(msg);

    // iterate the splitted version of the graph in parallel
    let thread_histograms: Vec<IntegerHistograms<EP>> = split_iter
        .into_iter()
        .enumerate()
        .par_bridge()
        .map_with(
            cpl.clone(),
            |pl, (thread_id, mut thread_lender)| -> Result<IntegerHistograms<EP>> {
                pl.info(format_args!("Started thread with id {}", thread_id));

                let Some((node_id, successors)) = thread_lender.next() else {
                    return Err(anyhow::anyhow!(
                        "Empty chunked size of compressors in thread {}",
                        thread_id
                    ));
                };

                let first_node = node_id;
                pl.info(format_args!(
                    "[{}] Starting compressing chunk from {}",
                    thread_id, first_node
                ));

                // Initialize local builder with the estimator from factory
                let mut thread_builder = HuffmanGraphEncoderBuilder::<_, _, EP>::new(
                    num_symbols,
                    factory.create_estimator(),
                    C::default(),
                );
                let offsets_writer = OffsetsWriter::from_write(io::empty(), true)?;

                match compression_parameters.compressor {
                    CompressorType::Approximated { chunk_size } => {
                        let mut compressor = BvCompZ::new(
                            &mut thread_builder,
                            offsets_writer,
                            compression_parameters.compression_window,
                            chunk_size,
                            compression_parameters.max_ref_count,
                            compression_parameters.min_interval_length,
                            first_node,
                        );
                        compressor.push(successors).unwrap();
                        pl.update();
                        for_![ (_, succ) in thread_lender {
                            compressor.push(succ)?;
                            pl.update();
                        }];
                        compressor.flush()?;
                    }
                    CompressorType::Greedy => {
                        let mut compressor = BvComp::new(
                            &mut thread_builder,
                            offsets_writer,
                            compression_parameters.compression_window,
                            compression_parameters.max_ref_count,
                            compression_parameters.min_interval_length,
                            first_node,
                        );
                        compressor.push(successors).unwrap();
                        pl.update();
                        for_![ (_, succ) in thread_lender {
                            compressor.push(succ)?;
                            pl.update();
                        }];
                        compressor.flush()?;
                    }
                }

                Ok(thread_builder.histograms())
            },
        )
        .collect::<Result<Vec<_>>>()?;

    cpl.info(format_args!("Merging histograms from separate threads"));

    // Merge Phase: Combine all local histograms into one
    let mut shared_histograms = IntegerHistograms::<EP>::new(C::num_contexts(), num_symbols);
    for h in thread_histograms {
        shared_histograms.add_all(&h);
    }
    cpl.done();

    Ok(shared_histograms)
}

/// Run the first compression round in parallel using the given estimator type.
/// Splits the graph, compresses each chunk with its own builder, then merges histograms.
fn parallel_first_reference_selection_round<
    EP: EncodeParams + Send + Sync,
    E: Encode + Default + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    graph: &G,
    compression_parameters: &CompressionParameters,
    pl: &mut ConcurrentWrapper,
    msg: impl AsRef<str>,
) -> Result<HuffmanGraphEncoderBuilder<E, C, EP>> {
    pl.info(format_args!(
        "called parallel_first_reference_selection_round"
    ));
    let factory = DefaultEstimatorFactory::<E>::default();
    pl.info(format_args!("Created default estimator"));
    let shared_histograms = parallel_compression_round_helper::<EP, E, C, G, _>(
        graph,
        compression_parameters,
        &factory,
        msg,
        pl,
    )?;

    let builder = HuffmanGraphEncoderBuilder::<_, _, EP>::from_histograms(
        shared_histograms,
        E::default(),
        C::default(),
    );
    Ok(builder)
}

#[allow(clippy::too_many_arguments)]
/// Run one reference-selection pass: collect symbols with the given estimator.
/// Returns a builder seeded with frequency estimates for the next stage.
fn parallel_reference_selection_round<
    EP: EncodeParams + Send + Sync,
    E: Encode,
    C: ContextModel + Default + Copy + Send + Sync,
>(
    graph: &(impl SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>),
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    compression_parameters: &CompressionParameters,
    msg: impl AsRef<str>,
    pl: &mut ConcurrentWrapper,
) -> Result<HuffmanEstimatedEncoderBuilder<EP, C>> {
    // obtain cost model of the previous iteration
    let cost_model = huffman_graph_encoder_builder.histograms().cost();

    // Run parallel compression with Huffman estimator factory
    let factory = HuffmanEstimatorFactory::<'_, EP, C> {
        cost_model: &cost_model,
        _marker: std::marker::PhantomData,
    };
    let shared_histograms = parallel_compression_round_helper::<EP, _, C, _, _>(
        graph,
        compression_parameters,
        &factory,
        msg,
        pl,
    )?;

    // Finalize builder with Huffman estimator and merged histograms
    let huffman_estimator = HuffmanEstimator::new(cost_model, C::default());
    let builder = HuffmanGraphEncoderBuilder::<_, _, EP>::from_histograms(
        shared_histograms,
        huffman_estimator,
        C::default(),
    );

    Ok(builder)
}

/// Build the Elias-Fano index for a graph file at `src` using offsets.
/// Reads the graph file size and constructs the EF index on disk.
fn build_eliasfano<E: Endianness + 'static>(src: impl AsRef<Path>) -> Result<()>
where
    for<'a> BufBitReader<E, MemWordReader<u32, &'a [u32]>>: CodesRead<E> + BitSeek,
{
    let mut pl = ProgressLogger::default();
    pl.display_memory(true).item_name("offset");

    let src = src.as_ref();
    // Creates the offsets file
    let of_file_path = src.with_extension(OFFSETS_EXTENSION);

    let graph_path = src.with_extension(GRAPH_EXTENSION);
    log::info!("Getting size of graph at '{}'", graph_path.display());
    let mut file = File::open(&graph_path)
        .with_context(|| format!("Could not open {}", graph_path.display()))?;
    let file_len = 8 * file
        .seek(std::io::SeekFrom::End(0))
        .with_context(|| format!("Could not seek in {}", graph_path.display()))?;

    // if the num_of_nodes is not present, read it from the properties file
    // otherwise use the provided value, this is so we can build the Elias-Fano
    // for offsets of any custom format that might not use the standard
    // properties file
    let properties_path = src.with_extension(PROPERTIES_EXTENSION);
    log::info!(
        "Reading num_of_nodes from properties file at '{}'",
        properties_path.display()
    );
    let f = File::open(&properties_path).with_context(|| {
        format!(
            "Could not open properties file: {}",
            properties_path.display()
        )
    })?;
    let map = java_properties::read(BufReader::new(f))?;
    let num_nodes = map
        .get("nodes")
        .ok_or(anyhow::anyhow!(
            "Cannot find 'nodes' field in properties file '{}'",
            properties_path.display()
        ))?
        .parse::<usize>()?;
    pl.expected_updates(Some(num_nodes));

    let mut efb = EliasFanoBuilder::new(num_nodes + 1, file_len as usize);

    log::info!("Checking if offsets exists at '{}'", of_file_path.display());
    // if the offset files exists, read it to build elias-fano
    assert!(of_file_path.exists(), "The offsets doesn't file exists");
    let of = <MmapHelper<u32>>::mmap(of_file_path, MmapFlags::SEQUENTIAL)?;
    build_eliasfano_from_offsets::<E>(num_nodes, of.new_reader(), &mut pl, &mut efb)?;

    serialize_eliasfano(src, efb, &mut pl)
}

/// Convert a stream of gamma-coded offsets into an Elias-Fano builder.
/// Pushes decoded offsets into `efb` and reports progress via `pl`.
pub fn build_eliasfano_from_offsets<E: Endianness>(
    num_nodes: usize,
    mut reader: impl GammaRead<E>,
    pl: &mut impl ProgressLog,
    efb: &mut EliasFanoBuilder,
) -> Result<()> {
    log::info!("Building Elias-Fano from offsets...");

    // progress bar
    pl.start("Translating offsets to EliasFano...");
    // read the graph a write the offsets
    let mut offset = 0;
    for _node_id in 0..num_nodes + 1 {
        // write where
        offset += reader.read_gamma().context("Could not read gamma")?;
        efb.push(offset as _);
        // decode the next nodes so we know where the next node_id starts
        pl.light_update();
    }
    pl.done();
    Ok(())
}

/// Finalize and write an Elias-Fano structure to disk at `src`.
/// Builds the high-bit index and serializes the EF to a file.
pub fn serialize_eliasfano(
    src: impl AsRef<Path>,
    efb: EliasFanoBuilder,
    pl: &mut impl ProgressLog,
) -> Result<()> {
    let ef = efb.build();
    pl.done();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Building the Index over the ones in the high-bits...");
    let ef: EF = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _, 12, 4>::new) };
    pl.done();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true);
    pl.start("Writing to disk...");

    let ef_path = src.as_ref().with_extension(EF_EXTENSION);
    log::info!("Creating Elias-Fano at '{}'", ef_path.display());
    let mut ef_file = BufWriter::new(
        File::create(&ef_path)
            .with_context(|| format!("Could not create {}", ef_path.display()))?,
    );

    // serialize and dump the schema to disk
    unsafe {
        ef.serialize(&mut ef_file)
            .with_context(|| format!("Could not serialize EliasFano to {}", ef_path.display()))
    }?;

    pl.done();
    Ok(())
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
pub fn sequential_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
) -> Result<()> {
    convert_graph_file::<C>(basename, output_basename, compression_parameters, false)
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph running the estimation rounds in parallel.
/// The converted graph is written to `output_basename`.
pub fn parallel_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
) -> Result<()> {
    convert_graph_file::<C>(basename, output_basename, compression_parameters, true)
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
/// The `parallel` arguments controls if the estimation rounds are executed in parallel or sequentially.
pub fn convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    parallel: bool,
) -> Result<()> {
    let seq_graph = BvGraphSeq::with_basename(&basename)
        .endianness::<BE>()
        .load()?;

    convert_graph::<C, _>(
        &seq_graph,
        output_basename,
        compression_parameters,
        parallel,
    )
}

#[allow(clippy::too_many_arguments)]
/// Run the iterative estimation process, starting from the first compression round.
/// Creates the initial builder with the given estimator type and handles both sequential
/// and parallel processing for all rounds.
fn run_conversion_rounds<
    EP: EncodeParams + Send + Sync,
    E: Encode + Default + Send + Sync,
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    seq_graph: &G,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    parallel: bool,
    starting_estimator_name: &str,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    let num_symbols = 1 << compression_parameters.max_bits;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    let msg = format!(
        "Pushing symbols into encoder builder with {}...",
        starting_estimator_name
    );

    // Run compression for the first round (sequential or parallel)
    let huffman_graph_encoder_builder = if parallel {
        pl.info(format_args!("Starting the first round in parallel!"));
        parallel_first_reference_selection_round::<EP, E, C, _>(
            seq_graph,
            compression_parameters,
            pl,
            msg,
        )?
    } else {
        pl.info(format_args!("Starting the first round sequentially"));
        pl.start(msg);
        let mut builder =
            HuffmanGraphEncoderBuilder::<_, _, EP>::new(num_symbols, E::default(), C::default());
        let offsets_writer = OffsetsWriter::from_write(io::empty(), true)?;
        match compression_parameters.compressor {
            CompressorType::Approximated { chunk_size } => {
                let mut compressor = BvCompZ::new(
                    &mut builder,
                    offsets_writer,
                    compression_parameters.compression_window,
                    chunk_size,
                    compression_parameters.max_ref_count,
                    compression_parameters.min_interval_length,
                    0,
                );
                for_![ (_, succ) in seq_graph.iter() {
                    compressor.push(succ)?;
                    pl.update();
                }];
                compressor.flush()?;
            }
            CompressorType::Greedy => {
                let mut compressor = BvComp::new(
                    &mut builder,
                    offsets_writer,
                    compression_parameters.compression_window,
                    compression_parameters.max_ref_count,
                    compression_parameters.min_interval_length,
                    0,
                );
                for_![ (_, succ) in seq_graph.iter() {
                    compressor.push(succ)?;
                    pl.update();
                }];
                compressor.flush()?;
            }
        }
        builder
    };
    pl.done();

    if compression_parameters.num_rounds == 1 {
        return write_graph_to_disk(
            &output_basename,
            huffman_graph_encoder_builder,
            seq_graph,
            compression_parameters,
            pl,
        );
    }

    // second round build the graph with the first Huffman estimator
    let mut huffman_graph_encoder_builder = if parallel {
        parallel_reference_selection_round(
            seq_graph,
            huffman_graph_encoder_builder,
            compression_parameters,
            "Pushing symbols into encoder builder on first round with Huffman estimator...",
            pl,
        )?
    } else {
        reference_selection_round(
            seq_graph,
            huffman_graph_encoder_builder,
            compression_parameters,
            "Pushing symbols into encoder builder on first round with Huffman estimator...",
            pl,
        )?
    };

    // execute all the subsequence rounds
    for round in 2..compression_parameters.num_rounds {
        huffman_graph_encoder_builder = if parallel {
            parallel_reference_selection_round(
                seq_graph,
                huffman_graph_encoder_builder,
                compression_parameters,
                format!(
                    "Pushing symbols into encoder builder with Huffman estimator for round {}...",
                    round + 1
                )
                .as_str(),
                pl,
            )?
        } else {
            reference_selection_round(
                seq_graph,
                huffman_graph_encoder_builder,
                compression_parameters,
                format!(
                    "Pushing symbols into encoder builder with Huffman estimator for round {}...",
                    round + 1
                )
                .as_str(),
                pl,
            )?
        };
    }

    write_graph_to_disk(
        &output_basename,
        huffman_graph_encoder_builder,
        seq_graph,
        compression_parameters,
        pl,
    )?;

    Ok(())
}

/// Convert a sequential graph to Huffman-encoded form and save to disk.
/// Runs estimation rounds, builds the encoder and writes the compressed graph with its offsets.
pub fn convert_graph<
    C: ContextModel + Default + Copy + Send + Sync,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    seq_graph: &G,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    parallel: bool,
) -> Result<()> {
    assert!(
        compression_parameters.num_rounds >= 1,
        "Needed at least one estimation round to compress the graph using a Huffman-based encoding."
    );
    let mut pl = concurrent_progress_logger!(
        display_memory = true,
        item_name = "node",
        local_speed = true,
        expected_updates = Some(seq_graph.num_nodes()),
        // log every five minutes
        log_interval = Duration::from_secs(5 * 60),
    );

    match compression_parameters.starting_estimator {
        Estimator::Log2 => {
            run_conversion_rounds::<DefaultEncodeParams, Log2Estimator, C, _>(
                seq_graph,
                &output_basename,
                compression_parameters,
                parallel,
                "Log2Estimator",
                &mut pl,
            )?;
        }
        Estimator::Fixed => {
            run_conversion_rounds::<DefaultEncodeParams, FixedEstimator, C, _>(
                seq_graph,
                &output_basename,
                compression_parameters,
                parallel,
                "FixedEstimator",
                &mut pl,
            )?;
        }
    }

    Ok(())
}

fn write_graph_to_disk<
    EP: EncodeParams,
    E: Encode,
    C: ContextModel,
    G: SequentialGraph + for<'a> SplitLabeling<SplitLender<'a>: ExactSizeLender + Send>,
>(
    output_basename: impl AsRef<Path>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<E, C, EP>,
    seq_graph: &G,
    compression_parameters: &CompressionParameters,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    pl.start("Building the encoder with the cost model obtained from estimation rounds...");

    let output_path = output_basename.as_ref().with_extension(GRAPH_EXTENSION);
    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(BufWriter::new(outfile)));
    let mut writer = CountBitWriter::<LE, _>::new(writer);
    let mut huffman_graph_encoder =
        huffman_graph_encoder_builder.build(&mut writer, compression_parameters.max_bits);

    pl.done();

    pl.info(format_args!("Writing header for the graph..."));
    let header_size = huffman_graph_encoder.write_header()?;
    let offsets_writer = OffsetsWriter::from_path(
        output_basename.as_ref().with_extension(OFFSETS_EXTENSION),
        true,
    )?;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph...");

    match compression_parameters.compressor {
        CompressorType::Approximated { chunk_size } => {
            let mut compressor = BvCompZ::new(
                huffman_graph_encoder,
                offsets_writer,
                compression_parameters.compression_window,
                chunk_size,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in seq_graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
        CompressorType::Greedy => {
            let mut compressor = BvComp::new(
                huffman_graph_encoder,
                offsets_writer,
                compression_parameters.compression_window,
                compression_parameters.max_ref_count,
                compression_parameters.min_interval_length,
                0,
            );
            for_![ (_, succ) in seq_graph.iter() {
                compressor.push(succ)?;
                pl.update();
            }];
            compressor.flush()?;
        }
    }

    pl.info(format_args!(
        "After last round with Huffman estimator: Recompressed graph using {} bits ({} bits of header)",
        writer.bits_written, header_size
    ));

    let properties = compression_parameters
        .to_properties(
            seq_graph.num_nodes(),
            seq_graph
                .num_arcs_hint()
                .expect("Cannot know how many arcs the source graph contains"),
            writer.bits_written as _,
            C::NAME,
        )
        .context("Cannot serialize properties file")?;
    let properties_path = output_basename.as_ref().with_extension("properties");
    std::fs::write(&properties_path, properties)
        .with_context(|| format!("Could not write {}", properties_path.display()))?;

    pl.done();
    Ok(())
}

/// Load a Huffman-compressed graph to be read in sequential order.
/// If the context model or the number of bits passed are different by the ones present
/// in the properties file this function returns an error.  
pub fn load_graph_seq<C: ContextModel + Default + Copy>(
    basename: impl AsRef<Path>,
    max_bits: usize,
) -> Result<BvGraphSeq<SequentialHuffmanDecoderFactory<DefaultEncodeParams, MmapHelper<u32>, C>>> {
    let basename = basename.as_ref();
    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;
    check_compression_parameters(&properties_path, max_bits, C::NAME)?;

    let graph_path = basename.with_extension(GRAPH_EXTENSION);
    let flags = MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::SEQUENTIAL;
    let mmap_factory = MmapHelper::mmap(&graph_path, flags.into())?;
    let factory =
        SequentialHuffmanDecoderFactory::<DefaultEncodeParams, _, _>::new(mmap_factory, max_bits);
    let graph = BvGraphSeq::new(
        factory,
        num_nodes,
        Some(num_arcs),
        comp_flags.compression_window,
        comp_flags.min_interval_length,
    );
    Ok(graph)
}

/// Checks that the compression parameter for statistical encoding are the same between the
/// expected one, and then ones in the properties file used to compress the graph, if presents.
pub fn check_compression_parameters(
    properties_path: impl AsRef<Path>,
    expected_max_bits: usize,
    expected_context_model_name: impl AsRef<str>,
) -> Result<()> {
    let properties_path = properties_path.as_ref();
    let properties_file = BufReader::new(File::open(properties_path)?);
    let name = properties_path.display();
    let properties_map = java_properties::read(properties_file)?;
    if let Some(max_bits) = properties_map.get("maxhuffmanbits") {
        let max_bits = max_bits
            .parse::<usize>()
            .with_context(|| format!("Cannot parse 'maxhuffmanbits' as usize in {}", name))?;
        if max_bits != expected_max_bits {
            return Err(anyhow::anyhow!(
                "Expected maximum length for huffman codewords to be '{}', but have '{}' in {}",
                expected_max_bits,
                max_bits,
                name
            ));
        }
    }
    if let Some(context_model_name) = properties_map.get("contextmodel") {
        if context_model_name != expected_context_model_name.as_ref() {
            return Err(anyhow::anyhow!(
                "Expected context model '{}', but have '{}' in {}",
                expected_context_model_name.as_ref(),
                context_model_name,
                name
            ));
        }
    }
    Ok(())
}

type HuffmanBvGraph<C> = BvGraph<RandomAccessHuffmanDecoderFactory<MmapHelper<u32>, Owned<EF>, C>>;

/// Load a Huffman-compressed graph for random access.
/// Ensures offsets/Elias-Fano exist and returns a `BvGraph`.
pub fn load_graph<C: ContextModel + Default + Copy>(
    basename: impl AsRef<Path>,
    max_bits: usize,
) -> Result<HuffmanBvGraph<C>> {
    let basename = basename.as_ref();
    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;
    check_compression_parameters(&properties_path, max_bits, C::NAME)?;

    let eliasfano_path = basename.with_extension(EF_EXTENSION);
    if !eliasfano_path.exists() {
        let offsets_path = basename.with_extension(OFFSETS_EXTENSION);
        assert!(offsets_path.exists(), "In order to load the graph from random access you should first convert it building the offsets with the mode 'offsets' in the 'graph' command");
        build_eliasfano::<BE>(&basename)
            .context("trying to build elias-fano for the current offsets")?;
        assert!(eliasfano_path.exists());
    }

    let graph_path = basename.with_extension(GRAPH_EXTENSION);
    let flags = MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::RANDOM_ACCESS;
    let mmap_factory = MmapHelper::mmap(&graph_path, flags.into())?;

    let ef = unsafe { EF::load_full(eliasfano_path)? };
    let factory =
        RandomAccessHuffmanDecoderFactory::new(mmap_factory, C::default(), ef.into(), max_bits)?;

    let graph = BvGraph::new(
        factory,
        num_nodes,
        num_arcs,
        comp_flags.compression_window,
        comp_flags.min_interval_length,
    );
    Ok(graph)
}

/// Produce the offsets file from a sequential `BvGraphSeq` by writing gamma deltas.
pub fn build_offsets<F: SequentialDecoderFactory>(
    graph: BvGraphSeq<F>,
    basename: impl AsRef<Path>,
) -> Result<()>
where
    for<'a> F::Decoder<'a>: Decode + BitSeek,
{
    let offsets_path = basename.as_ref().with_extension(OFFSETS_EXTENSION);
    let file = std::fs::File::create(&offsets_path)
        .with_context(|| format!("Could not create {}", offsets_path.display()))?;
    // create a bit writer on the file
    let mut offsets_writer = <BufBitWriter<BE, _>>::new(<WordAdapter<usize, _>>::new(
        BufWriter::with_capacity(1 << 20, file),
    ));

    let mut pl = ProgressLogger::default();
    // log every five minutes
    pl.log_interval(Duration::from_secs(5 * 60));
    pl.start("Start building the offsets...");
    pl.item_name("offset")
        .expected_updates(Some(graph.num_nodes()));

    let mut offset = 0;
    let mut degs_iter = graph.offset_deg_iter();
    for (new_offset, _) in &mut degs_iter {
        // write where
        offsets_writer
            .write_gamma(new_offset - offset)
            .context("Could not write gamma")?;
        pl.light_update();
        offset = new_offset;
    }
    // write the last offset, this is done to avoid decoding the last node
    offsets_writer
        .write_gamma((degs_iter.get_pos() - offset) as _)
        .context("Could not write final gamma")?;
    pl.light_update();
    pl.start("Done!");
    Ok(())
}

/// Decodes a graphs and returns the number of bits used by each component.
pub fn measure_stats<F: SequentialDecoderFactory>(graph: BvGraphSeq<F>) -> GraphStats
where
    for<'a> F::Decoder<'a>: Decode + BitSeek,
{
    let graph = graph.map_factory(stats::StatsDecoderFactory::new);

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("node")
        .expected_updates(Some(graph.num_nodes()));

    pl.start("Scanning...");

    let mut iter = graph.iter();
    while iter.next().is_some() {
        pl.light_update();
    }
    pl.done();

    // drop needed so the graph is no longer borrowed by the iterator
    drop(iter);
    graph.into_inner().stats()
}

/// Compare a Huffman-encoded graph against a BvGraph and return equality results.
/// Returns `Ok(())` when graphs are equal or an `EqError` describing the first mismatch.
pub fn compare_graphs<C: ContextModel + Default + Copy + 'static>(
    first_basename: impl AsRef<Path>,
    second_basename: impl AsRef<Path>,
    max_bits: usize,
) -> Result<Result<(), EqError>> {
    // first the Huffman-encoded graph and then the one in BvGraph format
    let first_graph = load_graph_seq::<C>(&first_basename, max_bits)?;
    let second_graph = BvGraphSeq::with_basename(&second_basename)
        .endianness::<BE>()
        .load()?;

    Ok(eq(&first_graph, &second_graph))
}
