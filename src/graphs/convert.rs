use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{concurrent_progress_logger, prelude::*};
use lender::*;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::Path;
use std::time::Duration;

use webgraph::prelude::*;

use crate::huffman::*;

use super::estimator::{FixedEstimator, HuffmanEstimator, Log2Estimator};
use super::huffman_graph_encoder::HuffmanGraphEncoderBuilder;
use super::CompressorType;
use super::Estimator;
use super::{CompressionParameters, ContextModel};

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
    let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;
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
    num_threads: usize,
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
    let num_symbols = 1 << compression_parameters.max_bits;
    let split_iter = graph
        .split_iter(num_threads)
        .into_iter()
        .collect::<Vec<_>>();

    cpl.start(msg);

    // iterate the splitted version of the graph in parallel
    let thread_histograms: Vec<IntegerHistograms<EP>> = split_iter
        .into_iter()
        .enumerate()
        .par_bridge()
        .map_with(
            cpl.clone(),
            |pl, (thread_id, mut thread_lender)| -> Result<IntegerHistograms<EP>> {
                pl.info(format_args!(
                    "Started compression with thread {}",
                    thread_id
                ));

                let Some((node_id, successors)) = thread_lender.next() else {
                    return Err(anyhow::anyhow!(
                        "Empty chunked size of compressors in thread {}",
                        thread_id
                    ));
                };

                let first_node = node_id;

                // Initialize local builder with the estimator from factory
                let mut thread_builder = HuffmanGraphEncoderBuilder::<_, _, EP>::new(
                    num_symbols,
                    factory.create_estimator(),
                    C::default(),
                );
                let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;

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
    num_threads: usize,
    pl: &mut ConcurrentWrapper,
    msg: impl AsRef<str>,
) -> Result<HuffmanGraphEncoderBuilder<E, C, EP>> {
    let factory = DefaultEstimatorFactory::<E>::default();
    let shared_histograms = parallel_compression_round_helper::<EP, E, C, G, _>(
        graph,
        compression_parameters,
        &factory,
        num_threads,
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
    num_threads: usize,
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
        num_threads,
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

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
pub fn sequential_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
) -> Result<()> {
    convert_graph_file::<C>(basename, output_basename, compression_parameters, 1)
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph running the estimation rounds in parallel.
/// The converted graph is written to `output_basename`.
pub fn parallel_convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
) -> Result<()> {
    convert_graph_file::<C>(
        basename,
        output_basename,
        compression_parameters,
        num_threads,
    )
}

/// Read a BVGraph from `basename` and convert it to a Huffman-encoded graph.
/// The converted graph is written to `output_basename`.
/// The `parallel` arguments controls if the estimation rounds are executed in parallel or sequentially.
pub fn convert_graph_file<C: ContextModel + Default + Copy + Send + Sync>(
    basename: impl AsRef<Path>,
    output_basename: impl AsRef<Path>,
    compression_parameters: &CompressionParameters,
    num_threads: usize,
) -> Result<()> {
    if basename.as_ref().with_extension(EF_EXTENSION).exists() {
        let graph = BvGraph::with_basename(&basename)
            .endianness::<BE>()
            .load()?;

        convert_graph::<C, _>(&graph, output_basename, compression_parameters, num_threads)
    } else {
        let seq_graph = BvGraphSeq::with_basename(&basename)
            .endianness::<BE>()
            .load()?;

        convert_graph::<C, _>(
            &seq_graph,
            output_basename,
            compression_parameters,
            num_threads,
        )
    }
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
    num_threads: usize,
    starting_estimator_name: &str,
    pl: &mut ConcurrentWrapper,
) -> Result<()> {
    let num_symbols = 1 << compression_parameters.max_bits;

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start(format!(
        "Pushing symbols into encoder builder with {}...",
        starting_estimator_name
    ));

    // Run compression for the first round (sequential or parallel)
    let huffman_graph_encoder_builder = if num_threads > 1 {
        parallel_first_reference_selection_round::<EP, E, C, _>(
            seq_graph,
            compression_parameters,
            num_threads,
            pl,
            "",
        )?
    } else {
        let mut builder =
            HuffmanGraphEncoderBuilder::<_, _, EP>::new(num_symbols, E::default(), C::default());
        let offsets_writer = OffsetsWriter::from_write(io::empty(), false)?;
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
    let mut huffman_graph_encoder_builder = if num_threads > 1 {
        parallel_reference_selection_round(
            seq_graph,
            huffman_graph_encoder_builder,
            compression_parameters,
            num_threads,
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
        huffman_graph_encoder_builder = if num_threads > 1 {
            parallel_reference_selection_round(
                seq_graph,
                huffman_graph_encoder_builder,
                compression_parameters,
                num_threads,
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
    num_threads: usize,
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
                num_threads,
                "Log2Estimator",
                &mut pl,
            )?;
        }
        Estimator::Fixed => {
            run_conversion_rounds::<DefaultEncodeParams, FixedEstimator, C, _>(
                seq_graph,
                &output_basename,
                compression_parameters,
                num_threads,
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
        false,
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
