mod component;
mod context_choice_strategy;
pub mod decoder_factories;
pub mod estimator;
mod huffman_graph_decoder;
mod huffman_graph_encoder;
mod huffman_graph_encoder_builder;

use anyhow::{Context, Result};
use dsi_bitstream::codes::GammaWrite;
use dsi_bitstream::impls::WordAdapter;
use dsi_bitstream::prelude::BufBitWriter;
use dsi_bitstream::traits::{BE, LE};
use dsi_bitstream::utils::CountBitWriter;
use dsi_progress_logger::prelude::*;
use lender::for_;
use std::io::BufWriter;
use std::{fs::File, path::PathBuf};
use webgraph::prelude::{SequentialLabeling, *};

use component::*;
use context_choice_strategy::*;
pub use decoder_factories::*;
use estimator::*;
pub use huffman_graph_decoder::*;
use huffman_graph_encoder::*;
pub use huffman_graph_encoder_builder::*;

use crate::huffman::{DefaultEncodeParams, EncodeParams};

#[allow(clippy::too_many_arguments)]
fn referece_selection_round<
    F: SequentialDecoderFactory,
    EP: EncodeParams,
    E: Encode,
    S: ContextChoiceStrategy,
>(
    graph: &BvGraphSeq<F>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<EP, E, S>,
    max_bits: usize,
    compression_window: usize,
    max_ref_count: usize,
    min_interval_length: usize,
    msg: &str,
    pl: &mut ProgressLogger,
) -> Result<HuffmanGraphEncoderBuilder<EP, HuffmanEstimator<EP, S>, SimpleChoiceStrategy>> {
    let num_symbols = 1 << max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder = HuffmanGraphEncoderBuilder::<EP, _, _>::new(
        num_symbols,
        huffman_estimator,
        SimpleChoiceStrategy,
    );
    let mut bvcomp = BvComp::new(
        &mut huffman_graph_encoder_builder,
        compression_window,
        max_ref_count,
        min_interval_length,
        0,
    );

    pl.item_name("node")
        .expected_updates(Some(graph.num_nodes()));
    pl.start(msg);
    for_![ (_, succ) in graph {
        bvcomp.push(succ)?;
        pl.update();
    }];
    bvcomp.flush()?;
    pl.done();
    Ok(huffman_graph_encoder_builder)
}

pub fn convert_graph(
    basename: PathBuf,
    output_basename: PathBuf,
    max_bits: usize,
    compression_window: usize,
    max_ref_count: usize,
    min_interval_length: usize,
    num_rounds: usize,
    build_offsets: bool,
) -> Result<()> {
    assert!(num_rounds >= 1, "num_rounds must be at least 1");
    let mut pl = ProgressLogger::default();
    let num_symbols = 1 << max_bits;

    let seq_graph = BvGraphSeq::with_basename(&basename)
        .endianness::<BE>()
        .load()?;

    // setup for the first iteration with Log2Estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<DefaultEncodeParams, _, _>::new(
            num_symbols,
            Log2Estimator,
            SimpleChoiceStrategy,
        );
    let mut bvcomp = BvComp::new(
        &mut huffman_graph_encoder_builder,
        compression_window,
        max_ref_count,
        min_interval_length,
        0,
    );

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Pushing symbols into encoder builder with Log2Estimator...");

    // first iteration: build a encoder with Log2Estimator
    for_![ (_, succ) in seq_graph {
        bvcomp.push(succ)?;
        pl.update();
    }];
    bvcomp.flush()?;
    pl.done();

    let mut huffman_graph_encoder_builder = referece_selection_round(
        &seq_graph,
        huffman_graph_encoder_builder,
        max_bits,
        compression_window,
        max_ref_count,
        min_interval_length,
        "Pushing symbols into encoder builder on first round...",
        &mut pl,
    )?;
    for round in 1..num_rounds {
        huffman_graph_encoder_builder = referece_selection_round(
            &seq_graph,
            huffman_graph_encoder_builder,
            max_bits,
            compression_window,
            max_ref_count,
            min_interval_length,
            format!(
                "Pushing symbols into encoder builder with Huffman estimator for round {}...",
                round + 1
            )
            .as_str(),
            &mut pl,
        )?;
    }

    pl.start("Building the encoder after estimation rounds...");

    let output_path = output_basename.with_extension(GRAPH_EXTENSION);
    let outfile = File::create(output_path)?;
    let writer = BufBitWriter::<LE, _>::new(WordAdapter::<u32, _>::new(BufWriter::new(outfile)));
    let mut writer = CountBitWriter::<LE, _>::new(writer);
    let mut huffman_graph_encoder = huffman_graph_encoder_builder.build(&mut writer, max_bits);

    pl.done();

    pl.info(format_args!("Writing header for the graph..."));
    huffman_graph_encoder.write_header()?;

    let mut bvcomp = BvComp::new(
        &mut huffman_graph_encoder,
        compression_window,
        max_ref_count,
        min_interval_length,
        0,
    );

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph...");

    // second iteration: build a model with the entropy mock writer
    if build_offsets {
        let offsets_path = output_basename.with_extension(OFFSETS_EXTENSION);
        let file = std::fs::File::create(&offsets_path)
            .with_context(|| format!("Could not create {}", offsets_path.display()))?;
        // create a bit writer on the file
        let mut offsets_writer = <BufBitWriter<LE, _>>::new(<WordAdapter<usize, _>>::new(
            BufWriter::with_capacity(1 << 20, file),
        ));

        offsets_writer
            .write_gamma(0)
            .context("Could not write initial delta")?;

        for_! [ (_, successors) in seq_graph {
            let delta = bvcomp.push(successors).context("Could not push successors")?;
            offsets_writer.write_gamma(delta).context("Could not write delta")?;
            pl.update();
        }];
    } else {
        for_![ (_, successors) in seq_graph {
            bvcomp.push(successors).context("Could not push successors")?;
            pl.update();
        }];
    }
    pl.done();

    println!(
        "After second round with Huffman estimator: Recompressed graph using {} bits",
        writer.bits_written
    );

    Ok(())
}

pub fn load_graph_seq(
    basename: PathBuf,
    max_bits: usize,
) -> Result<
    BvGraphSeq<
        HuffmanGraphDecoderFactory<DefaultEncodeParams, LE, FileFactory<LE>, SimpleChoiceStrategy>,
    >,
> {
    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;
    let graph_path = basename.with_extension(GRAPH_EXTENSION);

    let file_factory = FileFactory::<LE>::new(graph_path)?;
    let factory =
        HuffmanGraphDecoderFactory::<DefaultEncodeParams, _, _, _>::new(file_factory, max_bits);
    let graph = BvGraphSeq::new(
        factory,
        num_nodes,
        Some(num_arcs),
        comp_flags.compression_window,
        comp_flags.min_interval_length,
    );
    Ok(graph)
}
