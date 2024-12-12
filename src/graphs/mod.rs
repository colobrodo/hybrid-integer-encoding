mod component;
pub mod estimator;
mod huffman_graph_encoder;
mod huffman_graph_encoder_builder;

use anyhow::Result;
use dsi_bitstream::prelude::{BufBitWriter, MemWordWriterVec};
use dsi_bitstream::traits::{BE, LE};
use dsi_progress_logger::prelude::*;
use lender::for_;
use std::path::PathBuf;
use webgraph::prelude::{SequentialLabeling, *};

use component::*;
pub use estimator::*;
use huffman_graph_encoder::*;
pub use huffman_graph_encoder_builder::*;

use crate::huffman::{DefaultEncodeParams, EncodeParams};
use crate::utils::StatBitWriter;

#[allow(clippy::too_many_arguments)]
fn referece_selection_round<F: SequentialDecoderFactory, EP: EncodeParams, E: Encode>(
    graph: &BvGraphSeq<F>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<EP, E>,
    max_bits: usize,
    compression_window: usize,
    max_ref_count: usize,
    min_interval_length: usize,
    msg: &str,
    pl: &mut ProgressLogger,
) -> Result<HuffmanGraphEncoderBuilder<EP, HuffmanEstimator<EP>>> {
    let num_symbols = 1 << max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<EP, _>::new(num_symbols, huffman_estimator);
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
    max_bits: usize,
    compression_window: usize,
    max_ref_count: usize,
    min_interval_length: usize,
    num_rounds: usize,
) -> Result<()> {
    assert!(num_rounds >= 1, "num_rounds must be at least 1");
    let mut pl = ProgressLogger::default();
    let num_symbols = 1 << max_bits;

    let seq_graph = BvGraphSeq::with_basename(&basename)
        .endianness::<BE>()
        .load()?;

    // setup for the first iteration with Log2Estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<DefaultEncodeParams, _>::new(num_symbols, Log2Estimator);
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

    let word_write = MemWordWriterVec::new(Vec::<u64>::new());
    let writer = BufBitWriter::<LE, _>::new(word_write);
    let mut writer = StatBitWriter::new(writer);
    let mut huffman_graph_encoder = huffman_graph_encoder_builder.build(&mut writer, max_bits);
    pl.done();

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
    for_![ (_, succ) in seq_graph {
        bvcomp.push(succ)?;
        pl.update();
    }];
    pl.done();

    println!(
        "After second round with Huffman estimator: Recompressed graph using {} bits",
        writer.written_bits
    );

    Ok(())
}
