mod component;
pub mod compressors;
mod context_model;
pub mod estimator;
mod huffman_graph_decoder;
mod huffman_graph_encoder;

use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use epserde::deser::{Deserialize, MemCase};
use lender::for_;
use std::io::BufWriter;
use std::time::Duration;
use std::{fs::File, path::PathBuf};
use webgraph::cli::build::ef::{build_eliasfano, CliArgs};
use webgraph::prelude::{SequentialLabeling, *};

use component::*;
pub use compressors::*;
pub use context_model::*;
use estimator::*;
pub use huffman_graph_decoder::*;
use huffman_graph_encoder::*;

use crate::huffman::{DefaultEncodeParams, EncodeParams};

#[allow(clippy::too_many_arguments)]
fn reference_selection_round<
    F: SequentialDecoderFactory,
    EP: EncodeParams,
    E: Encode,
    C: ContextModel + Default,
>(
    graph: &BvGraphSeq<F>,
    huffman_graph_encoder_builder: HuffmanGraphEncoderBuilder<EP, E, C>,
    max_bits: usize,
    compression_parameters: &CompressionParameters,
    msg: &str,
    pl: &mut ProgressLogger,
) -> Result<HuffmanGraphEncoderBuilder<EP, HuffmanEstimator<EP, C>, C>> {
    let num_symbols = 1 << max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<EP, _, _>::new(num_symbols, huffman_estimator, C::default());
    let mut bvcomp = BvCompZ::new(
        &mut huffman_graph_encoder_builder,
        compression_parameters.compression_window,
        10000,
        compression_parameters.max_ref_count,
        compression_parameters.min_interval_length,
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

pub fn convert_graph<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    output_basename: PathBuf,
    max_bits: usize,
    compression_parameters: CompressionParameters,
    build_offsets: bool,
) -> Result<()> {
    assert!(
        compression_parameters.num_rounds >= 1,
        "num_rounds must be at least 1"
    );
    let mut pl = ProgressLogger::default();
    // log every five minutes
    pl.log_interval(Duration::from_secs(5 * 60));
    let num_symbols = 1 << max_bits;

    let seq_graph = BvGraphSeq::with_basename(&basename)
        .endianness::<BE>()
        .load()?;

    // setup for the first iteration with Log2Estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<DefaultEncodeParams, _, _>::new(
            num_symbols,
            Log2Estimator,
            C::default(),
        );
    let mut bvcomp = BvCompZ::new(
        &mut huffman_graph_encoder_builder,
        compression_parameters.compression_window,
        10000,
        compression_parameters.max_ref_count,
        compression_parameters.min_interval_length,
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

    let mut huffman_graph_encoder_builder = reference_selection_round(
        &seq_graph,
        huffman_graph_encoder_builder,
        max_bits,
        &compression_parameters,
        "Pushing symbols into encoder builder on first round...",
        &mut pl,
    )?;
    for round in 1..compression_parameters.num_rounds {
        huffman_graph_encoder_builder = reference_selection_round(
            &seq_graph,
            huffman_graph_encoder_builder,
            max_bits,
            &compression_parameters,
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
    let header_size = huffman_graph_encoder.write_header()?;

    let mut bvcomp = BvCompZ::new(
        &mut huffman_graph_encoder,
        compression_parameters.compression_window,
        10000,
        compression_parameters.max_ref_count,
        compression_parameters.min_interval_length,
        0,
    );

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph...");

    // final round
    if build_offsets {
        let offsets_path = output_basename.with_extension(OFFSETS_EXTENSION);
        let file = std::fs::File::create(&offsets_path)
            .with_context(|| format!("Could not create {}", offsets_path.display()))?;
        // create a bit writer on the file
        let mut offsets_writer = <BufBitWriter<BE, _>>::new(<WordAdapter<usize, _>>::new(
            BufWriter::with_capacity(1 << 20, file),
        ));

        offsets_writer
            .write_gamma(header_size as _)
            .context("Could not write initial delta")?;

        // we should also build the offsets: write the number of bits written for each list compression
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
    bvcomp.flush()?;
    pl.info(format_args!(
        "After second round with Huffman estimator: Recompressed graph using {} bits",
        writer.bits_written
    ));

    pl.done();

    Ok(())
}

pub fn load_graph_seq<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    max_bits: usize,
) -> Result<BvGraphSeq<SequentialHuffmanDecoderFactory<DefaultEncodeParams, MmapHelper<u32>, C>>> {
    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;

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

/// Load an huffman-encoded graph to be accessed in random order
pub fn load_graph<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    max_bits: usize,
) -> Result<BvGraph<RandomAccessHuffmanDecoderFactory<MmapHelper<u32>, EF, C>>> {
    let eliasfano_path = basename.with_extension(EF_EXTENSION);
    if !eliasfano_path.exists() {
        let offsets_path = basename.with_extension(OFFSETS_EXTENSION);
        assert!(offsets_path.exists(), "In order to load the graph from random access you should first convert it building the offsets with the option 'build-offsets' in the 'convert' command");
        build_eliasfano::<LE>(CliArgs {
            src: basename.clone(),
            n: None,
        })
        .context("trying to build elias-fano for the current offsets")?;
        assert!(eliasfano_path.exists());
    }

    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;

    let graph_path = basename.with_extension(GRAPH_EXTENSION);
    let flags = MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::RANDOM_ACCESS;
    let mmap_factory = MmapHelper::mmap(&graph_path, flags.into())?;

    let ef = EF::load_full(eliasfano_path)?;
    let factory = RandomAccessHuffmanDecoderFactory::<_, _, _, DefaultEncodeParams>::new(
        mmap_factory,
        C::default(),
        MemCase::encase(ef),
        max_bits,
    )?;

    let graph = BvGraph::new(
        factory,
        num_nodes,
        num_arcs,
        comp_flags.compression_window,
        comp_flags.min_interval_length,
    );
    Ok(graph)
}

pub fn build_offsets<F: SequentialDecoderFactory>(
    graph: BvGraphSeq<F>,
    basename: PathBuf,
) -> Result<()>
where
    for<'a> F::Decoder<'a>: Decode + BitSeek,
{
    let offsets_path = basename.with_extension(OFFSETS_EXTENSION);
    let file = std::fs::File::create(&offsets_path)
        .with_context(|| format!("Could not create {}", offsets_path.display()))?;
    // create a bit writer on the file
    let mut offsets_writer = <BufBitWriter<LE, _>>::new(<WordAdapter<usize, _>>::new(
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
        pl.update();
        offset = new_offset;
    }
    // write the last offset, this is done to avoid decoding the last node
    offsets_writer
        .write_gamma((degs_iter.get_pos() - offset) as _)
        .context("Could not write final gamma")?;
    pl.start("Done!");
    Ok(())
}
