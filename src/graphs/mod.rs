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
use std::io::{BufReader, BufWriter};
use std::path::Path;
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
    create_compressor: &impl CompressorFromEncoder,
    pl: &mut ProgressLogger,
) -> Result<HuffmanGraphEncoderBuilder<EP, HuffmanEstimator<EP, C>, C>> {
    let num_symbols = 1 << max_bits;
    let huffman_estimator = huffman_graph_encoder_builder.build_estimator();
    // setup for the new iteration with huffman estimator
    let mut huffman_graph_encoder_builder =
        HuffmanGraphEncoderBuilder::<EP, _, _>::new(num_symbols, huffman_estimator, C::default());
    let mut compressor = create_compressor
        .create_from_encoder(&mut huffman_graph_encoder_builder, compression_parameters);

    pl.item_name("node")
        .expected_updates(Some(graph.num_nodes()));
    pl.start(msg);
    for_![ (_, succ) in graph {
        compressor.push(succ)?;
        pl.update();
    }];
    compressor.flush()?;
    pl.done();
    Ok(huffman_graph_encoder_builder)
}

/// Copy the original properties file and update the compression parameters used for the new graph
fn copy_properties_file(
    basename: &Path,
    destination_basename: &Path,
    compression_parameters: &CompressionParameters,
    context_model_name: &str,
    max_bits: usize,
) -> Result<()> {
    let properties_path = basename.with_extension("properties");
    let temp_properties_path = destination_basename.with_extension("properties");
    let properties_file = BufReader::new(File::open(&properties_path)?);
    let mut properties_map = java_properties::read(properties_file)?;
    // Override the properties with passed compression parameters
    properties_map.insert(
        "windowsize".into(),
        compression_parameters.compression_window.to_string(),
    );
    properties_map.insert(
        "maxrefcount".into(),
        compression_parameters.max_ref_count.to_string(),
    );
    properties_map.insert(
        "minintervallength".into(),
        compression_parameters.min_interval_length.to_string(),
    );
    properties_map.insert("maxhuffmanbits".into(), max_bits.to_string());
    properties_map.insert("contextmodel".into(), context_model_name.to_string());

    let new_properties_file = BufWriter::new(File::create(&temp_properties_path)?);
    java_properties::write(new_properties_file, &properties_map)?;
    Ok(())
}

pub fn convert_graph<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    output_basename: PathBuf,
    max_bits: usize,
    create_compressor: impl CompressorFromEncoder,
    compression_parameters: CompressionParameters,
) -> Result<()> {
    assert!(
        compression_parameters.num_rounds >= 1,
        "num_rounds must be at least 1"
    );
    let mut pl = progress_logger!(
        display_memory = true,
        log_interval = Duration::from_secs(5 * 60)
    );
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
    let mut compressor = create_compressor
        .create_from_encoder(&mut huffman_graph_encoder_builder, &compression_parameters);

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Pushing symbols into encoder builder with Log2Estimator...");

    // first iteration: build a encoder with Log2Estimator
    for_![ (_, succ) in seq_graph {
        compressor.push(succ)?;
        pl.update();
    }];
    compressor.flush()?;
    pl.done();

    let mut huffman_graph_encoder_builder = reference_selection_round(
        &seq_graph,
        huffman_graph_encoder_builder,
        max_bits,
        &compression_parameters,
        "Pushing symbols into encoder builder on first round with Huffman estimator...",
        &create_compressor,
        &mut pl,
    )?;
    for round in 2..compression_parameters.num_rounds {
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
            &create_compressor,
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

    let mut compressor =
        create_compressor.create_from_encoder(&mut huffman_graph_encoder, &compression_parameters);

    pl.item_name("node")
        .expected_updates(Some(seq_graph.num_nodes()));
    pl.start("Compressing the graph...");

    for_![ (_, successors) in seq_graph {
        compressor.push(successors).context("Could not push successors")?;
        pl.update();
    }];
    compressor.flush()?;
    pl.info(format_args!(
        "After last round with Huffman estimator: Recompressed graph using {} bits ({} bits of header)",
        writer.bits_written, header_size
    ));

    copy_properties_file(
        &basename,
        &output_basename,
        &compression_parameters,
        C::NAME,
        max_bits,
    )
    .expect("Cannot copy the properties file");

    pl.done();

    Ok(())
}

pub fn load_graph_seq<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    max_bits: usize,
) -> Result<BvGraphSeq<SequentialHuffmanDecoderFactory<DefaultEncodeParams, MmapHelper<u32>, C>>> {
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
/// expected one, and then ones in the properties file used to compress the graph, if presents
fn check_compression_parameters(
    properties_path: &Path,
    expected_max_bits: usize,
    expected_context_model_name: &str,
) -> Result<()> {
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
                expected_max_bits.to_string(),
                max_bits.to_string(),
                name
            ));
        }
    }
    if let Some(context_model_name) = properties_map.get("contextmodel") {
        if context_model_name != expected_context_model_name {
            return Err(anyhow::anyhow!(
                "Expected context model '{}', but have '{}' in {}",
                expected_context_model_name,
                context_model_name,
                name
            ));
        }
    }
    Ok(())
}

/// Load an Huffman-encoded graph to be accessed in random order
pub fn load_graph<C: ContextModel + Default + Copy>(
    basename: PathBuf,
    max_bits: usize,
) -> Result<BvGraph<RandomAccessHuffmanDecoderFactory<MmapHelper<u32>, EF, C>>> {
    let properties_path = basename.with_extension(PROPERTIES_EXTENSION);
    let (num_nodes, num_arcs, comp_flags) = parse_properties::<BE>(&properties_path)?;
    check_compression_parameters(&properties_path, max_bits, C::NAME)?;

    let eliasfano_path = basename.with_extension(EF_EXTENSION);
    if !eliasfano_path.exists() {
        let offsets_path = basename.with_extension(OFFSETS_EXTENSION);
        assert!(offsets_path.exists(), "In order to load the graph from random access you should first convert it building the offsets with the mode 'offsets' in the 'graph' command");
        build_eliasfano::<LE>(CliArgs {
            src: basename.clone(),
            n: None,
        })
        .context("trying to build elias-fano for the current offsets")?;
        assert!(eliasfano_path.exists());
    }

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
