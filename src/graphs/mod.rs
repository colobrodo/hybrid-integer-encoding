mod component;
pub mod compressors;
mod context_model;
pub mod estimator;
mod huffman_graph_decoder;
mod huffman_graph_encoder;
mod stats;

use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use epserde::deser::{Deserialize, Owned};
use epserde::prelude::*;
use lender::*;
use mmap_rs::MmapFlags;
use std::io::{BufReader, BufWriter, Seek};
use std::path::Path;
use std::time::Duration;
use std::{fs::File, path::PathBuf};
use sux::prelude::*;
use webgraph::prelude::{SequentialLabeling, *};

use component::*;
pub use compressors::*;
pub use context_model::*;
use estimator::*;
pub use huffman_graph_decoder::*;
use huffman_graph_encoder::*;
pub use stats::*;

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

fn build_eliasfano<E: Endianness + 'static>(src: &Path) -> Result<()>
where
    for<'a> BufBitReader<E, MemWordReader<u32, &'a [u32]>>: CodesRead<E> + BitSeek,
{
    let mut pl = ProgressLogger::default();
    pl.display_memory(true).item_name("offset");

    // Creates the offsets file
    let of_file_path = src.with_extension(OFFSETS_EXTENSION);

    let graph_path = src.with_extension(GRAPH_EXTENSION);
    log::info!("Getting size of graph at '{}'", graph_path.display());
    let mut file = File::open(&graph_path)
        .with_context(|| format!("Could not open {}", graph_path.display()))?;
    let file_len = 8 * file
        .seek(std::io::SeekFrom::End(0))
        .with_context(|| format!("Could not seek in {}", graph_path.display()))?;
    log::info!("Graph file size: {} bits", file_len);

    // if the num_of_nodes is not present, read it from the properties file
    // otherwise use the provided value, this is so we can build the Elias-Fano
    // for offsets of any custom format that might not use the standard
    // properties file
    let num_nodes = {
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
        map.get("nodes").unwrap().parse::<usize>()?
    };
    pl.expected_updates(Some(num_nodes));

    let mut efb = EliasFanoBuilder::new(num_nodes + 1, file_len as usize);

    log::info!("Checking if offsets exists at '{}'", of_file_path.display());
    // if the offset files exists, read it to build elias-fano
    assert!(of_file_path.exists(), "The offsets doesn't file exists");
    let of = <MmapHelper<u32>>::mmap(of_file_path, MmapFlags::SEQUENTIAL)?;
    build_eliasfano_from_offsets::<E>(num_nodes, of.new_reader(), &mut pl, &mut efb)?;

    serialize_eliasfano(src, efb, &mut pl)
}

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

pub fn serialize_eliasfano(
    src: &Path,
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

    let ef_path = src.with_extension(EF_EXTENSION);
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

pub fn convert_graph<C: ContextModel + Default + Copy>(
    basename: &Path,
    output_basename: &Path,
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
        basename,
        output_basename,
        &compression_parameters,
        C::NAME,
        max_bits,
    )
    .expect("Cannot copy the properties file");

    pl.done();

    Ok(())
}

pub fn load_graph_seq<C: ContextModel + Default + Copy>(
    basename: &Path,
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
                expected_max_bits,
                max_bits,
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
) -> Result<BvGraph<RandomAccessHuffmanDecoderFactory<MmapHelper<u32>, Owned<EF>, C>>> {
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
    println!("graph file size: {} bytes", graph_path.metadata()?.len());
    let flags = MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::RANDOM_ACCESS;
    let mmap_factory = MmapHelper::mmap(&graph_path, flags.into())?;

    let ef = unsafe { EF::load_full(eliasfano_path)? };
    let factory = RandomAccessHuffmanDecoderFactory::<_, _, _, DefaultEncodeParams>::new(
        mmap_factory,
        C::default(),
        ef.into(),
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
    basename: &Path,
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

pub enum ComparisonResult {
    Equal,
    Different {
        node_id: usize,
        right_succs: Vec<usize>,
        left_succs: Vec<usize>,
    },
}

/// Compare two graphs, the first encoded using Huffman while the second is expected in the BvGraph format.
/// It returns a `ComparisonResult` that is Equal if both the graphs are equals otherwise returns which
/// is the first different node with the two different successors lists
pub fn compare_graphs<C: ContextModel + Default + Copy>(
    first_basename: PathBuf,
    second_basename: PathBuf,
    max_bits: usize,
) -> Result<ComparisonResult> {
    // first the Huffman-encoded graph and then the one in BvGraph format
    let first_graph = load_graph_seq::<C>(&first_basename, max_bits)?;
    let second_graph = BvGraphSeq::with_basename(second_basename.clone())
        .endianness::<BE>()
        .load()?;

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("compare graphs")
        .expected_updates(Some(second_graph.num_nodes()));

    let mut original_iter = second_graph.iter().enumerate();
    let mut iter = first_graph.iter();

    pl.start("Start comparing the graphs...");
    while let Some((i, (true_node_id, true_succ))) = original_iter.next() {
        let (node_id, succ) = iter.next().unwrap();

        assert_eq!(true_node_id, i);
        assert_eq!(true_node_id, node_id);
        let true_succs = true_succ.into_iter().collect::<Vec<_>>();
        let succs = succ.into_iter().collect::<Vec<_>>();
        if true_succs != succs {
            return Ok(ComparisonResult::Different {
                node_id,
                right_succs: true_succs,
                left_succs: succs,
            });
        }
        pl.light_update();
    }
    pl.done();
    Ok(ComparisonResult::Equal)
}
