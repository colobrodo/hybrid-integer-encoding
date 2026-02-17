mod component;
mod context_model;
mod convert;
pub mod estimator;
mod huffman_graph_decoder;
mod huffman_graph_encoder;
mod offsets;
pub mod parameters;
mod stats;
mod utils;

use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use epserde::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use webgraph::prelude::*;

use component::*;
pub use context_model::*;
pub use convert::*;
use estimator::*;
pub use huffman_graph_decoder::*;
pub use offsets::*;
pub use parameters::*;
pub use stats::*;
pub use utils::*;

use crate::huffman::DefaultEncodeParams;

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

type HuffmanBvGraph<C> = BvGraph<RandomAccessHuffmanDecoderFactory<MmapHelper<u32>, EF, C>>;

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

    let ef = unsafe { EF::mmap(eliasfano_path, flags.into())? };
    let factory = RandomAccessHuffmanDecoderFactory::new(mmap_factory, C::default(), ef, max_bits)?;

    let graph = BvGraph::new(
        factory,
        num_nodes,
        num_arcs,
        comp_flags.compression_window,
        comp_flags.min_interval_length,
    );
    Ok(graph)
}
