// The functions in this module are provided by the WebGraph CLI, but they
// should be duplicated here with slight modifications because of the Huffman encoding.
// Currently, compare_graph is equivalent to `webgraph::traits::eq`, so we can use this
// function, but unfortunately, it does not provide any indication of the progress made
// during the comparison.
// `measure_stats` instead, has no direct mapping with webgraph it report the number of
// bits used to encode each component of the graph.

use super::*;
use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use epserde::deser::Deserialize;
use epserde::prelude::*;
use lender::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use sux::traits::IndexedSeq;
use webgraph::prelude::{SequentialLabeling, *};

/// Compare the `ef` file against the `offsets` file, and if it is equal also against
/// the original graph.
/// If some differences are found it prints an error to the stderr and exits.
pub fn check_ef<C: ContextModel + Default + Copy + 'static>(
    basename: impl AsRef<Path>,
    max_bits: usize,
) -> Result<()> {
    let properties_path = basename.as_ref().with_extension(PROPERTIES_EXTENSION);
    let f = File::open(&properties_path).with_context(|| {
        format!(
            "Could not load properties file: {}",
            properties_path.display()
        )
    })?;
    let map = java_properties::read(BufReader::new(f))?;
    let num_nodes = map.get("nodes").unwrap().parse::<usize>()?;

    // Creates the offsets file
    let of_file_path = basename.as_ref().with_extension(OFFSETS_EXTENSION);

    let ef = unsafe {
        EF::mmap(
            basename.as_ref().with_extension(EF_EXTENSION),
            Flags::default(),
        )
    }?;
    let ef = ef.uncase();

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("offset")
        .expected_updates(Some(num_nodes));

    // if the offset files exists, read it to build elias-fano
    if of_file_path.exists() {
        // create a bit reader on the file
        let mut reader = buf_bit_reader::from_path::<BE, u32>(of_file_path)?;
        // progress bar
        pl.start("Checking offsets file against Elias-Fano...");
        // read the graph a write the offsets
        let mut offset = 0;
        for node_id in 0..num_nodes + 1 {
            // write where
            offset += reader.read_gamma()?;
            // read ef
            let ef_res = ef.get(node_id as _);
            assert_eq!(offset, ef_res as u64, "node_id: {}", node_id);
            // decode the next nodes so we know where the next node_id starts
            pl.light_update();
        }
    } else {
        pl.info(format_args!(
            "No offsets file, checking against graph file only"
        ));
    }

    // progress bar
    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("offset")
        .expected_updates(Some(num_nodes));

    pl.start("Checking graph against Elias-Fano...");

    // otherwise directly read the graph
    let graph = load_graph_seq::<C>(&basename, max_bits)?;
    // read the graph a write the offsets
    for (node, (new_offset, _degree)) in graph.offset_deg_iter().enumerate() {
        // decode the next nodes so we know where the next node_id starts
        // read ef
        let ef_res = ef.get(node as _);
        assert_eq!(new_offset, ef_res as u64, "node_id: {}", node);
        pl.light_update();
    }
    pl.done();
    Ok(())
}

/// Decodes a graphs and returns the number of bits used by each component.
pub fn measure_stats<F: SequentialDecoderFactory>(graph: BvGraphSeq<F>) -> GraphStats
where
    for<'a> F::Decoder<'a>: Decode + BitSeek,
{
    let graph = graph.map_factory(super::stats::StatsDecoderFactory::new);

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

    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("compare graphs")
        .expected_updates(Some(first_graph.num_nodes()));

    pl.start("Start comparing the graphs...");

    if first_graph.num_nodes() != second_graph.num_nodes() {
        return Ok(Err(EqError::NumNodes {
            first: first_graph.num_nodes(),
            second: second_graph.num_nodes(),
        }));
    }
    for_!(((node0, succ0), (node1, succ1)) in first_graph.iter().zip(second_graph.iter()) {
        debug_assert_eq!(node0, node1);
        pl.light_update();
        let mut succ0 = succ0.into_iter().collect::<Vec<_>>();
        let mut succ1 = succ1.into_iter().collect::<Vec<_>>();
        succ0.sort();
        succ1.sort();
        let result = labels::eq_succs(node0, succ0, succ1);
        if let Err(eq_error) = result {
            return Ok(Err(eq_error))
        }
    });

    pl.done();
    Ok(Ok(()))
}
