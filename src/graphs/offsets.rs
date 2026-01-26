use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use epserde::prelude::*;
use mmap_rs::MmapFlags;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek};
use std::path::Path;
use std::time::Duration;
use sux::prelude::*;
use webgraph::prelude::{SequentialLabeling, *};

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

/// Build the Elias-Fano index for a graph file at `src` using offsets.
/// Reads the graph file size and constructs the EF index on disk.
pub(crate) fn build_eliasfano<E: Endianness + 'static>(src: impl AsRef<Path>) -> Result<()>
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
