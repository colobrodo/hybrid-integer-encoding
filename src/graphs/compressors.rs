use anyhow::Result;
use webgraph::prelude::{BvComp, BvCompZ, EncodeAndEstimate, GraphCompressor};

/// Parameters used to configure the compression process.
pub struct CompressionParameters {
    /// The window size in which chose the next reference.
    pub compression_window: usize,
    /// The maximum length of references' chain allowed during compression.
    pub max_ref_count: usize,
    /// The minimum length of consecutive items to be encoded as intervals during compression.
    pub min_interval_length: usize,
    /// The number of compression rounds to perform.
    pub num_rounds: usize,
}

impl CompressionParameters {
    // sadly somehow duplicated from webgraph's CompFlags
    pub fn to_properties(
        &self,
        num_nodes: usize,
        num_arcs: u64,
        bitstream_len: u64,
        context_model_name: impl AsRef<str>,
        max_bits: usize,
    ) -> Result<String> {
        let mut s = String::new();
        s.push_str(&format!("nodes={num_nodes}\n"));
        s.push_str(&format!("arcs={num_arcs}\n"));
        s.push_str(&format!("minintervallength={}\n", self.min_interval_length));
        s.push_str(&format!("maxrefcount={}\n", self.max_ref_count));
        s.push_str(&format!("windowsize={}\n", self.compression_window));
        s.push_str(&format!(
            "bitsperlink={}\n",
            bitstream_len as f64 / num_arcs as f64
        ));
        s.push_str(&format!(
            "bitspernode={}\n",
            bitstream_len as f64 / num_nodes as f64
        ));
        s.push_str(&format!("length={bitstream_len}\n"));
        s.push_str(&format!("length={bitstream_len}\n"));
        let context_model_name = context_model_name.as_ref();
        s.push_str(&format!("contextmodel={context_model_name}\n"));
        s.push_str(&format!("maxhuffmanbits={max_bits}\n"));
        Ok(s)
    }
}

/// A trait for creating compressors from an encoder.
///
/// This trait defines a factory-like interface for creating objects that implement
/// the `GraphCompressor` trait. The compressor is initialized using an encoder
/// (which implements `EncodeAndEstimate`) and a set of compression parameters.
pub trait CompressorFromEncoder {
    /// Creates a new compressor instance using the provided encoder and compression parameters.
    fn create_from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor;
}

/// A factory for creating `BvComp` (greedy) compressors.
///
/// This struct provides an implementation of the `CompressorFromEncoder` trait
/// for creating `BvComp` instances, which are used for basic compression.
pub struct CreateBvComp;

impl CompressorFromEncoder for CreateBvComp {
    /// Creates a new `BvComp` compressor using the provided encoder and parameters.
    fn create_from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor {
        // Create a new BvComp instance with the provided parameters.
        BvComp::new(
            encoder,
            parameters.compression_window,
            parameters.max_ref_count,
            parameters.min_interval_length,
            0,
        )
    }
}

/// A factory for creating `BvCompZ` compressors with configurable chunk sizes.
///
/// This struct provides an implementation of the `CompressorFromEncoder` trait
/// for creating `BvCompZ` instances, approximated reference selection algorithm, as
/// described in Zuckerli, which works with chunk-based compression.
pub struct CreateBvCompZ {
    // The size of chunks used during compression.
    chunk_size: usize,
}

impl CreateBvCompZ {
    /// Creates a new `BvCompZCreate` instance with the specified chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> CreateBvCompZ {
        CreateBvCompZ { chunk_size }
    }
}

impl CompressorFromEncoder for CreateBvCompZ {
    /// Creates a new `BvCompZ` compressor using the provided encoder and parameters.
    fn create_from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor {
        // Create a new BvCompZ instance with the provided parameters and chunk size.
        BvCompZ::new(
            encoder,
            parameters.compression_window,
            self.chunk_size,
            parameters.max_ref_count,
            parameters.min_interval_length,
            0,
        )
    }
}
