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

/// A trait for creating compressors from an encoder.
///
/// This trait defines a factory-like interface for creating objects that implement
/// the `GraphCompressor` trait. The compressor is initialized using an encoder
/// (which implements `EncodeAndEstimate`) and a set of compression parameters.
pub trait CompressorFromEncoder {
    /// Creates a new compressor instance using the provided encoder and compression parameters.
    fn create_compressor(
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
    fn create_compressor(
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
    fn create_compressor(
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
