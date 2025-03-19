use webgraph::prelude::{BvComp, BvCompZ, EncodeAndEstimate, GraphCompressor};

/// Parameters used to configure the compression process.
///
/// # Fields
/// - `compression_window`: The size of the sliding window used during compression.
/// - `max_ref_count`: The maximum number of references allowed during compression.
/// - `min_interval_length`: The minimum length of intervals to be considered for compression.
/// - `num_rounds`: The number of compression rounds to perform.
pub struct CompressionParameters {
    pub compression_window: usize,
    pub max_ref_count: usize,
    pub min_interval_length: usize,
    pub num_rounds: usize,
}

/// A trait for creating compressors from an encoder.
///
/// This trait defines a factory-like interface for creating objects that implement
/// the `GraphCompressor` trait. The compressor is initialized using an encoder
/// (which implements `EncodeAndEstimate`) and a set of compression parameters.
pub trait CompressorFromEncoder {
    /// Creates a new compressor instance using the provided encoder and compression parameters.
    ///
    /// # Arguments
    /// - `encoder`: An object implementing the `EncodeAndEstimate` trait, used for encoding
    ///   and estimating compression efficiency.
    /// - `parameters`: A reference to the `CompressionParameters` struct, which provides
    ///   configuration for the compression process.
    ///
    /// # Returns
    /// An object implementing the `GraphCompressor` trait.
    fn from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor;
}

/// A factory for creating `BvComp` (greedy) compressors.
///
/// This struct provides an implementation of the `CompressorFromEncoder` trait
/// for creating `BvComp` instances, which are used for basic compression.
pub struct BvCompCreate;

impl CompressorFromEncoder for BvCompCreate {
    /// Creates a new `BvComp` compressor using the provided encoder and parameters.
    ///
    /// # Arguments
    /// - `encoder`: An object implementing the `EncodeAndEstimate` trait.
    /// - `parameters`: A reference to the `CompressionParameters` struct.
    ///
    /// # Returns
    /// A `BvComp` instance configured with the provided encoder and parameters.
    fn from_encoder(
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
            0, // Default value for an additional parameter (e.g., offset).
        )
    }
}

/// A factory for creating `BvCompZ` compressors with configurable chunk sizes.
///
/// This struct provides an implementation of the `CompressorFromEncoder` trait
/// for creating `BvCompZ` instances, approximated reference selection algorithm, as 
/// described in Zuckerli, which works with chunk-based compression.
pub struct BvCompZCreate {
    // The size of chunks used during compression.
    chunk_size: usize, 
}

impl BvCompZCreate {
    /// Creates a new `BvCompZCreate` instance with the specified chunk size.
    ///
    /// # Arguments
    /// - `chunk_size`: The size of chunks to be used during compression.
    ///
    /// # Returns
    /// A new `BvCompZCreate` instance.
    pub fn with_chunk_size(chunk_size: usize) -> BvCompZCreate {
        BvCompZCreate { chunk_size }
    }
}

impl CompressorFromEncoder for BvCompZCreate {
    /// Creates a new `BvCompZ` compressor using the provided encoder and parameters.
    ///
    /// # Arguments
    /// - `encoder`: An object implementing the `EncodeAndEstimate` trait.
    /// - `parameters`: A reference to the `CompressionParameters` struct.
    ///
    /// # Returns
    /// A `BvCompZ` instance configured with the provided encoder, parameters, and chunk size.
    fn from_encoder(
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
            0, // Default value for an additional parameter (e.g., offset).
        )
    }
}