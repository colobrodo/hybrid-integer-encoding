use webgraph::prelude::{BvComp, BvCompZ, EncodeAndEstimate, GraphCompressor};

pub struct CompressionParameters {
    pub compression_window: usize,
    pub max_ref_count: usize,
    pub min_interval_length: usize,
    pub num_rounds: usize,
}

pub trait CompressorFromEncoder {
    fn from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor;
}

pub struct BvCompCreate;

impl CompressorFromEncoder for BvCompCreate {
    fn from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor {
        BvComp::new(
            encoder,
            parameters.compression_window,
            parameters.max_ref_count,
            parameters.min_interval_length,
            0,
        )
    }
}

pub struct BvCompZCreate {
    chunk_size: usize,
}

impl BvCompZCreate {
    pub fn with_chunk_size(chunk_size: usize) -> BvCompZCreate {
        BvCompZCreate { chunk_size }
    }
}

impl CompressorFromEncoder for BvCompZCreate {
    fn from_encoder(
        &self,
        encoder: impl EncodeAndEstimate,
        parameters: &CompressionParameters,
    ) -> impl GraphCompressor {
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
