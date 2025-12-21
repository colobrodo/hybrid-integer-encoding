use anyhow::Result;

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
    /// The type of reference selection algorithm to use: greedy or approximate.
    pub compressor: CompressorType,
}

impl CompressionParameters {
    pub fn new() -> Self {
        Self {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
            compressor: CompressorType::Approximated { chunk_size: 10000 },
        }
    }

    pub fn with_compression_window(mut self, compression_window: usize) -> Self {
        self.compression_window = compression_window;
        self
    }

    pub fn with_max_reference_count(mut self, max_ref_count: usize) -> Self {
        self.max_ref_count = max_ref_count;
        self
    }

    pub fn with_min_interval_length(mut self, min_interval_length: usize) -> Self {
        self.min_interval_length = min_interval_length;
        self
    }

    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_greedy_compressor(mut self) -> Self {
        self.compressor = CompressorType::Greedy;
        self
    }

    pub fn with_approximated_compressor(mut self) -> Self {
        self.compressor = CompressorType::Approximated { chunk_size: 10000 };
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.compressor = CompressorType::Approximated { chunk_size };
        self
    }

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
        match self.compressor {
            CompressorType::Greedy => s.push_str("compressortype=greedy\n"),
            CompressorType::Approximated { chunk_size } => {
                s.push_str("compressortype=approximated\n");
                s.push_str(&format!("chunksize={chunk_size}\n"));
            }
        };
        Ok(s)
    }
}

/// Determine the type of the graph compressor to use: `Approximated` as `Zuckerli`, or `Greedy` as `WebGraph`.
/// The `Approximate` field requires the size of the chunks, for the parallel compression.
#[derive(Debug, Copy, Clone)]
pub enum CompressorType {
    Approximated { chunk_size: usize },
    Greedy,
}
