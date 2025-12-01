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

/// Determine the type of the graph compressor to use: `Approximated` as `Zuckerli`, or `Greedy` as `WebGraph`.
/// The `Approximate` field requires the size of the chunks, for the parallel compression.
#[derive(Debug, Copy, Clone)]
pub enum CompressorType {
    Approximated { chunk_size: usize },
    Greedy,
}
