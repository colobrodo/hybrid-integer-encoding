mod tests {
    use dsi_bitstream::traits::BE;
    use hybrid_integer_encoding::{
        graphs::{
            convert_graph, load_graph_seq, CompressionParameters, ContextModel, SimpleContextModel,
            ZuckerliContextModel,
        },
        huffman::DefaultEncodeParams,
    };
    use lender::Lender;
    use std::{fs, path::PathBuf, str::FromStr};
    use tempfile::TempDir;
    use webgraph::prelude::*;

    fn compress_graph<C: ContextModel + Default + Copy>(
        compression_parameters: CompressionParameters,
        max_bits: usize,
    ) -> anyhow::Result<()> {
        let basename = PathBuf::from_str("tests/data/cnr-2000")?;

        let original_graph = BvGraphSeq::with_basename(basename.clone())
            .endianness::<BE>()
            .load()?;

        // Create a temporary directory
        let temp_dir = TempDir::new()?;

        // Copy the properties file to the temporary directory
        let properties_path = basename.with_extension("properties");
        let temp_properties_path = temp_dir.path().join(properties_path.file_name().unwrap());
        // TODO: now only the default parameters are used, but we should allow to overwrite the ones in CompressionParameters
        //       to the copy of the properties file
        fs::copy(&properties_path, &temp_properties_path)?;

        // get output basename with the same full path as the properties file but without extension
        let output_basename = temp_properties_path.with_extension("");
        convert_graph::<C>(
            basename.clone(),
            output_basename.clone(),
            max_bits,
            compression_parameters,
            false,
        )?;

        let graph = load_graph_seq::<C>(output_basename, max_bits)?;

        let mut original_iter = original_graph.iter().enumerate();
        let mut iter = graph.iter();
        while let Some((i, (true_node_id, true_succ))) = original_iter.next() {
            let (node_id, succ) = iter.next().unwrap();

            assert_eq!(true_node_id, i);
            assert_eq!(true_node_id, node_id);
            assert_eq!(
                true_succ.into_iter().collect::<Vec<_>>(),
                succ.into_iter().collect::<Vec<_>>(),
                "node_id: {}",
                i
            );
        }

        Ok(())
    }

    #[test]
    fn compress_and_decompress_with_12_bits() {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 12)
            .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_with_8_bits() {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 8)
            .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_with_zuckerli_context_model() {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<ZuckerliContextModel<DefaultEncodeParams>>(compression_parameters, 12)
            .expect("Converting the graph");
    }
    // TODO: test with each context model and with different compression window
}
