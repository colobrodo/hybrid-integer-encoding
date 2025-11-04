mod tests {
    use dsi_bitstream::traits::BE;
    use epserde::deser::Owned;
    use hybrid_integer_encoding::{
        graphs::{
            build_offsets, compare_graphs, convert_graph, load_graph, load_graph_seq,
            measure_stats, ComparisonResult, CompressionParameters, ContextModel, CreateBvComp,
            CreateBvCompZ, SimpleContextModel, ZuckerliContextModel,
        },
        huffman::DefaultEncodeParams,
    };
    use lender::Lender;
    use std::{path::PathBuf, str::FromStr};
    use tempfile::TempDir;
    use webgraph::prelude::*;

    fn compress_graph<C: ContextModel + Default + Copy + 'static>(
        compression_parameters: CompressionParameters,
        max_bits: usize,
        greedily: bool,
        access_random: bool,
    ) -> anyhow::Result<()> {
        let basename = PathBuf::from_str("tests/data/cnr-2000")?;

        // Create a temporary directory
        let temp_dir = TempDir::new()?;
        // Get output basename with the same full path as the but in the temp folder
        let output_basename = temp_dir.path().join(basename.file_name().unwrap());

        if greedily {
            convert_graph::<C>(
                &basename,
                &output_basename,
                max_bits,
                CreateBvComp,
                &compression_parameters,
            )?;
        } else {
            convert_graph::<C>(
                &basename,
                &output_basename,
                max_bits,
                CreateBvCompZ::with_chunk_size(10000),
                &compression_parameters,
            )?;
        }

        if access_random {
            let original_graph = BvGraphSeq::with_basename(basename.clone())
                .endianness::<BE>()
                .load()?;
            let graph = load_graph_seq::<C>(&output_basename, max_bits)?;
            build_offsets(graph, &output_basename)?;
            compare_graph_randomly::<C>(max_bits, original_graph, output_basename)?;
        } else {
            let result = compare_graphs::<C>(output_basename, basename, max_bits)?;
            assert!(matches!(result, ComparisonResult::Equal));
        }

        Ok(())
    }

    fn compare_graph_randomly<C: ContextModel + Default + Copy + 'static>(
        max_bits: usize,
        original_graph: BvGraphSeq<
            DynCodesDecoderFactory<BE, MmapHelper<u32>, Owned<EmptyDict<usize, usize>>>,
        >,
        output_basename: PathBuf,
    ) -> anyhow::Result<()> {
        let graph = load_graph::<C>(output_basename, max_bits)?;
        let mut original_iter = original_graph.iter().enumerate();
        while let Some((i, (node_id, true_succ))) = original_iter.next() {
            let succ = graph.successors(node_id);

            assert_eq!(node_id, i);
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
        compress_graph::<SimpleContextModel>(compression_parameters, 12, false, false)
            .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_accessing_randomly() {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 12, false, true)
            .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_with_multiple_rounds() {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 3,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 12, false, false)
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
        compress_graph::<SimpleContextModel>(compression_parameters, 8, false, false)
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
        compress_graph::<ZuckerliContextModel<DefaultEncodeParams>>(
            compression_parameters,
            12,
            false,
            false,
        )
        .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_with_bigger_window_size() {
        let compression_parameters = CompressionParameters {
            compression_window: 32,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 12, false, false)
            .expect("Converting the graph");
    }

    #[test]
    fn compress_and_decompress_greedily() {
        let compression_parameters = CompressionParameters {
            compression_window: 32,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_graph::<SimpleContextModel>(compression_parameters, 12, true, false)
            .expect("Converting the graph");
    }

    #[test]
    fn check_stats() -> anyhow::Result<()> {
        let basename = PathBuf::from_str("tests/data/cnr-2000")?;

        // Create a temporary directory
        let temp_dir = TempDir::new()?;
        // Get output basename with the same full path as the but in the temp folder
        let output_basename = temp_dir.path().join(basename.file_name().unwrap());

        let compression_parameters = CompressionParameters {
            compression_window: 32,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        let max_bits = 12;
        convert_graph::<SimpleContextModel>(
            &basename,
            &output_basename,
            max_bits,
            CreateBvCompZ::with_chunk_size(10000),
            &compression_parameters,
        )?;
        let graph = load_graph_seq::<SimpleContextModel>(&output_basename, max_bits)?;

        let stats = measure_stats(graph);
        assert_eq!(stats.total, 7903485);
        assert_eq!(stats.outdegrees, 1450145);
        assert_eq!(stats.reference_offsets, 705343);
        assert_eq!(stats.block_counts, 367511);
        assert_eq!(stats.blocks, 687172);
        assert_eq!(stats.interval_counts, 237652);
        assert_eq!(stats.interval_starts, 419029);
        assert_eq!(stats.interval_lens, 192597);
        assert_eq!(stats.first_residuals, 1330119);
        assert_eq!(stats.residuals, 2513917);

        Ok(())
    }
}
