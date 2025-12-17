mod tests {
    use anyhow::{Context, Result};
    use dsi_bitstream::traits::BE;
    use epserde::deser::Owned;
    use hybrid_integer_encoding::graphs::{
        build_offsets, check_compression_parameters, compare_graphs, convert_graph,
        convert_graph_file, load_graph, load_graph_seq, measure_stats, parallel_convert_graph_file,
        CompressionParameters, CompressorType, ConstantContextModel, ContextModel,
        SimpleContextModel,
    };
    use lender::Lender;
    use std::{
        path::{Path, PathBuf},
        str::FromStr,
    };
    use tempfile::TempDir;
    use webgraph::graphs::random::ErdosRenyi;
    use webgraph::prelude::*;

    fn compress_cnr_2000<C: ContextModel + Default + Copy + 'static>(
        compression_parameters: CompressionParameters,
        max_bits: usize,
        greedily: bool,
        access_random: bool,
        parallel: bool,
    ) -> anyhow::Result<()> {
        let basename = PathBuf::from_str("tests/data/cnr-2000")?;

        // Create a temporary directory
        let temp_dir = TempDir::new()?;
        // Get output basename with the same full path as the but in the temp folder
        let output_basename = temp_dir.path().join(basename.file_name().unwrap());

        if parallel {
            parallel_convert_graph_file::<C>(
                &basename,
                &output_basename,
                max_bits,
                if greedily {
                    CompressorType::Greedy
                } else {
                    CompressorType::Approximated { chunk_size: 10000 }
                },
                &compression_parameters,
            )?;
        } else {
            convert_graph_file::<C>(
                &basename,
                &output_basename,
                max_bits,
                if greedily {
                    CompressorType::Greedy
                } else {
                    CompressorType::Approximated { chunk_size: 10000 }
                },
                &compression_parameters,
            )?;
        }

        if access_random {
            let original_graph = BvGraphSeq::with_basename(&basename)
                .endianness::<BE>()
                .load()?;
            let graph = load_graph_seq::<C>(&output_basename, max_bits)?;
            build_offsets(graph, &output_basename)?;
            compare_graph_randomly::<C>(max_bits, original_graph, output_basename)?;
        } else {
            let result = compare_graphs::<C>(output_basename, basename, max_bits)?;
            assert!(result.is_ok());
        }

        Ok(())
    }

    fn compress_random_graph<C: ContextModel + Default + Copy + 'static>(
        compression_parameters: CompressionParameters,
        max_bits: usize,
        greedily: bool,
        seed: u64,
    ) -> anyhow::Result<()> {
        // Create a temporary directory
        let temp_dir = TempDir::new()?;
        // Get output basename with the same full path as the but in the temp folder
        let output_basename = temp_dir.path().join("random-graph");

        let random_graph = ErdosRenyi::new(1000, 0.2, seed);
        let mut graph = VecGraph::new();
        graph.add_lender(random_graph.iter());

        if greedily {
            convert_graph::<C, _>(
                &graph,
                &output_basename,
                max_bits,
                CompressorType::Greedy,
                &compression_parameters,
                false,
            )?;
        } else {
            convert_graph::<C, _>(
                &graph,
                &output_basename,
                max_bits,
                CompressorType::Approximated { chunk_size: 10000 },
                &compression_parameters,
                false,
            )?;
        }
        let huffman_graph = load_graph_seq::<C>(&output_basename, max_bits)?;
        assert!(graph::eq(&graph, &huffman_graph).is_ok());

        Ok(())
    }

    fn compare_graph_randomly<C: ContextModel + Default + Copy + 'static>(
        max_bits: usize,
        original_graph: BvGraphSeq<
            DynCodesDecoderFactory<BE, MmapHelper<u32>, Owned<EmptyDict<usize, usize>>>,
        >,
        output_basename: impl AsRef<Path>,
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
    fn test_compress_and_decompress_random_graph() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 12, false, 0)
            .with_context(|| "Converting a random graph")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_random_graph2() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 12, false, 31415)
            .with_context(|| "Converting a random graph")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_accessing_randomly() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_cnr_2000::<SimpleContextModel>(compression_parameters, 12, false, true, false)
            .with_context(|| "Converting the graph and reading accessing randomly")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_cnr_2000_with_12_bits() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_cnr_2000::<SimpleContextModel>(compression_parameters, 12, false, false, false)
            .with_context(|| "Converting cnr-2000 with max 12 bits per word")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_cnr_2000_parallel() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_cnr_2000::<SimpleContextModel>(compression_parameters, 12, false, false, true)
            .with_context(|| "Converting cnr-2000 running the estimation rounds in parallel")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_with_multiple_rounds() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 3,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 12, false, 0)
            .with_context(|| "Converting the graph with multiple rounds")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_with_8_bits() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 8, false, 0)
            .with_context(|| "Converting the graph with 8-bit maximum size")?;
        Ok(())
    }

    // TODO: to remove when convert_graphs uses sequential encoding again
    // #[test]
    // fn test_compress_and_decompress_with_zuckerli_context_model() -> Result<()> {
    //     let compression_parameters = CompressionParameters {
    //         compression_window: 7,
    //         max_ref_count: 3,
    //         min_interval_length: 4,
    //         num_rounds: 1,
    //     };
    //     compress_random_graph::<ZuckerliContextModel<DefaultEncodeParams>>(
    //         compression_parameters,
    //         12,
    //         false,
    //         0,
    //     )
    //     .with_context(|| "Converting the graph with Zuckerli-like context model")?;
    //     Ok(())
    // }

    #[test]
    fn test_compress_and_decompress_with_bigger_window_size() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 32,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 12, false, 0)
            .with_context(|| "Converting the graph with bigger window size")?;
        Ok(())
    }

    #[test]
    fn test_compress_and_decompress_greedily() -> Result<()> {
        let compression_parameters = CompressionParameters {
            compression_window: 32,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };
        compress_random_graph::<SimpleContextModel>(compression_parameters, 12, true, 0)
            .with_context(|| "Converting the graph with greedy reference selection")?;
        Ok(())
    }

    #[test]
    fn test_check_compression_parameters_failures() -> Result<()> {
        // First create an in-memory graph with known parameters
        let graph = VecGraph::from_arcs([(0, 1), (1, 2), (2, 0), (2, 1)]);

        let temp_dir = TempDir::new()?;
        let output_basename = temp_dir.path().join("property_test");

        let compression_parameters = CompressionParameters {
            compression_window: 7,
            max_ref_count: 3,
            min_interval_length: 4,
            num_rounds: 1,
        };

        // Create and save a graph with SimpleContextModel
        let expected_max_bits = 12;
        convert_graph::<SimpleContextModel, _>(
            &graph,
            &output_basename,
            expected_max_bits,
            CompressorType::Approximated { chunk_size: 10000 },
            &compression_parameters,
            false,
        )?;

        // Test with wrong max_bits
        let properties_path = output_basename.with_extension("properties");
        let wrong_max_bits = 8;
        let wrong_max_bits_result = check_compression_parameters(
            &properties_path,
            wrong_max_bits,
            SimpleContextModel::NAME,
        );
        assert!(wrong_max_bits_result.is_err());

        // Test with wrong context model
        let wrong_model_name_result = check_compression_parameters(
            &output_basename,
            expected_max_bits,
            ConstantContextModel::NAME,
        );
        assert!(wrong_model_name_result.is_err());

        Ok(())
    }

    #[test]
    fn test_graph_stats() -> Result<()> {
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
        convert_graph_file::<SimpleContextModel>(
            &basename,
            &output_basename,
            max_bits,
            CompressorType::Approximated { chunk_size: 10000 },
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
