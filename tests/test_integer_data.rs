#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hybrid_integer_encoding::huffman::DefaultEncodeParams;
    use hybrid_integer_encoding::utils::IntegerData;

    #[test]
    fn test_new_integer_data() -> Result<()> {
        let data = IntegerData::<DefaultEncodeParams>::new(4, 10);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
        Ok(())
    }

    #[test]
    fn test_adding_values_updates_length() -> Result<()> {
        let mut data = IntegerData::<DefaultEncodeParams>::new(2, 5);
        assert_eq!(data.len(), 0);

        data.add(0, 1);
        assert_eq!(data.len(), 1);

        data.add(1, 2);
        assert_eq!(data.len(), 2);

        Ok(())
    }

    #[test]
    fn test_add_and_iterate() -> Result<()> {
        let mut data = IntegerData::<DefaultEncodeParams>::new(4, 10);

        // Add some test values with different contexts
        data.add(0, 5);
        data.add(1, 3);
        data.add(0, 7);
        data.add(2, 1);

        assert_eq!(data.len(), 4);
        assert!(!data.is_empty());

        // Collect the iterator results to verify the values
        let results: Vec<_> = data.iter().collect();
        assert_eq!(results, vec![(0, 5), (1, 3), (0, 7), (2, 1)]);

        Ok(())
    }
}
