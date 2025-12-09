#[cfg(test)]
mod tests {
    use hybrid_integer_encoding::huffman::{DefaultEncodeParams, IntegerHistograms};

    #[test]
    fn new_histogram_is_empty() {
        let hist = IntegerHistograms::<DefaultEncodeParams>::new(3, 256);
        assert_eq!(hist.number_of_contexts(), 3);
        assert!(hist.is_empty());
        assert_eq!(hist.count(), 0);
        for ctx in 0..3 {
            assert_eq!(hist.context_count(ctx as u8), 0);
        }
    }

    #[test]
    fn add_increments_context_and_total_counts() {
        let mut hist = IntegerHistograms::<DefaultEncodeParams>::new(4, 256);
        assert!(hist.is_empty());

        hist.add(1, 5);
        assert!(!hist.is_empty());
        assert_eq!(hist.context_count(1), 1);
        assert_eq!(hist.count(), 1);

        // Adding the same value again in same context increases count
        hist.add(1, 5);
        assert_eq!(hist.context_count(1), 2);
        assert_eq!(hist.count(), 2);

        // Adding a value in a different context
        hist.add(2, 0);
        assert_eq!(hist.context_count(2), 1);
        assert_eq!(hist.count(), 3);

        // Other contexts remain unchanged
        assert_eq!(hist.context_count(0), 0);
        assert_eq!(hist.context_count(3), 0);
    }

    #[test]
    fn add_all_merges_two_histograms() {
        let mut a = IntegerHistograms::<DefaultEncodeParams>::new(2, 256);
        let mut b = IntegerHistograms::<DefaultEncodeParams>::new(2, 256);

        // a: ctx0=1, ctx1=2
        a.add(0, 1);
        a.add(1, 2);
        a.add(1, 2);

        // b: ctx0=2, ctx1=0
        b.add(0, 1);
        b.add(0, 3);

        assert_eq!(a.count(), 3);
        assert_eq!(b.count(), 2);

        a.add_all(&b);

        // After merge: ctx0 = 1 + 2 = 3, ctx1 = 2 + 0 = 2
        assert_eq!(a.context_count(0), 3);
        assert_eq!(a.context_count(1), 2);
        assert_eq!(a.count(), 5);
    }
}
