use std::marker::PhantomData;

use crate::huffman::{EncodeParams, IntegerHistogram};

/// A collection that stores integer symbols together with a small context
/// identifier for each symbol and a set of per-context histograms.
///
/// IntegerData is generic over an EncodeParams implementation `EP` which is
/// used by the internal `IntegerHistogram<EP>` to track per-context symbol
/// encoding frequencies.
pub struct IntegerData<EP: EncodeParams> {
    values: Vec<u32>,
    contexts: Vec<u8>,
    histograms: IntegerHistogram<EP>,
    _marker: PhantomData<EP>,
}

impl<EP: EncodeParams> IntegerData<EP> {
    /// Creates a new IntegerData collection with the specified number of contexts and symbols.
    /// The contexts and symbols are used to initialize the internal histogram tracking.
    pub fn new(num_contexts: usize, num_symbols: usize) -> Self {
        Self {
            values: Vec::new(),
            contexts: Vec::new(),
            histograms: IntegerHistogram::new(num_contexts, num_symbols),
            _marker: PhantomData,
        }
    }

    /// Adds a new integer value with its associated context to the collection.
    /// Updates the internal histogram for the given context with the new value.
    pub fn add(&mut self, context: u8, value: u32) {
        self.values.push(value);
        self.contexts.push(context);
        self.histograms.add(context, value);
    }

    /// Returns an iterator over pairs of (context, value) stored in the collection.
    /// The iterator yields tuples containing each value and its associated context.
    pub fn iter(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.contexts
            .iter()
            .zip(self.values.iter())
            .map(|(&ctx, &value)| (ctx, value))
    }

    /// Returns the total number of integer values stored in the collection.
    /// This count includes values across all contexts.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Checks if the collection contains no values.
    /// Returns true if no integer values have been added to the collection.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Consumes the collection and returns the internal histogram data structure.
    /// This provides access to the accumulated frequency information for each context.
    pub fn histograms(self) -> IntegerHistogram<EP> {
        self.histograms
    }
}
