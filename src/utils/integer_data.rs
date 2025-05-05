use std::marker::PhantomData;

use crate::huffman::{EncodeParams, IntegerHistograms};

pub struct IntegerData<EP: EncodeParams> {
    values: Vec<u32>,
    contexts: Vec<u8>,
    histograms: IntegerHistograms<EP>,
    _marker: PhantomData<EP>,
}

impl<EP: EncodeParams> IntegerData<EP> {
    pub fn new(num_contexts: usize, num_symbols: usize) -> Self {
        Self {
            values: Vec::new(),
            contexts: Vec::new(),
            histograms: IntegerHistograms::new(num_contexts, num_symbols),
            _marker: PhantomData,
        }
    }

    pub fn add(&mut self, context: u8, value: u32) {
        self.values.push(value);
        self.contexts.push(context);
        self.histograms.add(context, value);
    }
    pub fn iter(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.contexts
            .iter()
            .zip(self.values.iter())
            .map(|(&ctx, &value)| (ctx, value))
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn histograms(self) -> IntegerHistograms<EP> {
        self.histograms
    }
}
