use super::BvGraphComponent;

/// A model defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextModel {
    /// Returns the number of contexts available.
    fn num_contexts(&self) -> usize;
    /// Choose the context based on the current component that should be encoded.
    fn choose_context(&mut self, component: BvGraphComponent) -> u8;
    /// Update the model with the last value seen, for stateful model (like the one described in zuckerli).
    fn update(&mut self, component: BvGraphComponent, value: u64);
    /// Signal the context model the start decoding of a new adjacency list
    fn reset(&mut self);
}

/// Simple context model that uses a different context for each graph component, and it's not based on previous values.
#[derive(Default, Clone, Copy)]
pub struct SimpleContextModel;

impl ContextModel for SimpleContextModel {
    #[inline(always)]
    fn num_contexts(&self) -> usize {
        BvGraphComponent::COMPONENTS
    }

    #[inline(always)]
    fn choose_context(&mut self, component: BvGraphComponent) -> u8 {
        component as u8
    }

    #[inline(always)]
    fn update(&mut self, _component: BvGraphComponent, _value: u64) {}

    fn reset(&mut self) {}
}

/// A model with only one fixed context.
#[derive(Default, Clone, Copy)]
pub struct SingleContextModel;

impl ContextModel for SingleContextModel {
    #[inline(always)]
    fn num_contexts(&self) -> usize {
        1
    }

    #[inline(always)]
    fn choose_context(&mut self, _component: BvGraphComponent) -> u8 {
        0
    }

    #[inline(always)]
    fn update(&mut self, _component: BvGraphComponent, _value: u64) {}

    fn reset(&mut self) {}
}

/// A partial implementation of the zuckerli context model, it does not implement
/// multiple contexts for outdegrees and references
#[derive(Default, Clone, Copy)]
pub struct ZuckerliContextModel {
    pub block_number: u32,
}

impl ZuckerliContextModel {
    const BASE_OUTDEGREE: usize = 0;
    const NUM_OUTDEGREES: usize = 1;
    const BASE_REFERENCE: usize = Self::BASE_OUTDEGREE + Self::NUM_OUTDEGREES;
    const NUM_REFERENCES: usize = 1;
    const BASE_BLOCK_COUNT: usize = Self::BASE_REFERENCE + Self::NUM_REFERENCES;
    const BASE_FIRST_BLOCK: usize = Self::BASE_BLOCK_COUNT + 1;
    const BASE_EVEN_BLOCK: usize = Self::BASE_FIRST_BLOCK + 1;
    const BASE_ODD_BLOCK: usize = Self::BASE_EVEN_BLOCK + 1;
    // TODO: For delta-encoding the first residual with respect to the current node, the symbol that would
    // be used to represent the number of residuals defines which distribution to use. This is because a
    // list with a high number of residuals will likely be harder to predict.
    // Finally, for all other residual deltas, the symbol that was used to encode the previous one is
    // used to choose the corresponding probability distribution for the current delta
    const BASE_INTERVAL_COUNT: usize = Self::BASE_ODD_BLOCK + 1;
    const BASE_INTERVAL_START: usize = Self::BASE_INTERVAL_COUNT + 1;
    const BASE_INTERVAL_LEN: usize = Self::BASE_INTERVAL_START + 1;
    const BASE_FIRST_RESIDUAL: usize = Self::BASE_INTERVAL_LEN + 1;
    const BASE_RESIDUAL: usize = Self::BASE_FIRST_RESIDUAL + 1;
    const NUM_RESIDUALS: usize = 1;
    const NUM_CONTEXTS: usize = Self::BASE_RESIDUAL + Self::NUM_RESIDUALS;
}

impl ContextModel for ZuckerliContextModel {
    fn num_contexts(&self) -> usize {
        Self::NUM_CONTEXTS
    }

    fn choose_context(&mut self, component: BvGraphComponent) -> u8 {
        (match component {
            BvGraphComponent::Outdegree => Self::BASE_OUTDEGREE,
            BvGraphComponent::ReferenceOffset => Self::BASE_REFERENCE,
            BvGraphComponent::BlockCount => Self::BASE_BLOCK_COUNT,
            BvGraphComponent::Blocks if self.block_number == 0 => Self::BASE_FIRST_BLOCK,
            BvGraphComponent::Blocks => {
                if self.block_number & 1 == 0 {
                    Self::BASE_EVEN_BLOCK
                } else {
                    Self::BASE_ODD_BLOCK
                }
            }
            BvGraphComponent::IntervalCount => Self::BASE_INTERVAL_COUNT,
            BvGraphComponent::IntervalStart => Self::BASE_INTERVAL_START,
            BvGraphComponent::IntervalLen => Self::BASE_INTERVAL_LEN,
            BvGraphComponent::FirstResidual => Self::BASE_FIRST_RESIDUAL,
            BvGraphComponent::Residual => Self::BASE_RESIDUAL,
        }) as u8
    }

    fn update(&mut self, component: BvGraphComponent, _value: u64) {
        if component == BvGraphComponent::Blocks {
            self.block_number += 1;
        }
    }

    fn reset(&mut self) {
        self.block_number = 0;
    }
}
