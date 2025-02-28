use crate::huffman::{encode, EncodeParams};

use super::BvGraphComponent;

/// A model defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextModel {
    /// The name of the context model to be readed and written on the properties file
    const NAME: &str;
    /// Returns the number of contexts available.
    fn num_contexts() -> usize;
    /// Choose the context based on the current component that should be encoded.
    fn choose_context(&mut self, component: BvGraphComponent) -> u8;
    /// Update the model with the last value seen, for stateful model (like the one described in zuckerli).
    fn update(&mut self, component: BvGraphComponent, value: u64);
    /// Callback called to signal the number of total residuals
    fn num_of_residuals(&mut self, _total_residuals: usize) {}
    /// Signal the context model the start decoding of a new adjacency list
    fn reset(&mut self);
}

/// Simple context model that uses a different context for each graph component, and it's not based on previous values.
#[derive(Default, Clone, Copy)]
pub struct SimpleContextModel;

impl ContextModel for SimpleContextModel {
    const NAME: &str = "simple";

    #[inline(always)]
    fn num_contexts() -> usize {
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
    const NAME: &str = "single";

    #[inline(always)]
    fn num_contexts() -> usize {
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

/// A partial implementation of the Zuckerli context model, it does not implement
/// multiple contexts for outdegrees and references and introduce a context model
/// for intervals (originaly not presents because Zuckerli uses RLE instead)
#[derive(Default, Clone, Copy)]
pub struct ZuckerliContextModel<EP: EncodeParams> {
    /// Index of the upcoming block
    block_number: u32,
    /// total number of residuals: used to predict the context of the
    /// first residual
    total_residuals: usize,
    /// Last seen residual
    last_residual: u64,
    _marker: core::marker::PhantomData<EP>,
}

impl<EP: EncodeParams> ZuckerliContextModel<EP> {
    // The original implementation divides the adjacency lists into blocks of 32, then encodes the
    // outdegree as a delta with the previous (except for the first) and then does the dame with
    // references.
    // When decoding they use the previous delta as context but when they randomly access a node
    // they first decode the outdegree and reference and the previous element in the block.
    const BASE_OUTDEGREE: usize = 0;
    const NUM_OUTDEGREES: usize = 1;
    const BASE_REFERENCE: usize = Self::BASE_OUTDEGREE + Self::NUM_OUTDEGREES;
    const NUM_REFERENCES: usize = 1;
    const BASE_BLOCK_COUNT: usize = Self::BASE_REFERENCE + Self::NUM_REFERENCES;
    const BASE_FIRST_BLOCK: usize = Self::BASE_BLOCK_COUNT + 1;
    const BASE_EVEN_BLOCK: usize = Self::BASE_FIRST_BLOCK + 1;
    const BASE_ODD_BLOCK: usize = Self::BASE_EVEN_BLOCK + 1;
    // TODO: zuckerli doesn't use intervals but we can try to figure out a way to assign context to the intervals
    const BASE_INTERVAL_COUNT: usize = Self::BASE_ODD_BLOCK + 1;
    const BASE_INTERVAL_START: usize = Self::BASE_INTERVAL_COUNT + 1;
    const BASE_INTERVAL_LEN: usize = Self::BASE_INTERVAL_START + 1;
    // For delta-encoding the first residual with respect to the current node, the symbol that would
    // be used to represent the number of residuals defines which distribution to use. This is because a
    // list with a high number of residuals will likely be harder to predict.
    const BASE_FIRST_RESIDUAL: usize = Self::BASE_INTERVAL_LEN + 1;
    const BASE_RESIDUAL: usize = Self::BASE_FIRST_RESIDUAL + Self::NUM_FIRST_RESIDUALS;
    // 32 in the original implementation
    const NUM_FIRST_RESIDUALS: usize = 16;
    // 80 in the original implementation
    const NUM_RESIDUALS: usize = 16;
    const NUM_CONTEXTS: usize = Self::BASE_RESIDUAL + Self::NUM_RESIDUALS;
}

impl<EP: EncodeParams> ContextModel for ZuckerliContextModel<EP> {
    const NAME: &str = "zuckerli";

    fn num_contexts() -> usize {
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
            BvGraphComponent::FirstResidual => {
                let (token, _, _) = encode::<EP>(self.total_residuals as u64);
                Self::BASE_FIRST_RESIDUAL + token.min(Self::NUM_FIRST_RESIDUALS - 1)
            }
            BvGraphComponent::Residual => {
                let (token, _, _) = encode::<EP>(self.last_residual);
                Self::BASE_RESIDUAL + token.min(Self::NUM_RESIDUALS - 1)
            }
        }) as u8
    }

    fn num_of_residuals(&mut self, total_residuals: usize) {
        self.total_residuals = total_residuals;
    }

    fn update(&mut self, component: BvGraphComponent, value: u64) {
        match component {
            BvGraphComponent::Blocks => {
                self.block_number += 1;
            }
            BvGraphComponent::FirstResidual | BvGraphComponent::Residual => {
                self.last_residual = value;
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.total_residuals = 0;
        self.block_number = 0;
        self.last_residual = 0;
    }
}

/// A debug decorator for context models usefull for debugging purpose:
/// It wraps an existing context model and log each operation before executing it.
pub struct DebugContextModel<C: ContextModel> {
    model: C,
}

impl<C: ContextModel> ContextModel for DebugContextModel<C> {
    const NAME: &str = C::NAME;

    fn num_contexts() -> usize {
        C::num_contexts()
    }

    fn choose_context(&mut self, component: BvGraphComponent) -> u8 {
        eprintln!("Choose context for the next component {}", component);
        self.model.choose_context(component)
    }

    fn update(&mut self, component: BvGraphComponent, value: u64) {
        eprintln!(
            "Updated context model with value {} on component {}",
            component, value
        );
        self.model.update(component, value);
    }

    fn num_of_residuals(&mut self, total_residuals: usize) {
        eprintln!("Total residuals: {}", total_residuals);
        self.model.num_of_residuals(total_residuals);
    }

    fn reset(&mut self) {
        eprintln!("Resetted context model");
        self.model.reset();
    }
}

impl<C: ContextModel> DebugContextModel<C> {
    #[allow(dead_code)]
    fn new(context_model: C) -> Self {
        DebugContextModel {
            model: context_model,
        }
    }
}

impl<C: ContextModel + Default> Default for DebugContextModel<C> {
    fn default() -> Self {
        Self {
            model: Default::default(),
        }
    }
}

impl<C: ContextModel + Clone> Clone for DebugContextModel<C> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
        }
    }
}

impl<C: ContextModel + Copy> Copy for DebugContextModel<C> {}
