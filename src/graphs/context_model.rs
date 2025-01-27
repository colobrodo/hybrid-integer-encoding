use crate::huffman::{encode, EncodeParams};

use super::BvGraphComponent;

/// A model defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextModel {
    /// Returns the number of contexts available.
    fn num_contexts() -> usize;
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

/// A partial implementation of the zuckerli context model, it does not implement
/// multiple contexts for outdegrees and references
#[derive(Default, Clone, Copy)]
pub struct ZuckerliContextModel<EP: EncodeParams> {
    block_number: u32,
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
    // TODO: For delta-encoding the first residual with respect to the current node, the symbol that would
    // be used to represent the number of residuals defines which distribution to use. This is because a
    // list with a high number of residuals will likely be harder to predict.
    // TODO: to calculate the number of residual we should keep track of the number of items written and deduce the remaining from the outdegree
    //       we can do that looking at the number of odd blocks and lenghts of the intervals except for the fact that the last block is implicit and never written
    //       because can be deduced from the reference's outdegree. unfortunatly we don't have it from bvcomp and we cannot calculate it :/
    const BASE_FIRST_RESIDUAL: usize = Self::BASE_INTERVAL_LEN + 1;
    const BASE_RESIDUAL: usize = Self::BASE_FIRST_RESIDUAL + 1;
    // 80 in the original implementation
    const NUM_RESIDUALS: usize = 16;
    const NUM_CONTEXTS: usize = Self::BASE_RESIDUAL + Self::NUM_RESIDUALS;
}

impl<EP: EncodeParams> ContextModel for ZuckerliContextModel<EP> {
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
            BvGraphComponent::FirstResidual => Self::BASE_FIRST_RESIDUAL,
            BvGraphComponent::Residual => {
                let (token, _, _) = encode::<EP>(self.last_residual);
                Self::BASE_RESIDUAL + token.min(Self::NUM_RESIDUALS - 1)
            }
        }) as u8
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
        self.block_number = 0;
        self.last_residual = 0;
    }
}
