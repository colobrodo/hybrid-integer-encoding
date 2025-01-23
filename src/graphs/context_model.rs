use super::BvGraphComponent;

/// A model defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextModel {
    /// Returns the number of contexts available.
    fn num_contexts(&self) -> usize;
    /// Choose the context based on the current component that should be encoded.
    fn choose_context(&mut self, component: BvGraphComponent) -> u8;
    /// Update the model with the last value seen, for stateful model (like the one described in zuckerli).
    fn update(&mut self, component: BvGraphComponent, value: u64);
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
}
