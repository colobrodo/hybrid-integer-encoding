use super::BvGraphComponent;

/// A strategy defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextChoiceStrategy {
    /// Returns the number of contexts available.
    fn num_contexts(&self) -> usize;
    /// Choose the context based on the current component that should be encoded.
    fn choose_context(&mut self, component: BvGraphComponent) -> u8;
    /// Update the strategy with the last value seen, for stateful strategy (like the one described in zuckerli).
    fn update(&mut self, component: BvGraphComponent, value: u64);
}

/// Simple context strategy that uses a different context for each graph component, and it's not based on previous values.
#[derive(Default, Clone, Copy)]
pub struct SimpleChoiceStrategy;

impl ContextChoiceStrategy for SimpleChoiceStrategy {
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

/// A strategy with only one context.
#[derive(Default, Clone, Copy)]
pub struct SingleContextStrategy;

impl ContextChoiceStrategy for SingleContextStrategy {
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
