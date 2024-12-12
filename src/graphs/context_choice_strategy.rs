use super::BvGraphComponent;

/// A strategy that defines how the context of each encoded or estimated value is chosen during graph compression.
pub trait ContextChoiceStrategy {
    fn num_contexts(&self) -> usize;
    fn choose_context(&mut self, component: BvGraphComponent, value: u64) -> u8;
}

/// Simple context strategy that uses a different context for each graph component
pub struct SimpleChoiceStrategy;

impl ContextChoiceStrategy for SimpleChoiceStrategy {
    fn num_contexts(&self) -> usize {
        BvGraphComponent::COMPONENTS as usize
    }

    fn choose_context(&mut self, component: BvGraphComponent, _value: u64) -> u8 {
        component as u8
    }
}
