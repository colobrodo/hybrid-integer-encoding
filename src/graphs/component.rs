/// An enumeration of the components composing the BVGraph format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BvGraphComponent {
    Outdegree = 0,
    ReferenceOffset = 1,
    BlockCount = 2,
    Blocks = 3,
    IntervalCount = 4,
    IntervalStart = 5,
    IntervalLen = 6,
    FirstResidual = 7,
    Residual = 8,
}

impl BvGraphComponent {
    /// The number of components in the BVGraph format.
    pub const COMPONENTS: usize = 9;
}
