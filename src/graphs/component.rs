use std::fmt;

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

impl fmt::Display for BvGraphComponent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BvGraphComponent::Outdegree => write!(f, "Outdegree"),
            BvGraphComponent::ReferenceOffset => write!(f, "Reference offset"),
            BvGraphComponent::BlockCount => write!(f, "Block count"),
            BvGraphComponent::Blocks => write!(f, "Blocks"),
            BvGraphComponent::IntervalCount => write!(f, "Interval count"),
            BvGraphComponent::IntervalStart => write!(f, "Interval start"),
            BvGraphComponent::IntervalLen => write!(f, "Interval Length"),
            BvGraphComponent::FirstResidual => write!(f, "First residuals"),
            BvGraphComponent::Residual => write!(f, "Residual"),
        }
    }
}
