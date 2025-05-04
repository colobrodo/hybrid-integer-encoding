use sux::{dict::elias_fano::EfSeqDict, prelude::*};

pub trait GraphPartition {
    fn partition_of_node(&self, node_id: usize) -> usize;
    fn num_partitions(&self) -> usize;
}
pub struct BlockPartition {
    partitions: EfSeqDict,
}

impl GraphPartition for BlockPartition {
    fn partition_of_node(&self, node_id: usize) -> usize {
        match self.partitions.pred(node_id) {
            Some((_index, pred)) => pred,
            None => 0,
        }
    }

    fn num_partitions(&self) -> usize {
        self.partitions.len() + 1
    }
}

pub struct FixedSizePartition {
    partition_size: usize,
    num_partitions: usize,
}

impl FixedSizePartition {
    pub fn new(partition_size: usize, num_nodes: usize) -> Self {
        FixedSizePartition {
            partition_size,
            num_partitions: num_nodes.div_ceil(partition_size),
        }
    }
}

impl GraphPartition for FixedSizePartition {
    fn partition_of_node(&self, node_id: usize) -> usize {
        node_id / self.partition_size
    }

    fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}
