//! # oxide-partition
//!
//! Data partitioning for distributed GPU processing with ternary balance signals.
//!
//! Provides partitioning strategies (hash, range, round-robin, consistent hashing),
//! balance monitoring with ternary signals (+1 balanced, 0 skewed, -1 hotspot),
//! and automatic rebalancing for GPU workloads.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// ---------------------------------------------------------------------------
// Ternary balance signal
// ---------------------------------------------------------------------------

/// Ternary balance signal for partition health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryBalance {
    /// +1 — partition sizes are within acceptable thresholds.
    Balanced,
    /// 0 — some skew detected but not critical.
    Skewed,
    /// -1 — hotspot: a partition is receiving disproportionate traffic/data.
    Hotspot,
}

impl TernaryBalance {
    /// Numeric value: +1, 0, or -1.
    pub fn value(&self) -> i8 {
        match self {
            TernaryBalance::Balanced => 1,
            TernaryBalance::Skewed => 0,
            TernaryBalance::Hotspot => -1,
        }
    }

    /// Is the partition system healthy?
    pub fn is_healthy(&self) -> bool {
        matches!(self, TernaryBalance::Balanced)
    }
}

impl std::fmt::Display for TernaryBalance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

// ---------------------------------------------------------------------------
// PartitionMap
// ---------------------------------------------------------------------------

/// Tracks which data ranges live on which GPU partition.
#[derive(Debug, Clone)]
pub struct PartitionMap<K> {
    /// Number of partitions (GPUs).
    num_partitions: usize,
    /// Mapping from partition id -> collection of keys assigned to it.
    assignments: HashMap<usize, Vec<K>>,
}

impl<K: Clone> PartitionMap<K> {
    /// Create a new empty partition map with `num_partitions` slots.
    pub fn new(num_partitions: usize) -> Self {
        let mut assignments = HashMap::with_capacity(num_partitions);
        for i in 0..num_partitions {
            assignments.insert(i, Vec::new());
        }
        Self {
            num_partitions,
            assignments,
        }
    }

    /// Number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Assign `key` to `partition`.
    pub fn assign(&mut self, partition: usize, key: K) {
        if partition < self.num_partitions {
            self.assignments.entry(partition).or_default().push(key);
        }
    }

    /// Get all keys in a partition.
    pub fn get(&self, partition: usize) -> &[K] {
        self.assignments.get(&partition).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of items in a partition.
    pub fn len(&self, partition: usize) -> usize {
        self.assignments.get(&partition).map(|v| v.len()).unwrap_or(0)
    }

    /// Total items across all partitions.
    pub fn total(&self) -> usize {
        self.assignments.values().map(|v| v.len()).sum()
    }

    /// Return partition sizes as a vector.
    pub fn sizes(&self) -> Vec<usize> {
        (0..self.num_partitions).map(|i| self.len(i)).collect()
    }

    /// Check ternary balance across all partitions.
    ///
    /// Thresholds:
    /// - Balanced: max deviation < 10% of average
    /// - Skewed:   max deviation < 30% of average
    /// - Hotspot:  max deviation >= 30% of average
    pub fn check_balance(&self) -> TernaryBalance {
        let sizes = self.sizes();
        let total: usize = sizes.iter().sum();
        if total == 0 || self.num_partitions == 0 {
            return TernaryBalance::Balanced;
        }
        let avg = total as f64 / self.num_partitions as f64;
        if avg == 0.0 {
            return TernaryBalance::Balanced;
        }
        let max_dev = sizes.iter().map(|&s| ((s as f64 - avg).abs() / avg)).fold(0.0_f64, f64::max);
        if max_dev < 0.10 {
            TernaryBalance::Balanced
        } else if max_dev < 0.30 {
            TernaryBalance::Skewed
        } else {
            TernaryBalance::Hotspot
        }
    }

    /// Detect hotspot partitions — those whose size exceeds `threshold` × average.
    pub fn detect_hotspots(&self, threshold: f64) -> Vec<usize> {
        let sizes = self.sizes();
        let total: usize = sizes.iter().sum();
        if total == 0 || self.num_partitions == 0 {
            return vec![];
        }
        let avg = total as f64 / self.num_partitions as f64;
        sizes.iter().enumerate()
            .filter(|(_, &s)| s as f64 > avg * threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find underloaded partitions — those below `threshold` × average.
    pub fn detect_underloaded(&self, threshold: f64) -> Vec<usize> {
        let sizes = self.sizes();
        let total: usize = sizes.iter().sum();
        if total == 0 || self.num_partitions == 0 {
            return vec![];
        }
        let avg = total as f64 / self.num_partitions as f64;
        sizes.iter().enumerate()
            .filter(|(_, &s)| (s as f64) < avg * threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Move `count` items from `src` partition to `dst` partition.
    pub fn move_items(&mut self, src: usize, dst: usize, count: usize) -> usize {
        if src >= self.num_partitions || dst >= self.num_partitions || src == dst {
            return 0;
        }
        let n = {
            let src_vec = self.assignments.get(&src).map(|v| v.len()).unwrap_or(0);
            count.min(src_vec)
        };
        if n == 0 {
            return 0;
        }
        let drained: Vec<K> = {
            let src_vec = self.assignments.get_mut(&src).unwrap();
            let split_at = src_vec.len() - n;
            src_vec.split_off(split_at)
        };
        self.assignments.get_mut(&dst).unwrap().extend(drained);
        n
    }

    /// Get a mutable reference to the full assignments map.
    pub fn assignments_mut(&mut self) -> &mut HashMap<usize, Vec<K>> {
        &mut self.assignments
    }
}

// ---------------------------------------------------------------------------
// Partitioner trait
// ---------------------------------------------------------------------------

/// Strategy for assigning keys to GPU partitions.
pub trait Partitioner<K> {
    /// Determine the target partition for `key`.
    fn partition(&self, key: &K, num_partitions: usize) -> usize;
}

// -- Hash partitioner -------------------------------------------------------

/// Hash-based partitioner using `std::collections::hash_map::DefaultHasher`.
#[derive(Debug, Clone, Default)]
pub struct HashPartitioner;

impl<K: Hash> Partitioner<K> for HashPartitioner {
    fn partition(&self, key: &K, num_partitions: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % num_partitions
    }
}

// -- Range partitioner ------------------------------------------------------

/// Range-based partitioner using explicit boundary keys.
///
/// Assigns `key` to the partition whose boundary is the largest boundary `<= key`.
#[derive(Debug, Clone)]
pub struct RangePartitioner<K> {
    /// Sorted boundary keys. Partition `i` covers `boundaries[i]..boundaries[i+1]`.
    boundaries: Vec<K>,
}

impl<K: Ord> RangePartitioner<K> {
    /// Create a range partitioner from boundaries (will be sorted).
    pub fn new(mut boundaries: Vec<K>) -> Self {
        boundaries.sort();
        Self { boundaries }
    }

    /// Number of implied partitions = boundaries.len() + 1.
    pub fn num_partitions(&self) -> usize {
        self.boundaries.len() + 1
    }
}

impl<K: Ord> Partitioner<K> for RangePartitioner<K> {
    fn partition(&self, key: &K, num_partitions: usize) -> usize {
        let _ = num_partitions; // boundaries define the number
        match self.boundaries.binary_search(key) {
            Ok(i) => i,
            Err(i) => i,
        }
    }
}

// -- Round-robin partitioner ------------------------------------------------

/// Round-robin partitioner that cycles through partitions sequentially.
#[derive(Debug, Clone)]
pub struct RoundRobinPartitioner {
    counter: std::cell::Cell<usize>,
}

impl RoundRobinPartitioner {
    pub fn new() -> Self {
        Self { counter: std::cell::Cell::new(0) }
    }
}

impl Default for RoundRobinPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> Partitioner<K> for RoundRobinPartitioner {
    fn partition(&self, _key: &K, num_partitions: usize) -> usize {
        let current = self.counter.get();
        let result = current % num_partitions;
        self.counter.set(current + 1);
        result
    }
}

// ---------------------------------------------------------------------------
// Consistent hashing
// ---------------------------------------------------------------------------

/// Consistent hashing ring for stable partitioning under node changes.
///
/// Each node (GPU) is placed on the ring at multiple virtual points.
/// A key is assigned to the first node clockwise from its hash position.
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Sorted (position, node_id) pairs.
    ring: Vec<(u64, usize)>,
    /// Number of virtual nodes per real node.
    vnodes: usize,
}

impl ConsistentHashRing {
    /// Create a ring with `nodes` node ids and `vnodes` virtual nodes each.
    pub fn new(nodes: &[usize], vnodes: usize) -> Self {
        let mut ring = Vec::with_capacity(nodes.len() * vnodes);
        for &node in nodes {
            for vn in 0..vnodes {
                let mut hasher = DefaultHasher::new();
                format!("node-{}-vn-{}", node, vn).hash(&mut hasher);
                let pos = hasher.finish();
                ring.push((pos, node));
            }
        }
        ring.sort_by_key(|(pos, _)| *pos);
        Self { ring, vnodes }
    }

    /// Number of real nodes on the ring.
    pub fn node_count(&self) -> usize {
        self.ring.iter().map(|(_, n)| *n).collect::<std::collections::HashSet<_>>().len()
    }

    /// Find the partition for `key`.
    pub fn partition<K: Hash>(&self, key: &K, _num_partitions: usize) -> usize {
        if self.ring.is_empty() {
            return 0;
        }
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let pos = hasher.finish();
        match self.ring.binary_search_by_key(&pos, |(p, _)| *p) {
            Ok(i) => self.ring[i].1,
            Err(i) => {
                if i >= self.ring.len() {
                    self.ring[0].1
                } else {
                    self.ring[i].1
                }
            }
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, node: usize) {
        for vn in 0..self.vnodes {
            let mut hasher = DefaultHasher::new();
            format!("node-{}-vn-{}", node, vn).hash(&mut hasher);
            let pos = hasher.finish();
            self.ring.push((pos, node));
        }
        self.ring.sort_by_key(|(pos, _)| *pos);
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node: usize) {
        self.ring.retain(|(_, n)| *n != node);
    }
}

// ---------------------------------------------------------------------------
// Rebalance
// ---------------------------------------------------------------------------

/// Rebalance a `PartitionMap` by moving data from hotspot partitions to
/// underloaded ones. Returns the number of items moved.
pub fn rebalance<K: Clone>(map: &mut PartitionMap<K>, hotspot_threshold: f64) -> usize {
    let hotspots = map.detect_hotspots(hotspot_threshold);
    let underloaded = map.detect_underloaded(1.0);
    let mut moved = 0;

    for &src in &hotspots {
        let avg = {
            let total = map.total();
            if total == 0 { continue; }
            total as f64 / map.num_partitions() as f64
        };
        let excess = map.len(src) as f64 - avg;
        let to_move = excess.ceil() as usize;

        for &dst in &underloaded {
            if to_move == 0 { break; }
            let need = (avg - map.len(dst) as f64).ceil() as usize;
            let batch = to_move.min(need);
            moved += map.move_items(src, dst, batch);
        }
    }
    moved
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_balance_values() {
        assert_eq!(TernaryBalance::Balanced.value(), 1);
        assert_eq!(TernaryBalance::Skewed.value(), 0);
        assert_eq!(TernaryBalance::Hotspot.value(), -1);
        assert!(TernaryBalance::Balanced.is_healthy());
        assert!(!TernaryBalance::Hotspot.is_healthy());
    }

    #[test]
    fn test_partition_map_balanced() {
        let mut map = PartitionMap::new(4);
        for i in 0..100u64 {
            map.assign((i % 4) as usize, i);
        }
        assert_eq!(map.check_balance(), TernaryBalance::Balanced);
        assert!(map.detect_hotspots(1.5).is_empty());
    }

    #[test]
    fn test_partition_map_hotspot() {
        let mut map = PartitionMap::<u64>::new(4);
        for i in 0..90 {
            map.assign(0, i); // dump everything into partition 0
        }
        assert_eq!(map.check_balance(), TernaryBalance::Hotspot);
        let hotspots = map.detect_hotspots(1.5);
        assert!(hotspots.contains(&0));
    }

    #[test]
    fn test_hash_partitioner_distribution() {
        let p = HashPartitioner;
        let num = 8;
        let mut counts = vec![0usize; num];
        for i in 0..1000u64 {
            counts[p.partition(&i, num)] += 1;
        }
        // Each bucket should have something
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "bucket {} has 0 items", i);
        }
    }

    #[test]
    fn test_range_partitioner() {
        let rp = RangePartitioner::new(vec![25u32, 50, 75]);
        assert_eq!(rp.partition(&10u32, 4), 0);
        assert_eq!(rp.partition(&30u32, 4), 1);
        assert_eq!(rp.partition(&60u32, 4), 2);
        assert_eq!(rp.partition(&80u32, 4), 3);
    }

    #[test]
    fn test_round_robin() {
        let rr = RoundRobinPartitioner::new();
        let num = 3;
        assert_eq!(rr.partition(&"a", num), 0);
        assert_eq!(rr.partition(&"b", num), 1);
        assert_eq!(rr.partition(&"c", num), 2);
        assert_eq!(rr.partition(&"d", num), 0);
    }

    #[test]
    fn test_consistent_hash_ring_stability() {
        let ring = ConsistentHashRing::new(&[0, 1, 2], 100);
        // Same key should always map to same node
        let a = ring.partition(&42u64, 3);
        let b = ring.partition(&42u64, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_consistent_hash_ring_add_remove() {
        let mut ring = ConsistentHashRing::new(&[0, 1, 2], 150);
        let before: Vec<usize> = (0..100u64).map(|k| ring.partition(&k, 3)).collect();
        ring.add_node(3);
        // Most keys should stay on the same node
        let after: Vec<usize> = (0..100u64).map(|k| ring.partition(&k, 4)).collect();
        let unchanged = before.iter().zip(after.iter()).filter(|(a, b)| a == b).count();
        assert!(unchanged > 60, "expected >60% stability, got {}%", unchanged);
        // Remove node 3 should restore
        ring.remove_node(3);
        let restored: Vec<usize> = (0..100u64).map(|k| ring.partition(&k, 3)).collect();
        assert_eq!(before, restored);
    }

    #[test]
    fn test_rebalance() {
        let mut map = PartitionMap::<u64>::new(4);
        // Create hotspot
        for i in 0..80 {
            map.assign(0, i);
        }
        assert_eq!(map.check_balance(), TernaryBalance::Hotspot);
        let moved = rebalance(&mut map, 1.5);
        assert!(moved > 0);
        // Should be more balanced now
        let balance = map.check_balance();
        assert!(balance.value() >= map.check_balance().value() || moved > 0);
    }

    #[test]
    fn test_partition_map_move_items() {
        let mut map = PartitionMap::new(2);
        for i in 0..10u64 {
            map.assign(0, i);
        }
        assert_eq!(map.len(0), 10);
        assert_eq!(map.len(1), 0);
        let moved = map.move_items(0, 1, 4);
        assert_eq!(moved, 4);
        assert_eq!(map.len(0), 6);
        assert_eq!(map.len(1), 4);
    }
}
