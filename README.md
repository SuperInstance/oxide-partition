# oxide-partition

**Data partitioning for distributed GPU processing with ternary balance signals.**

[![crates.io](https://img.shields.io/crates/v/oxide-partition.svg)](https://crates.io/crates/oxide-partition)

When you're distributing tensor data, batch workloads, or inference requests across multiple GPUs, how you split the data matters. Bad partitioning creates hotspots — one GPU drowning in work while others idle. `oxide-partition` gives you multiple partitioning strategies, real-time balance monitoring, and automatic rebalancing to keep your GPU cluster running flat out.

## Partitioning Strategies

Different workloads need different partitioning approaches. This crate ships with four:

### Hash Partitioning

The default for most workloads. Hash the key, mod by the number of partitions. Simple, fast, and statistically uniform for diverse key spaces.

```rust
use oxide_partition::{HashPartitioner, Partitioner};

let p = HashPartitioner;
let gpu_id = p.partition(&tensor_id, 8); // GPU 0..7
```

Best for: uniform key distributions, stateless inference, embedding lookups.

### Range Partitioning

Divide the key space into contiguous ranges. Keys within a range go to the same GPU, preserving spatial locality — critical for matrix operations where adjacent rows/columns are processed together.

```rust
use oxide_partition::{RangePartitioner, Partitioner};

let rp = RangePartitioner::new(vec![25u32, 50, 75]); // 4 partitions: [0,25), [25,50), [50,75), [75,+∞)
let gpu = rp.partition(&42u32, 4); // → GPU 1
```

Best for: sorted data, spatial workloads, sequential scans, matrix block distribution.

### Round-Robin Partitioning

Cycle through GPUs in order. Every Nth item goes to the same GPU. Dead simple and perfectly balanced for equal-sized work items.

```rust
use oxide_partition::{RoundRobinPartitioner, Partitioner};

let rr = RoundRobinPartitioner::new();
assert_eq!(rr.partition(&"item0", 3), 0); // GPU 0
assert_eq!(rr.partition(&"item1", 3), 1); // GPU 1
assert_eq!(rr.partition(&"item2", 3), 2); // GPU 2
assert_eq!(rr.partition(&"item3", 3), 0); // GPU 0 again
```

Best for: uniform work items, load testing, simple batch distribution.

### Consistent Hashing

The sophisticated choice for dynamic clusters. Each GPU gets placed at multiple virtual points on a hash ring. Keys map to the nearest clockwise point. When a GPU joins or leaves, only the keys between its neighbors need to move — the rest stay put.

```rust
use oxide_partition::ConsistentHashRing;

let ring = ConsistentHashRing::new(&[0, 1, 2], 150); // 3 GPUs, 150 virtual nodes each
let gpu = ring.partition(&my_key, 3);

// Add a GPU — most keys stay on their current GPU
let mut ring = ConsistentHashRing::new(&[0, 1, 2], 150);
ring.add_node(3);

// Remove a GPU — keys migrate only to the remaining nodes
ring.remove_node(1);
```

Best for: autoscaling GPU clusters, dynamic node membership, minimizing data movement during scale events.

## The Ternary Balance Signal

This is the core innovation. Instead of a binary "balanced/not-balanced" check, `oxide-partition` uses a **three-state signal**:

| Signal | Value | Meaning |
|--------|-------|---------|
| **Balanced** | +1 | All partitions within 10% of average — ship it |
| **Skewed** | 0 | Some partitions up to 30% off — monitor closely |
| **Hotspot** | -1 | A partition is >30% above average — action required |

```rust
use oxide_partition::{PartitionMap, TernaryBalance};

let mut map = PartitionMap::new(4); // 4 GPUs
for tensor_id in 0..100u64 {
    map.assign((tensor_id % 4) as usize, tensor_id);
}

match map.check_balance() {
    TernaryBalance::Balanced => println!("All good"),
    TernaryBalance::Skewed => println!("Keep an eye on it"),
    TernaryBalance::Hotspot => println!("Rebalance needed!"),
}
```

Why three states instead of two? Because in practice, perfect balance is neither achievable nor necessary. The skewed state gives you a warning window — you can log it, track the trend, but you don't need to interrupt processing. Only when you hit `-1` do you trigger a rebalance, avoiding unnecessary data movement.

## Hotspot Detection

Find which specific GPUs are overloaded or underloaded:

```rust
let hotspots = map.detect_hotspots(1.5);     // GPUs with >1.5× average load
let underloaded = map.detect_underloaded(0.7); // GPUs with <0.7× average load
```

## Automatic Rebalancing

Move data from hotspot partitions to underloaded ones:

```rust
use oxide_partition::rebalance;

let moved = rebalance(&mut map, 1.5); // threshold: 1.5× average
println!("Moved {} items", moved);
```

The rebalance function identifies hotspots, calculates how much excess data they hold, and moves it to underloaded partitions. It's a single-pass operation — run it periodically or when the ternary signal hits `-1`.

## Partition Map

`PartitionMap<K>` is the central data structure. It tracks which data items live on which GPU:

```rust
use oxide_partition::PartitionMap;

let mut map = PartitionMap::new(8); // 8 GPUs

// Assign data
map.assign(0, tensor_id_42);
map.assign(2, tensor_id_99);

// Query
map.len(0);          // items on GPU 0
map.total();         // total items
map.sizes();         // [size_per_gpu] vector

// Move data between GPUs
map.move_items(0, 3, 10); // move 10 items from GPU 0 to GPU 3
```

## Consistent Hashing in Depth

Traditional hash partitioning (`hash(key) % N`) has a fatal flaw for dynamic clusters: when `N` changes (a GPU joins or leaves), nearly every key remaps to a different node. That's a massive data reshuffle.

Consistent hashing solves this by placing both keys and nodes on the same hash ring. Each node occupies multiple virtual positions (controlled by the `vnodes` parameter). A key maps to the nearest node clockwise on the ring.

When node 3 joins a 3-node cluster:
- Only the keys between node 3's virtual positions and their clockwise neighbors need to move
- With 150 virtual nodes, that's roughly 1/4 of keys — not 3/4 like naive rehashing
- All other keys stay exactly where they are

When node 1 leaves:
- Its keys migrate to the next clockwise node
- No other keys move

The `vnodes` parameter controls the smoothness of distribution. More virtual nodes means more even spread but slightly more memory. 100-200 is a good default for most GPU cluster sizes.

## API Overview

| Type | Description |
|------|-------------|
| `PartitionMap<K>` | Tracks key-to-GPU assignments |
| `HashPartitioner` | Hash-based partitioning |
| `RangePartitioner<K>` | Range-based partitioning with boundaries |
| `RoundRobinPartitioner` | Sequential cycling |
| `ConsistentHashRing` | Consistent hashing ring |
| `TernaryBalance` | Three-state balance enum |
| `rebalance()` | Automatic data movement function |
| `Partitioner` trait | Implement your own strategy |

## Installation

```toml
[dependencies]
oxide-partition = "0.1"
```

## License

MIT
