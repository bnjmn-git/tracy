#![allow(unused)]

use std::ops::{Range, Index, IndexMut};

use cgmath::Vector3;

use self::bounds::{Bounds, Axis};

pub mod bounds;

#[derive(Clone, Copy)]
pub enum SplitMethod {
	Sah,
	Middle,
	EqualCounts
}

#[derive(Clone, Copy)]
pub struct Primitive {
	pub bounds: Bounds<f32>,
	pub id: usize
}

pub enum Node {
	Interior {
		bounds: Bounds<f32>,
		children: [Box<Node>; 2],
		split_axis: Axis
	},
	Leaf {
		bounds: Bounds<f32>,
		prim_range: Range<usize>
	}
}

impl Node {
	pub fn bounds(&self) -> Bounds<f32> {
		match self {
			Self::Interior { bounds, .. } => *bounds,
			Self::Leaf { bounds, .. } => *bounds
		}
	}
}

pub struct Bvh {
	pub root: Box<Node>,
	pub ids: Vec<usize>,
	pub num_nodes: usize
}

pub fn build(
	method: SplitMethod,
	max_leaf_primitives: usize,
	mut primitives: Vec<Primitive>
) -> Bvh
{
	let mut total_nodes = 0;
	let mut ordered_prims = Vec::new();
	let root = build_recursive(
		method,
		&mut primitives,
		max_leaf_primitives,
		&mut total_nodes,
		&mut ordered_prims
	);

	Bvh {
		root,
		ids: primitives.into_iter().map(|x| x.id).collect(),
		num_nodes: total_nodes
	}
}

fn create_leaf_node(
	bounds: Bounds<f32>,
	primitives: &[Primitive],
	ordered_prims: &mut Vec<Primitive>
) -> Node
{
	let start = ordered_prims.len();
	let end = primitives.len() + start;
	ordered_prims.extend_from_slice(primitives);

	Node::Leaf {
		bounds,
		prim_range: start..end
	}
}

fn build_recursive(
	method: SplitMethod,
	primitives: &mut [Primitive],
	max_leaf_primitives: usize,
	total_nodes: &mut usize,
	ordered_prims: &mut Vec<Primitive>
) -> Box<Node>
{
	*total_nodes += 1;

	let mut bounds = Bounds::default();
	for prim in primitives.iter() {
		bounds = bounds.union(prim.bounds);
	}

	// let ext = bounds.extent();
	// if ext.x < 0.1 {
	// 	bounds.min.x -= 0.1;
	// 	bounds.max.x += 0.1;
	// }
	// if ext.y < 0.1 {
	// 	bounds.min.y -= 0.1;
	// 	bounds.max.y += 0.1;
	// }
	// if ext.z < 0.1 {
	// 	bounds.min.z -= 0.1;
	// 	bounds.max.z += 0.1;
	// }

	// assert!(bounds.is_valid());

	let prims_count = primitives.len();
	if prims_count <= max_leaf_primitives {
		Box::new(create_leaf_node(
			bounds,
			primitives,
			ordered_prims
		))
	} else {
		let centroid_bounds = primitives.iter().fold(
			Bounds::default(),
			|b, p| {
				b.union_with_point(p.bounds.centroid())
			}
		);

		let axis = centroid_bounds.max_axis();
		let mid = match method {
			SplitMethod::Sah if prims_count < 4 => partition_equal_counts(axis, primitives),
			SplitMethod::Sah => partition_sah(bounds, centroid_bounds, max_leaf_primitives, axis, primitives),
			SplitMethod::Middle => {
				if let Some(mid) = partition_middle(centroid_bounds, axis, primitives) {
					Some(mid)
				} else {
					partition_equal_counts(axis, primitives)
				}
			},
			SplitMethod::EqualCounts => partition_equal_counts(axis, primitives)
		};

		if let Some(mid) = mid {
			let children = [
				build_recursive(
					method,
					&mut primitives[..mid],
					max_leaf_primitives,
					total_nodes,
					ordered_prims
				),
				build_recursive(
					method, 
					&mut primitives[mid..], 
					max_leaf_primitives, 
					total_nodes,
					ordered_prims 
				)
			];

			Box::new(Node::Interior {
				bounds: children[0].bounds().union(children[1].bounds()),
				children,
				split_axis: axis
			})
		} else {
			Box::new(create_leaf_node(
				bounds,
				primitives,
				ordered_prims
			))
		}
	}
}

/// Partially sorts an array such that the nth element becomes the element
/// that would be there if the array were fully sorted.
/// * arr - The array to find the nth element of
/// * f - Closure that takes two elements of an array and returns true
/// if the first element should appear before the second
fn nth_element<T, F>(arr: &mut [T], n: usize, f: F)
where
	F: Fn(&T, &T) -> bool
{
	let parti = |arr: &mut [T]| -> usize {
		let pivot = arr.len() - 1;
		let mut i = 0;
		for j in 0..arr.len() - 1 {
			if f(&arr[j], &arr[pivot]) {
				arr.swap(i, j);
				i += 1;
			}
		}

		arr.swap(i, pivot);
		i
	};

	let pi = parti(arr);
	if n < pi {
		nth_element(&mut arr[..pi], n, f)
	} else if n > pi {
		nth_element(&mut arr[pi+1..], n - pi - 1, f)
	}
}

// Partitions primitives into equal counts by rearranging them such that the
// first half will have a centroid along the desired axis be less than those
// in the second half. Returns the index of the first element of the second half
fn partition_equal_counts(
	axis: Axis,
	primitives: &mut [Primitive]
) -> Option<usize>
{
	let mid = primitives.len() / 2;
	nth_element(primitives, mid, |a, b| {
		a.bounds.centroid()[axis] < b.bounds.centroid()[axis]
	});

	Some(mid)
}

// Partitions primitives into two parts such that the first partition contains
// primitives whose centroid is less than the centroid of all primitives along
// a given axis, and the opposite is true for the second partition.
fn partition_middle(
	centroid_bounds: Bounds<f32>,
	axis: Axis,
	primitives: &mut [Primitive]
) -> Option<usize>
{
	let mid_point = (centroid_bounds.min[axis] + centroid_bounds.max[axis]) * 0.5;
	primitives.sort_by_key(|p| {
		p.bounds.centroid()[axis] >= mid_point
	});

	let mid = primitives.partition_point(|p| {
		p.bounds.centroid()[axis] < mid_point
	});

	if mid != 0 && mid != primitives.len() {
		Some(mid)
	} else {
		None
	}
}

#[derive(Clone, Copy, Default)]
struct Bucket {
	count: u32,
	bounds: Bounds<f32>
}

fn partition_sah(
	bounds: Bounds<f32>,
	centroid_bounds: Bounds<f32>,
	max_leaf_primitives: usize,
	axis: Axis,
	primitives: &mut [Primitive]
) -> Option<usize>
{
	const NUM_BUCKETS: usize = 24;
	let mut buckets = [Bucket::default(); NUM_BUCKETS];

	let calc_bucket_index = |p: &Primitive| {
		let mut b =
			(NUM_BUCKETS as f32 * centroid_bounds.offset(p.bounds.centroid())[axis]).trunc() as usize;
		if b == NUM_BUCKETS {
			b -= 1;
		}
		
		b
	};

	for prim in primitives.iter() {
		let b = calc_bucket_index(prim);
		buckets[b].count += 1;
		buckets[b].bounds = buckets[b].bounds.union(prim.bounds);
	}

	let mut costs = [0.0; NUM_BUCKETS - 1];
	for i in 0..NUM_BUCKETS - 1 {
		let mut b0 = Bounds::default();
		let mut b1 = Bounds::default();
		let mut count0 = 0;
		let mut count1 = 0;

		for b in &buckets[..=i] {
			b0 = b0.union(b.bounds);
			count0 += b.count;
		}

		for b in &buckets[i+1..NUM_BUCKETS] {
			b1 = b1.union(b.bounds);
			count1 += b.count;
		}

		costs[i] = 0.125 + (
				count0 as f32 * b0.surface_area() +
				count1 as f32 * b1.surface_area()
			) / bounds.surface_area();
	}

	let (min_cost_split_bucket, &min_cost) = costs
		.iter()
		.enumerate()
		.min_by(|a, b| {
			a.1.total_cmp(b.1)
		})
		.unwrap();

	let leaf_cost = primitives.len() as f32;
	if primitives.len() > max_leaf_primitives || min_cost < leaf_cost {
		primitives.sort_by_key(|p| {
			calc_bucket_index(p) > min_cost_split_bucket
		});

		let mid = primitives.partition_point(|p| {
			calc_bucket_index(p) <= min_cost_split_bucket
		});

		Some(mid)
	} else {
		None
	}
}

impl<S> Index<Axis> for Vector3<S> {
    type Output = S;

    fn index(&self, index: Axis) -> &Self::Output {
        match index {
			Axis::X => &self.x,
			Axis::Y => &self.y,
			Axis::Z => &self.z
		}
    }
}

impl<S> IndexMut<Axis> for Vector3<S> {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        match index {
			Axis::X => &mut self.x,
			Axis::Y => &mut self.y,
			Axis::Z => &mut self.z
		}
    }
}