use cgmath::{Vector3, vec3, Bounded, BaseNum, Matrix4};

#[derive(Clone, Copy)]
pub enum Axis {
	X,
	Y,
	Z
}

pub trait MinMax: BaseNum {
	fn min(self, other: Self) -> Self;
	fn max(self, other: Self) -> Self;
}

macro_rules! min_max_impl {
	($ty:ty) => {
		impl MinMax for $ty {
			fn min(self, other: Self) -> Self {
				<Self as Ord>::min(self, other)
			}

			fn max(self, other: Self) -> Self {
				<Self as Ord>::max(self, other)
			}
		}
	};
}

min_max_impl!(i8);
min_max_impl!(i16);
min_max_impl!(i32);
min_max_impl!(i64);

min_max_impl!(u8);
min_max_impl!(u16);
min_max_impl!(u32);
min_max_impl!(u64);

impl MinMax for f32 {
	fn min(self, other: Self) -> Self {
		self.min(other)
	}

	fn max(self, other: Self) -> Self {
		self.max(other)
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds<S> {
	pub min: Vector3<S>,
	pub max: Vector3<S>
}

impl<S: MinMax + cgmath::num_traits::Bounded> Default for Bounds<S> {
	fn default() -> Self {
		Self {
			min: vec3(S::max_value(), S::max_value(), S::max_value()),
			max: vec3(S::min_value(), S::min_value(), S::min_value())
		}
	}
}

impl<S: MinMax + cgmath::num_traits::Bounded> Bounds<S> {
	pub fn new<T: Into<Vector3<S>>>(min: T, max: T) -> Self {
		Self {
			min: min.into(),
			max: max.into()
		}
	}

	pub fn from_points<T>(points: impl IntoIterator<Item = T>) -> Self
	where
		T: Into<Vector3<S>>
	{
		let mut bounds = Bounds::default();
		for p in points {
			bounds = bounds.union_with_point(p.into());
		}
		bounds
	}

	pub fn union(self, other: Self) -> Self {
		Self {
			min: self.min.zip(other.min, |a, b| a.min(b)),
			max: self.max.zip(other.max, |a, b| a.max(b))
		}
	}

	pub fn union_with_point(self, point: Vector3<S>) -> Self {
		Self {
			min: self.min.zip(point, |a, b| a.min(b)),
			max: self.max.zip(point, |a, b| a.max(b))
		}
	}

	pub fn intersection(self, other: Self) -> Self {
		Self {
			min: self.min.zip(other.min, |a, b| a.max(b)),
			max: self.max.zip(other.max, |a, b| a.min(b))
		}
	}

	pub fn is_valid(&self) -> bool {
		let v = self.min.zip(self.max, |a, b| a <= b);
		v.x && v.y && v.z
	}

	pub fn centroid(&self) -> Vector3<S> {
		(self.min + self.max).map(|x| {
			S::from(x.to_f64().unwrap() * 0.5).unwrap()
		})
	}

	pub fn extent(&self) -> Vector3<S> {
		self.max - self.min
	}

	pub fn max_axis(&self) -> Axis {
		let ext = self.extent();
		if ext.x > ext.y && ext.x > ext.z {
			Axis::X
		} else if ext.y > ext.z {
			Axis::Y
		} else {
			Axis::Z
		}
	}

	pub fn surface_area(&self) -> S {
		let ext = self.extent();
		S::from(2.0).unwrap() * (ext.x*ext.y + ext.y*ext.z + ext.z*ext.y)
	}

	pub fn offset(&self, p: Vector3<S>) -> Vector3<S> {
		let mut o = p - self.min;
		let ext = self.extent();

		if self.max.x > self.min.x { o.x /= ext.x; }
		if self.max.y > self.min.y { o.y /= ext.y; }
		if self.max.z > self.min.z { o.z /= ext.z; }
		o
	}

	pub fn transform(&self, matrix: impl Into<Matrix4<S>>) -> Bounds<S>
		where S: cgmath::BaseFloat
	{
		let matrix = matrix.into();
		// let min = (matrix * self.min.extend(S::one())).truncate();
		// let max = (matrix * self.max.extend(S::one())).truncate();
		let min = self.min;
		let max = self.max;

		let right = matrix.x.truncate();
		let up = matrix.y.truncate();
		let forward = matrix.z.truncate();
		let pos = matrix.w.truncate();

		let xa = right * min.x;
		let xb = right * max.x;
		let ya = up * min.y;
		let yb = up * max.y;
		let za = forward * min.z;
		let zb = forward * max.z;

		let min = pos
		+ xa.zip(xb, |a, b| <S as MinMax>::min(a, b))
		+ ya.zip(yb, |a, b| <S as MinMax>::min(a, b))
		+ za.zip(zb, |a, b| <S as MinMax>::min(a, b));

		let max = pos
		+ xa.zip(xb, |a, b| <S as MinMax>::max(a, b))
		+ ya.zip(yb, |a, b| <S as MinMax>::max(a, b))
		+ za.zip(zb, |a, b| <S as MinMax>::max(a, b));

		Bounds {
			min,
			max
		}
	}
}