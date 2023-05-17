use cgmath::{Matrix4, BaseFloat, Rad, Vector3, Quaternion, Matrix3, Zero, One, vec3};

pub fn perspective<S, A>(fovy: A, aspect: S, near: S, far: S) -> Matrix4<S>
where
	S: BaseFloat,
	A: Into<Rad<S>>
{
	use cgmath::num_traits::cast;
	let a = <A as Into<Rad<S>>>::into(fovy).0;
	let cot = (a * cast(0.5).unwrap()).tan().recip();
	let x = cot / aspect;
	let y = cot;
	let k = far / (far - near);
	Matrix4::new(
		x, S::zero(), S::zero(), S::zero(),
		S::zero(), y, S::zero(), S::zero(),
		S::zero(), S::zero(), k, S::one(),
		S::zero(), S::zero(), -near*k, S::zero()
	)
}

#[derive(Clone, Copy, PartialEq)]
pub struct Transform {
	pub pos: Vector3<f32>,
	pub rot: Quaternion<f32>,
	pub scale: Vector3<f32>
}

impl Default for Transform {
	fn default() -> Self {
		Self {
			pos: Vector3::zero(),
			rot: Quaternion::one(),
			scale: vec3(1.0, 1.0, 1.0)
		}
	}
}

impl Transform {
	pub fn right(&self) -> Vector3<f32> {
		self.rot * Vector3::unit_x()
	}
	pub fn up(&self) -> Vector3<f32> {
		self.rot * Vector3::unit_y()
	}
	pub fn forward(&self) -> Vector3<f32> {
		self.rot * Vector3::unit_z()
	}
}

impl From<Transform> for Matrix4<f32> {
	fn from(value: Transform) -> Self {
		let mut m: Matrix3<f32> = value.rot.into();
		m.x *= value.scale.x;
		m.y *= value.scale.y;
		m.z *= value.scale.z;
		let mut m: Matrix4<f32> = m.into();
		m.w = value.pos.extend(1.0);
		m
	}
}