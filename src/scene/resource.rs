use cgmath::{Vector3, vec3};

use crate::{mesh, math::Transform};

use super::registry::{RefCount, Stored, Id};

pub(super) trait Resource {
	fn ref_count(&self) -> &RefCount;
}

pub(super) struct Mesh {
	pub ref_count: RefCount,
	pub mesh: mesh::Mesh,
}

impl Resource for Mesh {
	fn ref_count(&self) -> &RefCount {
		&self.ref_count
	}
}

pub struct MeshDescriptor {
	pub mesh: mesh::Mesh
}

pub(super) struct Object {
	pub ref_count: RefCount,
	pub enabled: bool,
	pub transform: Transform,
	pub mesh: Stored<MeshId>,
	pub material: Stored<MaterialId>,
}

impl Resource for Object {
	fn ref_count(&self) -> &RefCount {
		&self.ref_count
	}
}

pub struct ObjectDescriptor {
	pub enabled: bool,
	pub transform: Transform,
	pub mesh: Option<MeshId>,
	pub material: Option<MaterialId>,
}

#[derive(Clone, Copy)]
pub struct MaterialDescriptor {
	pub albedo: Vector3<f32>,
	pub opacity: f32,
	pub emission: Vector3<f32>,
	pub metallic: f32,
	pub spec_tint: Vector3<f32>,
	pub spec_trans: f32,
	pub roughness: f32,
	pub ior: f32
}

impl Default for MaterialDescriptor {
	fn default() -> Self {
		Self {
			albedo: vec3(1.0, 1.0, 1.0),
			emission: vec3(0.0, 0.0, 0.0),
			ior: 1.1,
			metallic: 0.0,
			opacity: 1.0,
			roughness: 0.0,
			spec_tint: vec3(0.0, 0.0, 0.0),
			spec_trans: 0.0
		}
	}
}

pub(super) struct Material {
	pub ref_count: RefCount,
	pub desc: MaterialDescriptor,
}

impl Resource for Material {
	fn ref_count(&self) -> &RefCount {
		&self.ref_count
	}
}

pub struct DummyObject;
pub struct DummyMaterial;
pub struct DummyMesh;

pub type MeshId = Id<DummyMesh>;
pub type ObjectId = Id<DummyObject>;
pub type MaterialId = Id<DummyMaterial>;