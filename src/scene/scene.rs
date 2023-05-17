use std::{collections::{HashMap, hash_map}, hash::Hash};

use cgmath::{Rad, Deg, Vector3, vec3, Zero, Vector2};

use crate::{math::Transform, bvh::{Bvh, self, bounds::Bounds}, mesh::{Vertex, self}};

use super::{
	registry::{Registry, RefCount, Stored},
	resource::{Mesh, MeshId, Object, ObjectId, Material, MaterialId, MaterialDescriptor},
	MeshDescriptor,
	ObjectDescriptor
};

#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
	pub transform: Transform,
	pub focal_distance: f32,
	pub aperture: f32,
	pub fov: Rad<f32>,
	pub bokeh_blades: i32,
}

impl Default for Camera {
	fn default() -> Self {
		Self {
			transform: Transform::default(),
			focal_distance: 10.0,
			aperture: 0.01,
			fov: Deg(60.0).into(),
			bokeh_blades: 0
		}
	}
}

bitflags::bitflags! {
	struct UpdateFlags : u32 {
		const MESH = 1 << 0;
		const OBJECT = 1 << 1;
		const BVH = 1 << 2;
		const CAMERA = 1 << 3;
		const MATERIAL = 1 << 4;
	}
}

#[derive(Default)]
struct Cache {
	mesh_to_idx_offset: HashMap<MeshId, usize>,
	material_to_idx: HashMap<MaterialId, usize>,
}

/// Holds all resources in a scene
pub struct Scene {
	camera: Camera,
	update_flags: UpdateFlags,
	meshes: Registry<Mesh, MeshId>,
	objects: Registry<Object, ObjectId>,
	materials: Registry<Material, MaterialId>,
	cache: Cache,

	default_mesh: Stored<MeshId>,
	default_material: Stored<MaterialId>,
}

pub struct ObjectData {
	pub material_indexes: Vec<usize>,
	pub mesh_indexes: Vec<usize>,
	pub mesh_bvh_index_offsets: Vec<usize>,
	pub transforms: Vec<Transform>
}

pub enum Update {
	Camera(Camera),
	Mesh(Vec<Vertex>, Vec<u32>),
	Material(Vec<MaterialDescriptor>),
	Bvh {
		scene_bvh: Bvh,
		mesh_bvhs: Vec<Bvh>,
		object_data: ObjectData,
	}
}

#[allow(unused)]
impl Scene {
	pub fn new() -> Self {
		let mut meshes = Registry::new();
		let mut objects = Registry::new();
		let mut materials = Registry::new();

		let default_mesh = {
			let ref_count = RefCount::new();
			let id = meshes.register(Mesh {
				mesh: mesh::Mesh {
					name: "Default".into(),
					vertices: vec![Vertex {
						pos: Vector3::zero(),
						normal: Vector3::zero(),
						uv: Vector2::zero(),
					}],
					indices: vec![0, 0, 0]
				},
				ref_count: ref_count.clone()
			});

			Stored {
				id,
				ref_count
			}
		};

		let default_material = {
			let ref_count = RefCount::new();
			let id: MaterialId = materials.register(Material {
				ref_count: ref_count.clone(),
				desc: MaterialDescriptor {
					albedo: vec3(1.0, 1.0, 1.0),
					emission: vec3(0.0, 0.0, 0.0),
					ior: 1.1,
					metallic: 0.0,
					opacity: 1.0,
					roughness: 0.0,
					spec_tint: vec3(0.0, 0.0, 0.0),
					spec_trans: 0.0
				}
			});

			Stored {
				id,
				ref_count
			}
		};

		objects.register(Object {
			enabled: true,
			mesh: default_mesh.clone(),
			material: default_material.clone(),
			ref_count: RefCount::new(),
			transform: Transform::default()
		});
		
		Self {
			camera: Camera::default(),
			update_flags: UpdateFlags::all(),
			meshes,
			objects,
			materials,
			cache: Cache::default(),
			default_material,
			default_mesh
		}
	}

	pub fn add_mesh(&mut self, desc: MeshDescriptor) -> MeshId {
		self.update_flags |= UpdateFlags::MESH;
		self.meshes.register(Mesh {
			ref_count: RefCount::new(),
			mesh: desc.mesh
		})
	}
	
	pub fn add_object(&mut self, desc: ObjectDescriptor) -> ObjectId {
		self.update_flags |= UpdateFlags::OBJECT;

		let material = desc.material.map_or(self.default_material.clone(), |id| {
			self.materials.get_stored(id).unwrap()
		});
		
		let mesh = desc.mesh.map_or(self.default_mesh.clone(), |id| {
			self.meshes.get_stored(id).unwrap()
		});

		self.objects.register(Object {
			ref_count: RefCount::new(),
			enabled: desc.enabled,
			transform: desc.transform,
			material,
			mesh
		})
	}

	pub fn add_material(&mut self, desc: MaterialDescriptor) -> MaterialId {
		self.update_flags |= UpdateFlags::MATERIAL;
		self.materials.register(Material {
			desc,
			ref_count: RefCount::new()
		})
	}

	pub fn remove_mesh(&mut self, id: MeshId) {
		let mesh = self.meshes.unregister(id);
		self.update_flags |= UpdateFlags::MESH;
		
		if mesh.ref_count.load() > 1 {
			// The only resource that could still be referencing this
			// mesh are objects, so flag those as well.
			self.update_flags |= UpdateFlags::OBJECT;
		}
	}

	pub fn remove_object(&mut self, id: ObjectId) {
		let object = self.objects.unregister(id);
		assert!(object.ref_count.load() == 1);
		self.update_flags |= UpdateFlags::OBJECT;
	}
	
	pub fn remove_material(&mut self, id: MaterialId) {
		let material = self.materials.unregister(id);
		self.update_flags |= UpdateFlags::MATERIAL;
		if material.ref_count.load() > 1 {
			// The only resource that could still be referencing this
			// mesh are objects, so flag those as well.
			self.update_flags |= UpdateFlags::OBJECT;
		}
	}

	pub fn material(&self, id: MaterialId) -> MaterialDescriptor {
		let material = self.materials.get(id).unwrap();
		material.desc
	}

	pub fn update_camera(&mut self, f: impl FnOnce(&mut Camera)) {
		f(&mut self.camera);
		self.update_flags |= UpdateFlags::CAMERA;
	}

	pub fn update_material(&mut self, id: MaterialId, f: impl FnOnce(&mut MaterialDescriptor)) {
		let material = self.materials.get_mut(id).unwrap();
		f(&mut material.desc);
		self.update_flags |= UpdateFlags::MATERIAL;
	}

	pub fn update_object(&mut self, id: ObjectId, f: impl FnOnce(&mut ObjectDescriptor)) {
		let mut desc = self.object(id);
		f(&mut desc);

		let material = desc.material.map_or(self.default_material.clone(), |id| {
			let material = self.materials.get(id).unwrap();
			Stored {
				id,
				ref_count: material.ref_count.clone()
			}
		});
		
		let mesh = desc.mesh.map_or(self.default_mesh.clone(), |id| {
			let mesh = self.meshes.get(id).unwrap();
			Stored {
				id,
				ref_count: mesh.ref_count.clone()
			}
		});

		let object = self.objects.get_mut(id).unwrap();
		object.enabled = desc.enabled;
		object.transform = desc.transform;
		object.material = material;
		object.mesh = mesh;

		self.update_flags |= UpdateFlags::OBJECT;
	}

	pub fn object(&self, id: ObjectId) -> ObjectDescriptor {
		let mut object = self.objects.get(id).unwrap();
		let mut desc = ObjectDescriptor {
			enabled: object.enabled,
			material: if object.material.id == self.default_material.id {
				None
			} else {
				Some(object.material.id)
			},
			mesh: if object.mesh.id == self.default_mesh.id {
				None
			} else {
				Some(object.mesh.id)
			},
			transform: object.transform
		};

		desc
	}

	pub fn camera(&self) -> Camera {
		self.camera
	}

	/// Applies any changes to the scene and passes the updated data to the relevant
	/// callbacks.
	/// 
	/// Returns `true` if there were any updates, otherwise `false`.
	pub fn apply_updates(
		&mut self,
		mut f: impl FnMut(Update)
	) -> bool
	{
		if self.update_flags.intersects(UpdateFlags::CAMERA) {
			f(Update::Camera(self.camera));
		}

		if self.update_flags.contains(UpdateFlags::MESH) {
			let (num_vertices, num_indices) = self.meshes.iter().fold(
				(0, 0),
				|(num_vertices, num_triangles), (_, mesh)| {
					(
						num_vertices + mesh.mesh.vertices.len(),
						num_triangles + mesh.mesh.indices.len()
					)
				}
			);

			let mut vertices = Vec::with_capacity(num_vertices);
			let mut indices = Vec::with_capacity(num_indices);

			let cache = &mut self.cache;
			cache.mesh_to_idx_offset.clear();

			for (id, mesh) in self.meshes.iter() {
				let mesh = &mesh.mesh;
				let idx_offset = vertices.len() as u32;

				cache.mesh_to_idx_offset.insert(id, indices.len());
				vertices.extend(mesh.vertices.iter());
				indices.extend(mesh.indices.iter().map(|i| *i + idx_offset));
			}

			f(Update::Mesh(vertices, indices));
		}

		if self.update_flags.contains(UpdateFlags::MATERIAL) {
			let cache = &mut self.cache;
			cache.material_to_idx.clear();

			f(Update::Material(self.materials.iter().enumerate().map(|(i, m)| {
				cache.material_to_idx.insert(m.0, i);
				m.1.desc
			}).collect()));
		}

		// Restore defaults if resource references are invalid
		if self.update_flags.contains(UpdateFlags::OBJECT) {
			self.objects.iter_mut().for_each(|(_, object)| {
				if !self.meshes.contains(object.mesh.id) {
					object.mesh = self.default_mesh.clone();
				}
				if !self.materials.contains(object.material.id) {
					object.material = self.default_material.clone();
				}
			});
		}

		if self.update_flags.intersects(UpdateFlags::BVH | UpdateFlags::OBJECT) {
			let mut material_indexes = Vec::new();
			let mut mesh_indexes = Vec::new();
			let mut mesh_bvh_index_offsets = Vec::new();
			let mut mesh_bvhs = Vec::new();
			let mut transforms = Vec::new();
			
			let scene_bvh = {
				let mut primitives = Vec::new();
				let mut mesh_to_bvh_idx = HashMap::new();

				self.objects.iter()
				.filter(|(_, object)| object.enabled)
				.enumerate()
				.for_each(|(i, (_id, object))| {
					let mesh = &self.meshes.get(object.mesh.id).unwrap().mesh;
					let mesh_bvh_idx = match mesh_to_bvh_idx.entry(object.mesh.id) {
						hash_map::Entry::Occupied(e) => {
							*e.get()
						}
						hash_map::Entry::Vacant(e) => {
							let idx_offset = self.cache.mesh_to_idx_offset[&object.mesh.id];
							let vertices = &mesh.vertices;
							let indices = &mesh.indices;
							assert!(mesh.indices.len() % 3 == 0);
							let num_tris = mesh.indices.len() / 3;

							let mesh_bvh = {
								let mut primitives = Vec::with_capacity(num_tris);
								
								for i in 0..num_tris {
									let bounds = Bounds::from_points([
										vertices[indices[i*3+0] as usize].pos,
										vertices[indices[i*3+1] as usize].pos,
										vertices[indices[i*3+2] as usize].pos
									]);
		
									primitives.push(bvh::Primitive {
										bounds,
										id: i
									});
								}
								
								bvh::build(bvh::SplitMethod::Sah, 16, primitives)
							};

							let idx = mesh_bvhs.len();
							mesh_bvhs.push(mesh_bvh);
							mesh_bvh_index_offsets.push(idx_offset);
							*e.insert(idx)
						}
					};

					mesh_indexes.push(mesh_bvh_idx);
					material_indexes.push(self.cache.material_to_idx[&object.material.id]);
					transforms.push(object.transform);

					let bounds = mesh_bvhs[mesh_bvh_idx].root.bounds().transform(object.transform);

					primitives.push(bvh::Primitive {
						bounds,
						id: i
					});
				});

				bvh::build(bvh::SplitMethod::Sah, 1, primitives)
			};

			f(Update::Bvh {
				scene_bvh,
				mesh_bvhs,
				object_data: ObjectData {
					material_indexes,
					mesh_indexes,
					mesh_bvh_index_offsets,
					transforms
				}
			});
		}

		let had_updates = !self.update_flags.is_empty();
		self.update_flags = UpdateFlags::empty();
		had_updates
	}
}