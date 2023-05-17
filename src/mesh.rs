use std::{path::Path, fmt::Debug};

use cgmath::{Vector3, Vector2, vec3, vec2};



#[derive(Clone, Copy)]
pub struct Vertex {
	pub pos: Vector3<f32>,
	pub normal: Vector3<f32>,
	pub uv: Vector2<f32>
}

pub struct Mesh {
	pub name: String,
	pub vertices: Vec<Vertex>,
	pub indices: Vec<u32>
}

#[derive(Debug)]
pub enum MeshLoadError {
	MismatchedNormals,
	TinyObj(tobj::LoadError)
}

impl Mesh {
	pub fn from_obj(filename: impl AsRef<Path> + Debug) -> Result<Vec<Self>, MeshLoadError> {
		let (model, _) = tobj::load_obj(filename, &tobj::LoadOptions {
			ignore_lines: true,
			ignore_points: true,
			single_index: true,
			triangulate: true
		}).map_err(|e| MeshLoadError::TinyObj(e))?;

		let mut meshes = Vec::new();
		for tobj::Model { name, mesh } in model.into_iter() {
			if mesh.normals.is_empty() || mesh.positions.len() != mesh.normals.len() {
				return Err(MeshLoadError::MismatchedNormals);
			}

			let num_vertices = mesh.positions.len() / 3;

			let mut vertices = Vec::with_capacity(num_vertices);
			for i in 0..num_vertices {
				vertices.push(Vertex {
					pos: vec3(
						mesh.positions[i*3+0],
						mesh.positions[i*3+1],
						mesh.positions[i*3+2]
					),
					normal: vec3(
						mesh.normals[i*3+0],
						mesh.normals[i*3+1],
						mesh.normals[i*3+2]
					),
					uv: if !mesh.texcoords.is_empty() {
						vec2(
							mesh.texcoords[i*2+0],
							mesh.texcoords[i*2+1]
						)
					} else {
						vec2(0.0, 0.0)
					}
				});
			}
			
			meshes.push(Mesh {
				name,
				indices: mesh.indices,
				vertices
			});
		}

		Ok(meshes)
	}
}