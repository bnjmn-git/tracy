struct CameraUb {
	position: vec3<f32>,
	fov: f32,
	right: vec3<f32>,
	aperture: f32,
	up: vec3<f32>,
	focal_dist: f32,
	forward: vec3<f32>,
	bokeh_blades: i32,
}

struct CameraState {
	position: vec3<f32>,
	right: vec3<f32>,
	up: vec3<f32>,
	forward: vec3<f32>,
	fov: f32,
	aperture: f32,
	focal_dist: f32
}

@group(0) @binding(0)
var<uniform> camera: CameraUb;

@group(0) @binding(1)
var<uniform> frame_idx: u32;

struct Vertex {
	pos: vec3<f32>,
	uvx: f32,
	normal: vec3<f32>,
	uvy: f32
}

struct BvhNode {
	min: vec3<f32>,
	param0: i32,
	max: vec3<f32>,
	param1: i32,
	param2: i32,
}

@group(1) @binding(0)
var<storage, read> vertices: array<Vertex>;

@group(1) @binding(1)
var<storage, read> triangles: array<u32>;

@group(1) @binding(2)
var<storage, read> bvh: array<BvhNode>;

struct Transform {
	world: mat4x4<f32>,
	inv_world: mat4x4<f32>
}

@group(1) @binding(3)
var<storage, read> transforms: array<Transform>;

@group(1) @binding(4)
var<storage, read> materials: array<Material>;

@group(2) @binding(0)
var env_sampler: sampler;

@group(2) @binding(1)
var env_tex: texture_2d<f32>;

@group(2) @binding(2)
var env_cdf: texture_storage_2d<r32float, read>;

struct Material {
	albedo: vec3<f32>,
	opacity: f32,
	emission: vec3<f32>,
	metallic: f32,
	spec_tint: vec3<f32>,
	spec_trans: f32,
	roughness: f32,
	ior: f32,
}