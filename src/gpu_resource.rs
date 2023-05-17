use std::ops::Deref;

use egui_wgpu::wgpu;

pub struct Texture {
	pub raw: wgpu::Texture,
	pub desc: wgpu::TextureDescriptor<'static>
}

impl Texture {
	pub fn new(device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'static>) -> Self {
		Self {
			raw: device.create_texture(desc),
			desc: desc.clone()
		}
	}

	pub fn from_raw(raw: wgpu::Texture, desc: &wgpu::TextureDescriptor<'static>) -> Self {
		Self {
			raw,
			desc: desc.clone()
		}
	}
}

impl Deref for Texture {
	type Target = wgpu::Texture;
	fn deref(&self) -> &Self::Target {
		&self.raw
	}
}

pub struct TextureView {
	pub raw: wgpu::TextureView,
	pub texture_desc: wgpu::TextureDescriptor<'static>,
	pub view_desc: wgpu::TextureViewDescriptor<'static>
}

impl TextureView {
	pub fn new(texture: &Texture, desc: &wgpu::TextureViewDescriptor<'static>) -> Self {
		Self {
			raw: texture.create_view(desc),
			texture_desc: texture.desc.clone(),
			view_desc: desc.clone()
		}
	}
}

impl Deref for TextureView {
    type Target = wgpu::TextureView;

    fn deref(&self) -> &Self::Target {
		&self.raw
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
	pub position: [f32; 3],
	pub fov: f32,
	pub right: [f32; 3],
	pub aperture: f32,
	pub up: [f32; 3],
	pub focal_dist: f32,
	pub forward: [f32; 3],
	pub bokeh_blades: i32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
	/// The bounds min.
	pub min: [f32; 3],

	/// If interior node, the index into left child.
	/// If scene leaf node, the index into the root mesh bvh node.
	/// If mesh leaf node, the base offset into the triangles array.
	pub param0: i32,

	/// The bounds max.
	pub max: [f32; 3],

	/// If interior node, then index into right child.
	/// If scene leaf node, then index into materials array.
	/// If mesh leaf node, then the number of triangles.
	pub param1: i32,

	/// Determines what type of bvh we are dealing with.
	/// 0 = interior node of either scene or mesh bvh
	/// >0 = mesh bvh leaf node
	/// <0 = scene bvh leaf node storing -(object_idx) - 1
	pub param2: i32,
	arr_pad: [i32; 3]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
	pub pos: [f32; 3],
	pub uvx: f32,
	pub normal: [f32; 3],
	pub uvy: f32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransform {
	pub world: [[f32; 4]; 4],
	pub inv_world: [[f32; 4]; 4]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
	pub albedo: [f32; 3],
	pub opacity: f32,
	pub emission: [f32; 3],
	pub metallic: f32,
	pub spec_tint: [f32; 3],
	pub spec_trans: f32,
	pub roughness: f32,
	pub ior: f32,
	_arr_pad: [f32; 2]
}