use cgmath::{Vector3, vec3, Matrix4, InnerSpace};
use egui_wgpu::wgpu::{self, util::DeviceExt};
use crate::bvh::Bvh;
use crate::scene::ObjectData;
use crate::wgsl_preprocessor::ShaderBuilder;

use crate::{gpu_resource::TextureView, path, scene::Camera, bvh::{bounds::Bounds, self}, math::perspective};

pub struct BvhVisual {
	lines: Option<wgpu::Buffer>,
	bind_group: wgpu::BindGroup,
	uniform_buffer: wgpu::Buffer,
	pipeline: wgpu::RenderPipeline,
}

impl BvhVisual {
	pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
		let shader = {
			let builder = ShaderBuilder::new(path::shaders().join("lines.wgsl").to_str().unwrap()).unwrap();
			device.create_shader_module(builder.build())
		};

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				count: None,
				ty: wgpu::BindingType::Buffer {
					has_dynamic_offset: false,
					min_binding_size: None,
					ty: wgpu::BufferBindingType::Uniform
				},
				visibility: wgpu::ShaderStages::VERTEX
			}]
		});

		let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			mapped_at_creation: false,
			size: std::mem::size_of::<Matrix4<f32>>() as _,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&bind_group_layout],
			..Default::default()
		});

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
					buffer: &uniform_buffer,
					offset: 0,
					size: None
				})
			}]
		});

		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: None,
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				buffers: &[wgpu::VertexBufferLayout {
					array_stride: std::mem::size_of::<Vector3<f32>>() as _,
					attributes: &[wgpu::VertexAttribute {
						format: wgpu::VertexFormat::Float32x3,
						offset: 0,
						shader_location: 0
					}],
					step_mode: wgpu::VertexStepMode::Vertex
				}],
				entry_point: "vs_main"
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs_main",
				targets: &[Some(wgpu::ColorTargetState {
					blend: Some(wgpu::BlendState::REPLACE),
					format: output_format,
					write_mask: wgpu::ColorWrites::ALL
				})]
			}),
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			primitive: wgpu::PrimitiveState {
				cull_mode: None,
				polygon_mode: wgpu::PolygonMode::Line,
				topology: wgpu::PrimitiveTopology::LineList,
				..Default::default()
			}
		});

		Self {
			lines: None,
			bind_group,
			uniform_buffer,
			pipeline
		}
	}

	pub fn set_camera(&mut self, queue: &wgpu::Queue, aspect: f32, camera: &Camera) {
		let trans = camera.transform;
		let proj = perspective(camera.fov, aspect, 0.1, 1000.0);
		// let view = 
		
		let r = trans.right();
		let u = trans.up();
		let f = trans.forward();

		let view = Matrix4::new(
			r.x, u.x, f.x, 0.0,
			r.y, u.y, f.y, 0.0,
			r.z, u.z, f.z, 0.0,
			-r.dot(trans.pos), -u.dot(trans.pos), -f.dot(trans.pos), 1.0
		);

		let mvp = proj * view;
		let mvp: [[f32; 4]; 4] = mvp.into();

		queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&mvp));
	}

	pub fn set_data_from_scene(&mut self, device: &wgpu::Device, scene_bvh: &Bvh, mesh_bvhs: &[Bvh], object_data: &ObjectData) {
		let mut lines = Vec::new();
		
		let mut stack = Vec::new();
		stack.push(scene_bvh.root.as_ref());
		
		while let Some(node) = stack.pop() {
			match node {
				bvh::Node::Interior { bounds: _, children, .. } => {
					// push_lines_from_bounds(bounds, &mut lines);
					stack.push(children[0].as_ref());
					stack.push(children[1].as_ref());
				}
				bvh::Node::Leaf { prim_range, .. } => {
					assert!(prim_range.len() == 1);

					let trans: Matrix4<f32> = object_data.transforms[prim_range.start].into();
					let mesh_bvh = &mesh_bvhs[object_data.mesh_indexes[prim_range.start]];
					let mut stack = Vec::new();
					stack.push(mesh_bvh.root.as_ref());

					while let Some(node) = stack.pop() {
						match node {
							bvh::Node::Interior { bounds: _, children, .. } => {
								stack.push(children[0].as_ref());
								stack.push(children[1].as_ref());
							}
							bvh::Node::Leaf { bounds, .. } => {
								push_lines_from_bounds(&bounds.transform(trans), &mut lines);
							}
						}
					}
				}
			}
		}

		self.lines = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: unsafe {
				std::slice::from_raw_parts(
					lines.as_ptr() as _,
					std::mem::size_of::<Vector3<f32>>() * lines.len()
				)
			},
			usage: wgpu::BufferUsages::VERTEX
		}));
	}
	
	pub fn render(&self, device: &wgpu::Device, output: &TextureView) -> wgpu::CommandBuffer {
		let lines = self.lines.as_ref().unwrap();
		
		let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: None,
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: output,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Load,
					store: true
				}
			})],
			depth_stencil_attachment: None
		});

		pass.set_pipeline(&self.pipeline);
		pass.set_vertex_buffer(0, lines.slice(..));
		pass.set_bind_group(0, &self.bind_group, &[]);
		pass.draw(0..(lines.size() / std::mem::size_of::<Vector3<f32>>() as u64) as u32, 0..1);
		
		drop(pass);

		enc.finish()
	}
}

fn push_lines_from_bounds(bounds: &Bounds<f32>, lines: &mut Vec<Vector3<f32>>) {
	let min = bounds.min;
	let max = bounds.max;

	let v0 = min;
	let v1 = vec3(max.x, min.y, min.z);
	let v2 = vec3(max.x, max.y, min.z);
	let v3 = vec3(min.x, max.y, min.z);

	let v4 = vec3(min.x, min.y, max.z);
	let v5 = vec3(max.x, min.y, max.z);
	let v6 = max;
	let v7 = vec3(min.x, max.y, max.z);

	lines.extend([
		v0, v1,
		v1, v2,
		v2, v3,
		v3, v0,

		v4, v5,
		v5, v6,
		v6, v7,
		v7, v4,

		v0, v4,
		v1, v5,
		v2, v6,
		v3, v7
	]);
}