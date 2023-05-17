use egui_wgpu::wgpu;

use crate::{wgsl_preprocessor::ShaderBuilder, path, gpu_resource::TextureView};

pub struct AccumPipeline {
	persistent_bind_group: wgpu::BindGroup,
	per_resolve_bgl: wgpu::BindGroupLayout,
	pipeline: wgpu::ComputePipeline,
	iterations_buffer: wgpu::Buffer,
}

impl AccumPipeline {
	pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
		let persistent_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					count: None,
					ty: wgpu::BindingType::Buffer {
						has_dynamic_offset: false,
						min_binding_size: None,
						ty: wgpu::BufferBindingType::Uniform
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					count: None,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					visibility: wgpu::ShaderStages::COMPUTE
				}
			]
		});

		let per_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					count: None,
					ty: wgpu::BindingType::Texture {
						multisampled: false,
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					count: None,
					ty: wgpu::BindingType::StorageTexture {
						access: wgpu::StorageTextureAccess::ReadWrite,
						format,
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::COMPUTE
				}
			]
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&persistent_bgl, &per_resolve_bgl],
			..Default::default()
		});

		let shader = {
			let builder = ShaderBuilder::new(path::shaders().join("accum.wgsl").to_str().unwrap()).unwrap();
			device.create_shader_module(builder.build())
		};

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			entry_point: "main",
			label: None,
			layout: Some(&pipeline_layout),
			module: &shader
		});

		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			min_filter: wgpu::FilterMode::Nearest,
			mag_filter: wgpu::FilterMode::Linear,
			..Default::default()
		});

		let iterations_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			mapped_at_creation: false,
			size: std::mem::size_of::<u32>() as _,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM
		});

		let persistent_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
						buffer: &iterations_buffer,
						offset: 0,
						size: None
					})
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::Sampler(&sampler)
				}
			],
			label: None,
			layout: &persistent_bgl
		});

		Self {
			persistent_bind_group,
			iterations_buffer,
			per_resolve_bgl,
			pipeline,
		}
	}

	pub fn resolve(
		&self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		src: &TextureView,
		dst: &TextureView,
		iterations: u32,
		clear_dst: bool
	) -> wgpu::CommandBuffer {
		queue.write_buffer(&self.iterations_buffer, 0, bytemuck::bytes_of(&iterations));

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.per_resolve_bgl,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(src)
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(dst)
				}
			]
		});

		if clear_dst {
			let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
			let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: None,
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: dst,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
						store: false
					}
				})],
				depth_stencil_attachment: None
			});

			drop(pass);

			queue.submit([enc.finish()]);
		}

		let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
			..Default::default()
		});

		pass.set_bind_group(0, &self.persistent_bind_group, &[]);
		pass.set_bind_group(1, &bind_group, &[]);
		pass.set_pipeline(&self.pipeline);

		let output_size = (dst.texture_desc.size.width, dst.texture_desc.size.height);
		pass.dispatch_workgroups(
			(output_size.0 as f32 / 16.0).ceil() as u32,
			(output_size.1 as f32 / 16.0).ceil() as u32,
			1
		);

		drop(pass);

		enc.finish()
	}
}