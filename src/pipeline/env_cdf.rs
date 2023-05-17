use egui_wgpu::wgpu;

use crate::{wgsl_preprocessor::ShaderBuilder, path, gpu_resource::TextureView};

pub struct EnvCdfPipeline {
	pipeline: wgpu::ComputePipeline,
	bind_group_layout: wgpu::BindGroupLayout
}

impl EnvCdfPipeline {
	pub fn new(device: &wgpu::Device) -> Self {
		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					count: None,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					visibility: wgpu::ShaderStages::COMPUTE
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					count: None,
					ty: wgpu::BindingType::Texture {
						multisampled: false,
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
				wgpu::BindGroupLayoutEntry {
					binding: 2,
					count: None,
					ty: wgpu::BindingType::StorageTexture {
						access: wgpu::StorageTextureAccess::ReadWrite,
						format: wgpu::TextureFormat::R32Float,
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
			]
		});

		let shader = device.create_shader_module(
			ShaderBuilder::new(path::shaders().join("env_cdf.wgsl")).unwrap().build()
		);

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&bind_group_layout],
			..Default::default()
		});

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			entry_point: "main",
			layout: Some(&pipeline_layout),
			module: &shader
		});

		Self {
			bind_group_layout,
			pipeline
		}
	}

	pub fn resolve(
		&self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		src: &TextureView,
		sampler: &wgpu::Sampler,
		dst: &TextureView
	) {
		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(sampler)
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(src)
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::TextureView(dst)
				},
			]
		});

		let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
		pass.set_bind_group(0, &bind_group, &[]);
		pass.set_pipeline(&self.pipeline);
		pass.dispatch_workgroups(f32::ceil(dst.texture_desc.size.width as f32 / 32.0) as u32, 1, 1);
		
		drop(pass);

		queue.submit([enc.finish()]);
	}
}