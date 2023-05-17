use egui_wgpu::wgpu;

use crate::{wgsl_preprocessor::ShaderBuilder, path, gpu_resource::TextureView};

pub struct TonemapPipeline {
	bind_group_layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
	pipeline: wgpu::RenderPipeline
}

impl TonemapPipeline {
	pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					count: None,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					visibility: wgpu::ShaderStages::FRAGMENT
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					count: None,
					ty: wgpu::BindingType::Texture {
						multisampled: false,
						sample_type: wgpu::TextureSampleType::Float { filterable: true },
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::FRAGMENT
				}
			]
		});

		let shader = device.create_shader_module(ShaderBuilder::new(path::shaders().join("tonemap.wgsl")).unwrap().build());
		
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&bind_group_layout],
			..Default::default()
		});

		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			min_filter: wgpu::FilterMode::Nearest,
			mag_filter: wgpu::FilterMode::Linear,
			..Default::default()
		});

		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: None,
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: "vs_main",
				buffers: &[]
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs_main",
				targets: &[Some(wgpu::ColorTargetState {
					blend: Some(wgpu::BlendState::REPLACE),
					format,
					write_mask: wgpu::ColorWrites::ALL
				})]
			}),
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			primitive: wgpu::PrimitiveState {
				cull_mode: None,
				polygon_mode: wgpu::PolygonMode::Fill,
				topology: wgpu::PrimitiveTopology::TriangleStrip,
				..Default::default()
			}
		});

		Self {
			bind_group_layout,
			pipeline,
			sampler
		}
	}

	pub fn resolve(
		&self,
		device: &wgpu::Device,
		src: &TextureView,
		dst: &TextureView
	) -> wgpu::CommandBuffer
	{
		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&self.sampler)
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(src)
				}
			]
		});

		let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: None,
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: dst,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
					store: true
				}
			})],
			depth_stencil_attachment: None
		});
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &bind_group, &[]);
		pass.draw(0..4, 0..1);

		drop(pass);

		enc.finish()
	}
}