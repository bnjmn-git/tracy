use bytemuck::Zeroable;
use egui_wgpu::wgpu::{self, util::DeviceExt};

use crate::{path, gpu_resource::{TextureView, Texture, GpuVertex, GpuBvhNode, GpuTransform, GpuMaterial, GpuCamera}, wgsl_preprocessor, pipeline::env_cdf::EnvCdfPipeline};

struct GeometryResources {
	bind_group_layout: wgpu::BindGroupLayout,
	bind_group: wgpu::BindGroup,

	vertices: wgpu::Buffer,
	triangles: wgpu::Buffer,
	bvhs: wgpu::Buffer,
	transforms: wgpu::Buffer,
	materials: wgpu::Buffer,
}

impl GeometryResources {
	pub fn new(device: &wgpu::Device) -> Self {
		let bgl_entries: Vec<_> = (0..5).map(|i| {
			wgpu::BindGroupLayoutEntry {
				binding: i,
				count: None,
				ty: wgpu::BindingType::Buffer {
					has_dynamic_offset: false,
					min_binding_size: None,
					ty: wgpu::BufferBindingType::Storage { read_only: true }
				},
				visibility: wgpu::ShaderStages::COMPUTE
			}
		}).collect();

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &bgl_entries
		});

		fn create_empty_buffer<T>(device: &wgpu::Device) -> wgpu::Buffer {
			device.create_buffer(&wgpu::BufferDescriptor {
				label: None,
				mapped_at_creation: false,
				size: std::mem::size_of::<T>() as _,
				usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE
			})
		}

		let vertices = create_empty_buffer::<GpuVertex>(device);
		let triangles = create_empty_buffer::<u32>(device);
		let bvhs = create_empty_buffer::<GpuBvhNode>(device);
		let transforms = create_empty_buffer::<GpuTransform>(device);
		let materials = create_empty_buffer::<GpuMaterial>(device);

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: vertices.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: triangles.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: bvhs.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: transforms.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 4,
					resource: materials.as_entire_binding()
				},
			]
		});
		
		Self {
			bind_group,
			bind_group_layout,
			vertices,
			bvhs,
			materials,
			transforms,
			triangles
		}
	}

	pub fn update_data(
		&mut self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		f: impl FnOnce(
			&mut Option<Vec<GpuVertex>>,
			&mut Option<Vec<u32>>,
			&mut Option<Vec<GpuBvhNode>>,
			&mut Option<Vec<GpuTransform>>,
			&mut Option<Vec<GpuMaterial>>
		)
	) {
		let mut vertices = None;
		let mut triangles = None;
		let mut bvhs = None;
		let mut transforms = None;
		let mut materials = None;

		f(
			&mut vertices,
			&mut triangles,
			&mut bvhs,
			&mut transforms,
			&mut materials
		);

		fn fill(
			device: &wgpu::Device,
			queue: &wgpu::Queue,
			buffer: &mut wgpu::Buffer,
			data: &[u8]
		) -> bool
		{
			if buffer.size() < data.len() as u64 {
				*buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
					label: None,
					usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
					contents: data
				});
				
				true
			} else {
				queue.write_buffer(buffer, 0, data);
				false
			}
		}

		let mut update = false;
		if let Some(vertices) = vertices {
			update = update | fill(device, queue, &mut self.vertices, bytemuck::cast_slice(&vertices));
		}
		if let Some(triangles) = triangles {
			update = update | fill(device, queue, &mut self.triangles, bytemuck::cast_slice(&triangles));
		}
		if let Some(bvhs) = bvhs {
			update = update | fill(device, queue, &mut self.bvhs, bytemuck::cast_slice(&bvhs));
		}
		if let Some(transforms) = transforms {
			update = update | fill(device, queue, &mut self.transforms, bytemuck::cast_slice(&transforms));
		}
		if let Some(materials) = materials {
			update = update | fill(device, queue, &mut self.materials, bytemuck::cast_slice(&materials));
		}

		if update {
			self.recreate_bind_group(device);
		}
	}

	fn recreate_bind_group(&mut self, device: &wgpu::Device) {
		self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: self.vertices.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: self.triangles.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: self.bvhs.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: self.transforms.as_entire_binding()
				},
				wgpu::BindGroupEntry {
					binding: 4,
					resource: self.materials.as_entire_binding()
				},
			]
		});
	}
}

struct EnvironmentResources {
	env_map: TextureView,
	env_cdf: TextureView,
	env_sampler: wgpu::Sampler,

	bind_group: wgpu::BindGroup,
	bind_group_layout: wgpu::BindGroupLayout
}

impl EnvironmentResources {
	fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
		let env_map = Self::create_env_map(device, queue, 1, 1, Some(&[[0.0; 4]]));
		let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			min_filter: wgpu::FilterMode::Nearest,
			mag_filter: wgpu::FilterMode::Linear,
			..Default::default()
		});
		let env_cdf = Self::create_env_cdf_map(device, queue, 1, 1, Some(&[0.0]));

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
						access: wgpu::StorageTextureAccess::ReadOnly,
						format: wgpu::TextureFormat::R32Float,
						view_dimension: wgpu::TextureViewDimension::D2
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
			]
		});

		let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&env_sampler)
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&env_map)
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::TextureView(&env_cdf)
				}
			]
		});

		Self {
			bind_group,
			bind_group_layout,
			env_cdf,
			env_map,
			env_sampler
		}
	}

	fn update_env(
		&mut self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		width: u32,
		height: u32,
		data: &[[f32; 4]]
	) {
		self.env_map = Self::create_env_map(device, queue, width, height, Some(data));
		self.env_cdf = Self::create_env_cdf_map(device, queue, width/4, height/4, None);

		let cdf = EnvCdfPipeline::new(device);
		cdf.resolve(
			device,
			queue,
			&self.env_map,
			&self.env_sampler,
			&self.env_cdf
		);

		self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&self.env_sampler)
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&self.env_map)
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: wgpu::BindingResource::TextureView(&self.env_cdf)
				}
			]
		});
	}
	
	fn create_env_map(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		width: u32,
		height: u32,
		data: Option<&[[f32; 4]]>
	) -> TextureView
	{
		let desc = wgpu::TextureDescriptor {
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba32Float,
			label: None,
			mip_level_count: 1,
			sample_count: 1,
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1
			},
			usage: wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[]
		};
		let raw = if let Some(data) = data {
			device.create_texture_with_data(
				queue, 
				&desc,
				bytemuck::cast_slice(data)
			)
		} else {
			device.create_texture(&desc)
		};
		
		TextureView::new(&Texture::from_raw(raw, &desc), &wgpu::TextureViewDescriptor::default())
	}

	fn create_env_cdf_map(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		width: u32,
		height: u32,
		data: Option<&[f32]>
	) -> TextureView
	{
		let desc = wgpu::TextureDescriptor {
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::R32Float,
			label: None,
			mip_level_count: 1,
			sample_count: 1,
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1
			},
			usage: wgpu::TextureUsages::STORAGE_BINDING,
			view_formats: &[]
		};
		let raw = if let Some(data) = data {
			device.create_texture_with_data(
				queue, 
				&desc,
				bytemuck::cast_slice(data)
			)
		} else {
			device.create_texture(&desc)
		};

		TextureView::new(&Texture::from_raw(raw, &desc), &wgpu::TextureViewDescriptor::default())
	}
}

pub struct PathTracer {
	pipeline: wgpu::ComputePipeline,
	persistent_bind_group: wgpu::BindGroup,
	camera_ub: wgpu::Buffer,
	frame_idx_buffer: wgpu::Buffer,
	output_bgl: wgpu::BindGroupLayout,
	geometry: GeometryResources,
	environment: EnvironmentResources,	
}

impl PathTracer {
	pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
		let shader = {
			let builder = wgsl_preprocessor::ShaderBuilder::new(
				path::shaders().join("pathtrace.wgsl").to_str().unwrap()
			).unwrap();

			device.create_shader_module(builder.build())
		};

		let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
					ty: wgpu::BindingType::Buffer {
						has_dynamic_offset: false,
						min_binding_size: None,
						ty: wgpu::BufferBindingType::Uniform
					},
					visibility: wgpu::ShaderStages::COMPUTE
				},
			]
		});

		let camera_ub = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			mapped_at_creation: false,
			size: std::mem::size_of::<GpuCamera>() as _,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM
		});

		let frame_idx_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			mapped_at_creation: false,
			size: std::mem::size_of::<GpuCamera>() as _,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM
		});

		let persistent_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &bgl,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
						buffer: &camera_ub,
						offset: 0,
						size: None
					})
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
						buffer: &frame_idx_buffer,
						offset: 0,
						size: None
					})
				}
			]
		});

		let output_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: None,
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				count: None,
				ty: wgpu::BindingType::StorageTexture {
					access: wgpu::StorageTextureAccess::WriteOnly,
					format: wgpu::TextureFormat::Rgba32Float,
					view_dimension: wgpu::TextureViewDimension::D2
				},
				visibility: wgpu::ShaderStages::COMPUTE
			}]
		});

		let geometry = GeometryResources::new(device);
		let environment = EnvironmentResources::new(device, queue);

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&bgl, &geometry.bind_group_layout, &environment.bind_group_layout, &output_bgl],
			push_constant_ranges: &[]
		});

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			entry_point: "main",
			label: Some("compute pipeline"),
			layout: Some(&pipeline_layout),
			module: &shader
		});

		Self {
			pipeline,
			persistent_bind_group,
			camera_ub,
			output_bgl,
			geometry,
			frame_idx_buffer,
			environment
		}
	}

	pub fn update_camera(&self, queue: &wgpu::Queue, f: impl FnOnce(&mut GpuCamera)) {
		let mut kcam = GpuCamera::zeroed();
		f(&mut kcam);
		queue.write_buffer(&self.camera_ub, 0, bytemuck::bytes_of(&kcam));
	}

	pub fn update_geometry(
		&mut self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		f: impl FnOnce(
			&mut Option<Vec<GpuVertex>>,
			&mut Option<Vec<u32>>,
			&mut Option<Vec<GpuBvhNode>>,
			&mut Option<Vec<GpuTransform>>,
			&mut Option<Vec<GpuMaterial>>
		)
	) {
		self.geometry.update_data(device, queue, f);
	}

	pub fn set_environment(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, data: &[[f32; 4]]) {
		self.environment.update_env(device, queue, width, height, data);
	}

	pub fn sample(
		&self,
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		output: &TextureView,
		frame_idx: u32
	) -> wgpu::CommandBuffer {
		queue.write_buffer(&self.frame_idx_buffer, 0, bytemuck::bytes_of(&frame_idx));
		
		let output_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &self.output_bgl,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(output)
				}
			]
		});

		let output_size = (output.texture_desc.size.width, output.texture_desc.size.height);

		let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
		pass.set_pipeline(&self.pipeline);
		pass.set_bind_group(0, &self.persistent_bind_group, &[]);
		pass.set_bind_group(1, &self.geometry.bind_group, &[]);
		pass.set_bind_group(2, &self.environment.bind_group, &[]);
		pass.set_bind_group(3, &output_bg, &[]);
		pass.dispatch_workgroups(
			(output_size.0 as f32 / 16.0).ceil() as u32,
			(output_size.1 as f32 / 16.0).ceil() as u32,
			1
		);
		drop(pass);

		enc.finish()
	}
}