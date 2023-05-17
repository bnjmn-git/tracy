mod path;
mod bvh;
mod mesh;
mod math;
mod thread_pool;
mod path_tracing;
mod gpu_resource;
mod scene;
mod pipeline;
mod wgsl_preprocessor;

use std::{sync::{Arc, Mutex}, path::Path};

use bvh::Bvh;
use path_tracing::PathTracer;
use pipeline::{accum::AccumPipeline, tonemap::TonemapPipeline, bvh_visual::BvhVisual};
use bytemuck::Zeroable;
use cgmath::{Vector3, Zero, vec3, Deg, Matrix4, SquareMatrix, Quaternion, Rotation3, InnerSpace, Euler, VectorSpace, Rad};
use egui_wgpu::wgpu;
use eframe::egui;
use gpu_resource::{Texture, TextureView};
use gpu_resource::{GpuVertex, GpuBvhNode, GpuTransform, GpuMaterial};
use rfd::FileDialog;
use scene::{Camera, Scene, ObjectId, MaterialId};
use thread_pool::ThreadPool;

use crate::{mesh::Mesh, math::Transform, scene::{MeshDescriptor, ObjectDescriptor, MaterialDescriptor}};

const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

enum SceneMessage {
	CameraInput(Vec<egui::Event>)
}

enum Message {
	Scene(SceneMessage),
	MeshLoaded(Mesh),
	EnvImageLoaded(String, image::DynamicImage),
}

struct RenderTextures {
	sample_texture_view: TextureView,
	accum_texture_view: TextureView,
	output_texture_view: TextureView,
}

struct CameraController {
	speed: f32,
	pos_smooth_time: f32,
	rot_smooth_time: f32,

	target_pos: Vector3<f32>,
	target_rot: Vector3<f32>,
	move_axis: Vector3<f32>,
}

impl CameraController {
	pub fn new(speed: f32, pos_smooth_time: f32, rot_smooth_time: f32) -> Self {
		let camera_controller = Self {
			speed,
			pos_smooth_time,
			rot_smooth_time,
			target_pos: Vector3::zero(),
			target_rot: Vector3::zero(),
			move_axis: Vector3::zero()
		};
		camera_controller
	}

	fn euler_rot(&self) -> Euler<Deg<f32>> {
		Euler::new(
			Deg(self.target_rot.x),
			Deg(self.target_rot.y),
			Deg(self.target_rot.z)
		)
	}

	fn quat_rot(&self) -> Quaternion<f32> {
		let euler = self.euler_rot();
		Quaternion::from_angle_y(euler.y) *
		Quaternion::from(Euler::new(euler.x, Deg(0.0), euler.z))
	}

	pub fn update(&mut self, camera: &mut Camera, dt: f32) -> bool {
		let pos_lerp_pct = 1.0 - (0.01f32).powf(dt / self.pos_smooth_time);
		let rot_lerp_pct = 1.0 - (0.01f32).powf(dt / self.rot_smooth_time);

		if self.move_axis.magnitude2() > 1e-3 {
			let move_delta = self.move_axis.normalize() * self.speed * dt;
			self.target_pos += self.quat_rot() * move_delta;
		}

		let mut updated = false;

		let pos = camera.transform.pos;
		let tpos = self.target_pos;
		if f32::abs(pos.x - tpos.x) > 1e-2
		|| f32::abs(pos.y - tpos.y) > 1e-2
		|| f32::abs(pos.z - tpos.z) > 1e-2 {
			camera.transform.pos = pos.lerp(tpos, pos_lerp_pct);
			updated |= true;
		}

		let quat = self.quat_rot();
		let rot = camera.transform.rot;
		if f32::abs(quat.v.x - rot.v.x) > 1e-2
		|| f32::abs(quat.v.y - rot.v.y) > 1e-2
		|| f32::abs(quat.v.z - rot.v.z) > 1e-2
		|| f32::abs(quat.s - rot.s) > 1e-2 {
			camera.transform.rot = rot.slerp(quat, rot_lerp_pct);
			updated |= true;
		}

		updated
	}

	pub fn reset(&mut self) {
		self.move_axis = Vector3::zero();
	}
}

#[derive(Clone)]
struct AppMessageSender {
	messages: Arc<Mutex<Vec<Message>>>,
	ctx: egui::Context
}

impl AppMessageSender {
	pub fn send_message(&self, msg: Message) {
		self.messages.lock().unwrap().push(msg);
		self.ctx.request_repaint();
	}
}

struct App {
	device: Arc<wgpu::Device>,
	queue: Arc<wgpu::Queue>,

	thread_pool: ThreadPool,
	messages: Arc<Mutex<Vec<Message>>>,

	scene: Scene,
	path_tracer: PathTracer,
	accum_pipeline: AccumPipeline,
	tonemape_pipeline: TonemapPipeline,
	visualize_bvh: bool,
	bvh_visualizer: BvhVisual,

	output_texture_id: Option<egui::TextureId>,
	render_textures: Option<RenderTextures>,
	
	iterations: u32,
	max_iterations: u32,
	camera_controller: CameraController,
	prev_pointer_pos: egui::Pos2,
	delta_time: f32,
	aspect: f32,	// TODO: put this into camera instead

	selected_env_map: Option<usize>,
	environment_maps: Vec<(String, image::DynamicImage)>,

	selected_object: Option<usize>,
	objects: Vec<(String, ObjectId)>,

	ground_material: MaterialId,
	material: MaterialId,

	force_update: bool,
	hide_ui: bool,
}

impl App {
	fn new(cc: &eframe::CreationContext<'_>) -> Self {
		let wgpu = cc.wgpu_render_state.as_ref().unwrap();
		let device = &wgpu.device;
		let queue = &wgpu.queue;

		log::info!("{:#?}", device.limits());

		let thread_pool = ThreadPool::new();

		let mut scene = Scene::new();
		let path_tracer = PathTracer::new(device, queue);
		let accum_pipeline = AccumPipeline::new(device, wgpu::TextureFormat::Rgba32Float);

		let mesh = scene.add_mesh(MeshDescriptor {
			mesh: Mesh::from_obj(path::mesh().join("cube.obj")).unwrap().into_iter().nth(0).unwrap()
		});

		let ground_material = scene.add_material(MaterialDescriptor::default());
		scene.add_object(ObjectDescriptor {
			enabled: true,
			material: Some(ground_material),
			mesh: Some(mesh),
			transform: Transform {
				pos: vec3(0.0, -2.0, 10.0),
				scale: vec3(10.0, 1.0, 10.0),
				..Default::default()
			}
		});

		let material = scene.add_material(MaterialDescriptor::default());

		let messages = Arc::new(Mutex::new(Vec::new()));
		let msg_sender = AppMessageSender {
			ctx: cc.egui_ctx.clone(),
			messages: messages.clone()
		};
		
		for i in 0..4 {
			let msg_sender = msg_sender.clone();
			let filename = path::textures().join(format!("hdr{}.exr", i));
			
			thread_pool.add_job(move || {
				load_environment_map(msg_sender, filename);
			});
		}
		
		for name in ["cube.obj", "suzanne.obj", "sphere.obj"] {
			let msg_sender = msg_sender.clone();
			let filename = path::mesh().join(name);
			thread_pool.add_job(move || {
				load_mesh(msg_sender, filename);
			});
		}

		Self {
			device: device.clone(),
			queue: queue.clone(),
			thread_pool,
			messages,
			scene,
			path_tracer,
			accum_pipeline,
			tonemape_pipeline: TonemapPipeline::new(device, OUTPUT_TEXTURE_FORMAT),
			bvh_visualizer: BvhVisual::new(device, OUTPUT_TEXTURE_FORMAT),
			visualize_bvh: false,
			output_texture_id: None,
			render_textures: None,
			iterations: 0,
			max_iterations: 32,
			camera_controller: CameraController::new(5.0, 0.2, 0.4),
			prev_pointer_pos: egui::Pos2::default(),
			delta_time: 0.0,
			aspect: 1.0,
			environment_maps: Vec::new(),
			selected_env_map: None,
			objects: Vec::new(),
			selected_object: None,
			force_update: false,
			material,
			ground_material,
			hide_ui: false
		}
	}

	fn drain_messages(&mut self) {
		let messages: Vec<_> = self.messages.lock().unwrap().drain(..).collect();

		for message in messages {
			match message {
				Message::Scene(msg) => match msg {
					SceneMessage::CameraInput(events) => {
						self.process_camera_inputs(events);
					}
				}
				Message::MeshLoaded(mesh) => {
					let name = mesh.name.clone();
					let mesh = self.scene.add_mesh(MeshDescriptor { mesh });
					let object = self.scene.add_object(ObjectDescriptor {
						enabled: self.selected_object.is_none(),
						transform: Transform {
							pos: vec3(0.0, 0.0, 10.0),
							..Default::default()
						},
						material: Some(self.material),
						mesh: Some(mesh)
					});

					if self.selected_object.is_none() {
						self.selected_object = Some(self.objects.len());
						self.force_update = true;
					}
					self.objects.push((name, object));
				}
				Message::EnvImageLoaded(name, img) => {
					if self.selected_env_map.is_none() {
						self.path_tracer.set_environment(
							&self.device,
							&self.queue,
							img.width(),
							img.height(),
							bytemuck::cast_slice(&img.to_rgba32f())
						);

						self.selected_env_map = Some(self.environment_maps.len());
						
						self.force_update = true;
					}
					self.environment_maps.push((name, img));
				}
			}
		}
	}
}

fn load_environment_map(msg_sender: AppMessageSender, filename: impl AsRef<Path>) {
	let filename = filename.as_ref();
	let img = match image::io::Reader::open(filename) {
		Ok(ok) => ok.decode().unwrap(),
		Err(e) => {
			log::error!("{}", e);
			return;
		}
	};

	msg_sender.send_message(Message::EnvImageLoaded(filename.file_name().unwrap().to_str().unwrap().to_owned(), img));
}

fn load_mesh(msg_sender: AppMessageSender, filename: impl AsRef<Path>) {
	let filename = filename.as_ref();
	let meshes = match Mesh::from_obj(filename) {
		Ok(meshes) => meshes,
		Err(e) => {
			log::error!("{:?}", e);
			return;
		}
	};

	for mesh in meshes {
		msg_sender.send_message(Message::MeshLoaded(mesh));
	}
}

fn process_scene_bvh_node(
	node: &bvh::Node,
	ids: &[usize],
	object_data: &scene::ObjectData,
	mesh_bvh_root_indexes: &[usize],
	nodes: &mut Vec<GpuBvhNode>
) -> i32
{
	let node_idx = nodes.len();
	nodes.push(GpuBvhNode::zeroed());

	match node {
		bvh::Node::Interior { bounds, children, .. } => {
			nodes[node_idx].min = bounds.min.into();
			nodes[node_idx].max = bounds.max.into();
			nodes[node_idx].param0 = process_scene_bvh_node(&children[0], ids, object_data, mesh_bvh_root_indexes, nodes);
			nodes[node_idx].param1 = process_scene_bvh_node(&children[1], ids, object_data, mesh_bvh_root_indexes, nodes);
			nodes[node_idx].param2 = 0;
		}
		bvh::Node::Leaf { bounds, prim_range } => {
			nodes[node_idx].min = bounds.min.into();
			nodes[node_idx].max = bounds.max.into();
			
			let scene_object_idx = ids[prim_range.start];
			let mesh_idx = object_data.mesh_indexes[scene_object_idx];
			let mesh_bvh_root_idx = mesh_bvh_root_indexes[mesh_idx];
			let material_idx = object_data.material_indexes[scene_object_idx];

			nodes[node_idx].param0 = mesh_bvh_root_idx as i32;
			nodes[node_idx].param1 = material_idx as i32;
			nodes[node_idx].param2 = -(scene_object_idx as i32) - 1;
		}
	}

	node_idx as i32
}

fn process_scene_bvh(
	scene_bvh: &Bvh,
	mesh_bvh_root_indexes: &[usize],
	object_data: &scene::ObjectData,
	nodes: &mut Vec<GpuBvhNode>
) {
	process_scene_bvh_node(
		&scene_bvh.root,
		&scene_bvh.ids,
		object_data,
		mesh_bvh_root_indexes,
		nodes
	);
}

fn process_mesh_bvh_node(node: &bvh::Node, prim_offset: usize, nodes: &mut Vec<GpuBvhNode>) -> i32 {
	let node_idx = nodes.len();
	nodes.push(GpuBvhNode::zeroed());

	match node {
		bvh::Node::Interior { bounds, children, .. } => {
			nodes[node_idx].min = bounds.min.into();
			nodes[node_idx].max = bounds.max.into();
			nodes[node_idx].param0 = process_mesh_bvh_node(&children[0], prim_offset, nodes);
			nodes[node_idx].param1 = process_mesh_bvh_node(&children[1], prim_offset, nodes);
			nodes[node_idx].param2 = 0;
		}
		bvh::Node::Leaf { bounds, prim_range } => {
			nodes[node_idx].min = bounds.min.into();
			nodes[node_idx].max = bounds.max.into();
			nodes[node_idx].param0 = (prim_offset + prim_range.start * 3) as i32;
			nodes[node_idx].param1 = prim_range.len() as i32;
			nodes[node_idx].param2 = 1;
		}
	}

	node_idx as i32
}

fn process_mesh_bvhs(mesh_bvhs: &[Bvh], object_data: &scene::ObjectData, nodes: &mut Vec<GpuBvhNode>) {
	for (i, bvh) in mesh_bvhs.iter().enumerate() {
		process_mesh_bvh_node(&bvh.root, object_data.mesh_bvh_index_offsets[i], nodes);
	}
}

fn flatten_bvh_for_gpu(
	scene_bvh: &Bvh,
	mesh_bvhs: &Vec<Bvh>,
	object_data: &scene::ObjectData
) -> Vec<GpuBvhNode> {
	let total_nodes = mesh_bvhs.iter().fold(scene_bvh.num_nodes, |v, bvh| {
		v + bvh.num_nodes
	});

	let mut nodes = Vec::with_capacity(total_nodes);
	let mut mesh_bvh_root_indexes = Vec::with_capacity(mesh_bvhs.len());
	let mut root_idx_offset = scene_bvh.num_nodes;
	for bvh in mesh_bvhs.iter() {
		mesh_bvh_root_indexes.push(root_idx_offset);
		root_idx_offset += bvh.num_nodes;
	}

	process_scene_bvh(scene_bvh, &mesh_bvh_root_indexes, object_data, &mut nodes);
	process_mesh_bvhs(mesh_bvhs, object_data, &mut nodes);

	nodes
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
		ctx.input(|input| {
			self.delta_time = input.stable_dt;
			for e in input.events.iter() {
				match e {
					egui::Event::Key { key, pressed, repeat, .. } => {
						if *key == egui::Key::F1 && *pressed && !*repeat {
							self.hide_ui = !self.hide_ui;
						}
					}
					_ => {}
				}
			}
		});
		self.drain_messages();

		if !self.hide_ui {
			egui::SidePanel::left("settings").resizable(true).show(ctx, |ui| {
				egui::ScrollArea::both().show(ui, |ui| {
					ui.label(format!("Sample Count: {}/{}", self.iterations, self.max_iterations));
					ui.horizontal(|ui| {
						let label = ui.label("Max Samples");
						ui.add(egui::DragValue::new(&mut self.max_iterations).clamp_range(1..=u32::MAX)).labelled_by(label.id);
					});
	
					ui.group(|ui| {
						ui.heading("Camera");
	
						ui.label(format!("Camera Speed: {}", self.camera_controller.speed));
	
						let mut camera = self.scene.camera();
						if show_ui_for_camera(ui, &mut camera) {
							self.scene.update_camera(|c| *c = camera);
						}
					});
					
	
					ui.toggle_value(&mut self.visualize_bvh, "Visualize BVH");
	
					ui.group(|ui| {
						ui.heading("Environment Maps");
	
						if ui.button("Add").clicked() {
							let files = FileDialog::new()
								.add_filter("format", &["hdr", "exr"])
								.set_directory(path::data())
								.pick_files()
								.unwrap();
	
							self.thread_pool.add_job({
								let msg_sender = AppMessageSender {
									ctx: ctx.clone(),
									messages: self.messages.clone()
								};
	
								move || {
									for file in files {
										load_environment_map(msg_sender.clone(), file);
									}
								}
							});
						}
	
						let prev_selected = self.selected_env_map.unwrap_or(0);
						for (i, (name, _)) in self.environment_maps.iter().enumerate() {
							ui.radio_value(&mut self.selected_env_map, Some(i), name);
						}
		
						if let Some(selected) = self.selected_env_map {
							let (_, img) = &self.environment_maps[selected];
							if selected != prev_selected {
								self.path_tracer.set_environment(
									&self.device,
									&self.queue,
									img.width(),
									img.height(),
									bytemuck::cast_slice(&img.to_rgba32f())
								);
								self.force_update = true;
							}
						}
					});
	
					ui.group(|ui| {
						ui.heading("Objects");
	
						if ui.button("Add").clicked() {
							let files = FileDialog::new()
								.add_filter("format", &["obj"])
								.pick_files()
								.unwrap();
	
							self.thread_pool.add_job({
								let msg_sender = AppMessageSender {
									ctx: ctx.clone(),
									messages: self.messages.clone()
								};
	
								move || {
									for file in files {
										load_mesh(msg_sender.clone(), file);
									}
								}
							});
						}

						if let Some(selected) = self.selected_object {
							let object_id = self.objects[selected].1;
							let object = self.scene.object(object_id);

							let mut trans = object.transform;
							let mut pos = trans.pos;
							let rot: Euler<Rad<f32>> = trans.rot.into();
							let mut rot: Euler<Deg<f32>> = Euler::new(rot.x.into(), rot.y.into(), rot.z.into());
							trans.rot = rot.into();
							let mut scale = trans.scale;

							let mut updated = false;

							egui::Grid::new("pos")
							.num_columns(4)
							.show(ui, |ui| {
								ui.label("Pos");
								updated |= ui.add(egui::DragValue::new(&mut pos.x)).changed();
								updated |= ui.add(egui::DragValue::new(&mut pos.y)).changed();
								updated |= ui.add(egui::DragValue::new(&mut pos.z)).changed();
								ui.end_row();
								
								ui.label("Rot");
								updated |= ui.add(egui::DragValue::new(&mut rot.x.0)).changed();
								updated |= ui.add(egui::DragValue::new(&mut rot.y.0)).changed();
								updated |= ui.add(egui::DragValue::new(&mut rot.z.0)).changed();
								ui.end_row();

								ui.label("Scale");
								updated |= ui.add(egui::DragValue::new(&mut scale.x)).changed();
								updated |= ui.add(egui::DragValue::new(&mut scale.y)).changed();
								updated |= ui.add(egui::DragValue::new(&mut scale.z)).changed();
								ui.end_row();
							});

							if updated {
								self.scene.update_object(object_id, |desc| {
									desc.transform = Transform {
										pos,
										rot: rot.into(),
										scale
									};
								});
							}
						}
	
						let prev_selected = self.selected_object.unwrap_or(0);
						for (i, (name, _)) in self.objects.iter().enumerate() {
							ui.radio_value(&mut self.selected_object, Some(i), name);
						}
	
						if let Some(selected) = self.selected_object {
							if selected != prev_selected {
								let object = &self.objects[selected];
								self.scene.update_object(object.1, |o| o.enabled = true);
	
								let prev_object = &self.objects[prev_selected];
								self.scene.update_object(prev_object.1, |o| o.enabled = false);
							}
						}
					});
	
					ui.group(|ui| {
						ui.heading("Ground Material");
	
						let mut mat = self.scene.material(self.ground_material);
						if show_ui_for_material("ground material", ui, &mut mat) {
							self.scene.update_material(self.ground_material, |m| {
								m.albedo = mat.albedo.into();
								m.metallic = mat.metallic;
								m.ior = mat.ior;
								m.emission = mat.emission.into();
								m.spec_tint = mat.spec_tint.into();
								m.spec_trans = mat.spec_trans;
								m.roughness = mat.roughness;
							});
						}
	
						ui.heading("Material");
	
						let mut mat = self.scene.material(self.material);
						if show_ui_for_material("material", ui, &mut mat) {
							self.scene.update_material(self.material, |m| {
								m.albedo = mat.albedo.into();
								m.metallic = mat.metallic;
								m.ior = mat.ior;
								m.emission = mat.emission.into();
								m.spec_tint = mat.spec_tint.into();
								m.spec_trans = mat.spec_trans;
								m.roughness = mat.roughness;
							});
						}
					});
				})
			});
		}

		egui::CentralPanel::default().show(ctx, |ui| {
			self.run_scene_view(ctx, frame, ui);
		});
    }
}

impl App {
	fn process_camera_inputs(&mut self, events: Vec<egui::Event>) {
		let cc = &mut self.camera_controller;
		
		for e in events {
			match e {
				egui::Event::Key { key, pressed, repeat, .. } => {
					if repeat {
						continue;
					}

					let delta = if pressed { 1.0 } else { -1.0 };

					match key {
						egui::Key::A => cc.move_axis.x -= delta,
						egui::Key::D => cc.move_axis.x += delta,
						egui::Key::S => cc.move_axis.z -= delta,
						egui::Key::W => cc.move_axis.z += delta,
						egui::Key::Q => cc.move_axis.y -= delta,
						egui::Key::E => cc.move_axis.y += delta,
						_ => {}
					}
				}
				egui::Event::Scroll(delta) => {
					cc.speed = f32::max(0.1, cc.speed + delta.y / 50.0);
				}
				egui::Event::PointerButton { pos, .. } => {
					self.prev_pointer_pos = pos;
				}
				egui::Event::PointerMoved(pos) => {
					let delta = (pos - self.prev_pointer_pos) / 2.0;
					self.prev_pointer_pos = pos;
					cc.target_rot.x += delta.y;
					cc.target_rot.y += delta.x;
				}
				_ => {}
			}
		}
	}

	fn run_scene_view(
		&mut self, 
		ctx: &egui::Context,
		frame: &eframe::Frame,
		ui: &mut egui::Ui
	) {
		let render_state = frame.wgpu_render_state().unwrap();

		let mut camera = self.scene.camera();
		if self.camera_controller.update(&mut camera, self.delta_time) {
			self.scene.update_camera(|cam| *cam = camera);
		}

		let mut scene_updated = {
			let mut vertices = None;
			let mut triangles = None;
			let mut bvhs = None;
			let mut transforms = None;
			let mut materials = None;

			let scene_updated = self.scene.apply_updates(|update| {
				match update {
					scene::Update::Camera(camera) => {
						self.bvh_visualizer.set_camera(&self.queue, self.aspect, &camera);
						self.path_tracer.update_camera(&self.queue, |gpu| {
							gpu.aperture = camera.aperture;
							gpu.focal_dist = camera.focal_distance;
							gpu.fov = camera.fov.0;
							gpu.position = camera.transform.pos.into();
							gpu.forward = camera.transform.forward().into();
							gpu.right = camera.transform.right().into();
							gpu.up = camera.transform.up().into();
							gpu.bokeh_blades = camera.bokeh_blades;
						});
					}
					scene::Update::Mesh(verts, indices) => {
						let v = verts.into_iter().map(|v| {
							GpuVertex {
								normal: v.normal.into(),
								pos: v.pos.into(),
								uvx: v.uv.x,
								uvy: v.uv.y
							}
						});
						vertices = Some(v.collect());
						triangles = Some(indices);
					}
					scene::Update::Material(mats) => {
						let gpu = mats.into_iter().map(|m| {
							let mut g = GpuMaterial::zeroed();
							g.albedo = m.albedo.into();
							g.emission = m.emission.into();
							g.opacity = m.opacity;
							g.ior = m.ior;
							g.metallic = m.metallic;
							g.roughness = m.roughness;
							g.spec_tint = m.spec_tint.into();
							g.spec_trans = m.spec_trans.into();
							g
						});

						materials = Some(gpu.collect());
					}
					scene::Update::Bvh { scene_bvh, mesh_bvhs, object_data } => {
						self.bvh_visualizer.set_data_from_scene(&self.device, &scene_bvh, &mesh_bvhs, &object_data);

						let gpu_bvhs = flatten_bvh_for_gpu(&scene_bvh, &mesh_bvhs, &object_data);
						let gpu_transforms = object_data.transforms.into_iter().map(|t| {
							let m: Matrix4<_> = t.into();
							GpuTransform {
								world: m.into(),
								inv_world: m.invert().unwrap_or(Matrix4::identity()).into()
							}
						});

						bvhs = Some(gpu_bvhs);
						transforms = Some(gpu_transforms.collect());
					}
				}
			});

			self.path_tracer.update_geometry(
				&self.device,
				&self.queue,
				|v, t, b, trans, mats| {
					*v = vertices;
					*t = triangles;
					*b = bvhs;
					*trans = transforms;
					*mats = materials;
				}
			);

			scene_updated
		};

		scene_updated |= self.force_update;
		self.force_update = false;
		
		let size = wgpu::Extent3d {
			width: ui.available_width() as u32,
			height: ui.available_height() as u32,
			depth_or_array_layers: 1
		};
		let output_size = self.render_textures.as_ref().map_or(
			wgpu::Extent3d { width: u32::MAX, height: u32::MAX, depth_or_array_layers: 1 },
			|rts| {
				rts.output_texture_view.texture_desc.size
			}
		);

		if output_size != size {
			let create_texture = || {
				Texture::new(&self.device, &wgpu::TextureDescriptor {
					label: None,
					dimension: wgpu::TextureDimension::D2,
					format: OUTPUT_TEXTURE_FORMAT,
					mip_level_count: 1,
					sample_count: 1,
					size,
					usage: wgpu::TextureUsages::STORAGE_BINDING
						| wgpu::TextureUsages::TEXTURE_BINDING
						| wgpu::TextureUsages::RENDER_ATTACHMENT,
					view_formats: &[]
				})
			};

			let output_texture_view = TextureView::new(&create_texture(), &wgpu::TextureViewDescriptor::default());

			if let Some(id) = self.output_texture_id {
				render_state.renderer.write().update_egui_texture_from_wgpu_texture(
					&self.device,
					&output_texture_view,
					wgpu::FilterMode::Linear,
					id
				);
			} else {
				self.output_texture_id = Some(render_state.renderer.write().register_native_texture(
					&self.device,
					&output_texture_view,
					wgpu::FilterMode::Linear
				));
			}

			let sample_texture_view = TextureView::new(&create_texture(), &wgpu::TextureViewDescriptor::default());
			let accum_texture_view = TextureView::new(&create_texture(), &wgpu::TextureViewDescriptor::default());

			self.render_textures = Some(RenderTextures {
				output_texture_view,
				sample_texture_view,
				accum_texture_view
			});

			scene_updated = true;

			self.aspect = size.width as f32 / size.height as f32;

		}

		if scene_updated || self.iterations < self.max_iterations {
	
			if scene_updated {
				self.iterations = 1;
			} else {
				self.iterations += 1;
			}
	
			if let Some(rts) = self.render_textures.as_ref() {
				let mut cmd_bufs = Vec::with_capacity(4);

				cmd_bufs.push(self.path_tracer.sample(&self.device, &self.queue, &rts.sample_texture_view, self.iterations));
				cmd_bufs.push(self.accum_pipeline.resolve(&self.device, &self.queue, &rts.sample_texture_view, &rts.accum_texture_view, self.iterations, scene_updated));
				cmd_bufs.push(self.tonemape_pipeline.resolve(&self.device, &rts.accum_texture_view , &rts.output_texture_view));

				if self.visualize_bvh {
					cmd_bufs.push(self.bvh_visualizer.render(&self.device, &rts.output_texture_view));
				}
				
				self.queue.submit(cmd_bufs.into_iter());

				ctx.request_repaint();
			}
		}
		
		let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
		if response.dragged_by(egui::PointerButton::Primary) {
			response.request_focus();
		}
		if response.drag_released_by(egui::PointerButton::Primary) {
			response.surrender_focus();

			self.camera_controller.reset();
		}

		if response.has_focus() {
			ctx.input(|input| {
				let mut events = Vec::new();
				for event in &input.events {
					match event {
						egui::Event::Key { .. }
						| egui::Event::PointerMoved(..)
						| egui::Event::PointerButton { .. }
						| egui::Event::Scroll(..) => events.push(event.clone()),
						_ => {}
					}
				}

				self.messages.lock().unwrap().push(Message::Scene(SceneMessage::CameraInput(events)));
			});
		}

		if let Some(id) = self.output_texture_id {
			ui.allocate_ui_at_rect(rect, |ui| {
				ui.image(id, ui.available_size())
			});
		}
	}
}

fn show_ui_for_camera(ui: &mut egui::Ui, camera: &mut Camera) -> bool {
	let mut updated = false;

	egui::Grid::new("camera")
	.num_columns(2)
	.striped(true)
	.show(ui, |ui| {
		let label = ui.label("Fov");
		let mut deg: Deg<f32> = camera.fov.into();
		updated |= ui
			.add(egui::Slider::new(&mut deg.0, 1.0..=179.0))
			.labelled_by(label.id)
			.changed();
		camera.fov = deg.into();

		ui.end_row();
		
		let label = ui.label("Aperture");
		let mut aperture = camera.aperture * 1000.0;
		updated |= ui
			.add(egui::Slider::new(&mut aperture, 0.0..=1000.0).suffix("mm"))
			.labelled_by(label.id)
			.changed();
		camera.aperture = aperture / 1000.0;
		ui.end_row();
	
		let label = ui.label("Focal Distance");
		updated |= ui
		.add(egui::Slider::new(&mut camera.focal_distance, 0.0..=50.0))
		.labelled_by(label.id)
		.changed();
		ui.end_row();
	
		let label = ui.label("Bokeh Blades");
		updated |= ui
		.add(egui::Slider::new(&mut camera.bokeh_blades, 0..=8))
		.labelled_by(label.id)
		.changed();
		ui.end_row();
	});

	updated
}

fn show_ui_for_material(id: impl std::hash::Hash, ui: &mut egui::Ui, mat: &mut MaterialDescriptor) -> bool {
	let mut updated = false;

	egui::Grid::new(id)
	.num_columns(2)
	.striped(true)
	.show(ui, |ui| {
		let albedo: &mut [f32; 3] = mat.albedo.as_mut();
		ui.label("Albedo");		
		updated |= ui.color_edit_button_rgb(albedo).changed();
		ui.end_row();
	
		let emission: &mut [f32; 3] = mat.emission.as_mut();
		ui.label("Emission");
		updated |= ui.color_edit_button_rgb(emission).changed();
		ui.end_row();
		
		ui.label("Metallic");
		updated |= ui.add(egui::Slider::new(&mut mat.metallic, 0.0..=1.0)).changed();
		ui.end_row();
		
		ui.label("Roughness");
		updated |= ui.add(egui::Slider::new(&mut mat.roughness, 0.0..=1.0)).changed();
		ui.end_row();
		
		let specular_tint: &mut [f32; 3] = mat.spec_tint.as_mut();
		ui.label("Specular Tint");
		updated |= ui.color_edit_button_rgb(specular_tint).changed();
		ui.end_row();

		ui.label("Transmission");
		updated |= ui.add(egui::Slider::new(&mut mat.spec_trans, 0.0..=1.0)).changed();
		ui.end_row();
		
		ui.label("IOR");
		updated |= ui.add(egui::Slider::new(&mut mat.ior, 0.5..=2.0)).changed();
		ui.end_row();
	});

	updated
}

fn main() -> Result<(), eframe::Error> {
	env_logger::Builder::new()
	.filter_level(log::LevelFilter::Trace)
	.filter(Some("wgpu_core"), log::LevelFilter::Info)
	.filter(Some("naga"), log::LevelFilter::Off)
	.filter(Some("wgpu_hal"), log::LevelFilter::Off)
	.init();

	let options = eframe::NativeOptions {
		centered: true,
		initial_window_size: Some(egui::Vec2::new(800.0, 600.0)),
		renderer: eframe::Renderer::Wgpu,
		wgpu_options: egui_wgpu::WgpuConfiguration {
			device_descriptor: wgpu::DeviceDescriptor {
				features:
					wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES |
					wgpu::Features::POLYGON_MODE_LINE,
				limits: wgpu::Limits {
					// max_bind_groups: 8,
					..Default::default()
				},
				label: None
			},
			power_preference: wgpu::PowerPreference::HighPerformance,
			backends: wgpu::Backends::all(),
			..Default::default()
		},
		..Default::default()
	};

	eframe::run_native(
		"Tracy",
		options,
		Box::new(|cc| {
			Box::new(App::new(cc))
		})
	)
}