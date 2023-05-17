#include "scene_data.wgsl"
#include "ray.wgsl"

fn invert_if_less(v: f32) -> f32 {
	let s = sign(v);
	let s2 = s*s;
	return mix(INF, s2 / (v + s2 - 1.0), s2);
}

fn vec3_invert_if_less(v: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		invert_if_less(v.x),
		invert_if_less(v.y),
		invert_if_less(v.z)
	);
}

fn ray_aabb_intersects(ro: vec3<f32>, inv_dir: vec3<f32>, bb_min: vec3<f32>, bb_max: vec3<f32>) -> bool {
	let n = (bb_min - ro) * inv_dir;
	let f = (bb_max - ro) * inv_dir;

	let tmin = min(n, f);
	let tmax = max(n, f);

	let t0 = max(tmin.x, max(tmin.y, tmin.z));
	let t1 = min(tmax.x, min(tmax.y, tmax.z));

	let tn = min(t0, t1);
	let tf = max(t0, t1);

	// Not proper checking because this algorithm isn't working properly.
	// Holes appear in the mesh as this algorithm falsely reports not intersection.
	return tf > 0.0 || tn < 0.0;
}

fn tri_intersection(
	ray: Ray, 
	p0: vec3<f32>, 
	p1: vec3<f32>, 
	p2: vec3<f32>, 
	puvt: ptr<function, vec4<f32>>, 
	pback_face: ptr<function, bool>
) -> bool {
	let e0 = p1 - p0;
	let e1 = p2 - p0;
	let pv = cross(ray.dir, e1);
	let det = dot(e0, pv);
	let tv = ray.origin - p0;
	let qv = cross(tv, e0);

	var uvt: vec4<f32>;
	uvt.x = dot(tv, pv);
	uvt.y = dot(ray.dir, qv);
	uvt.z = dot(e1, qv);
	uvt = vec4<f32>(uvt.xyz / det, uvt.w);
	uvt.w = 1.0 - uvt.x - uvt.y;

	if all(uvt >= vec4<f32>(0.)) {
		*pback_face = dot(-ray.dir, cross(e0, e1)) < 0.0;
		*puvt = uvt;
		return true;
	}

	return false;
}

struct SurfaceData {
	dist: f32,
	pos: vec3<f32>,
	normal: vec3<f32>,
	uv: vec2<f32>,
	back_face: bool,
	ffnormal: vec3<f32>,
	tangent: vec3<f32>,
	bitangent: vec3<f32>,
	material: Material,
	eta: f32,
}

fn closest_hit(ray: Ray, psurface: ptr<function, SurfaceData>) -> bool {
	if !ray_aabb_intersects(ray.origin, 1.0 / ray.dir, bvh[0].min, bvh[0].max) {
		return false;
	}

	var surface: SurfaceData;

	var t: f32 = INF;
	var intersected: bool = false;
	var bary: vec3<f32>;
	var verts: array<Vertex, 3>;
	var trans: mat4x4<f32>;
	var material_idx: i32;
	var is_back_face: bool;
	
	var bvh_idx: i32;
	var nodes_to_visit: array<i32, 64>;
	var visit_len = 0;
	var traversing_mesh = false;
	var ray_trans: Ray = ray;
	var ray_inv_dir = 1.0 / ray.dir;
	var cur_trans: mat4x4<f32>;
	var cur_material_idx: i32;
	
	nodes_to_visit[visit_len] = -1;
	visit_len++;

	nodes_to_visit[visit_len] = 0;
	visit_len++;

	while true {
		visit_len--;
		bvh_idx = nodes_to_visit[visit_len];
		if bvh_idx == -1 {
			if traversing_mesh {
				traversing_mesh = false;
				ray_trans = ray;
				ray_inv_dir = 1.0 / ray.dir;
				continue;
			} else {
				break;
			}
		}

		let node = bvh[bvh_idx];

		// Interior node of either scene/mesh
		if node.param2 == 0 {
			let left_node = bvh[node.param0];
			let right_node = bvh[node.param1];
			let left_hit = ray_aabb_intersects(ray_trans.origin, ray_inv_dir, left_node.min, left_node.max);
			let right_hit = ray_aabb_intersects(ray_trans.origin, ray_inv_dir, right_node.min, right_node.max);

			if left_hit {
				nodes_to_visit[visit_len] = node.param0;
				visit_len++;
			}

			if right_hit {
				nodes_to_visit[visit_len] = node.param1;
				visit_len++;
			}
		}
		// Leaf node of scene bvh
		else if node.param2 < 0 {
			let object_idx = -node.param2 - 1;
			let mesh_root_bvh_idx = node.param0;
			
			traversing_mesh = true;
			let transform = transforms[object_idx];
			ray_trans.origin = (transform.inv_world * vec4<f32>(ray_trans.origin, 1.0)).xyz;
			ray_trans.dir = normalize((transform.inv_world * vec4<f32>(ray_trans.dir, 0.0)).xyz);
			ray_inv_dir = 1.0 / ray_trans.dir;

			cur_trans = transform.world;
			cur_material_idx = node.param1;
			
			nodes_to_visit[visit_len] = -1;
			visit_len++;
			nodes_to_visit[visit_len] = mesh_root_bvh_idx;
			visit_len++;
		}
		// Leaf node of mesh bvh
		else if node.param2 > 0 {
			for (var i = 0; i < node.param1; i += 1) {
				let v0 = vertices[triangles[node.param0 + i*3+0]];
				let v1 = vertices[triangles[node.param0 + i*3+1]];
				let v2 = vertices[triangles[node.param0 + i*3+2]];

				let p0 = (cur_trans * vec4<f32>(v0.pos, 1.0)).xyz;
				let p1 = (cur_trans * vec4<f32>(v1.pos, 1.0)).xyz;
				let p2 = (cur_trans * vec4<f32>(v2.pos, 1.0)).xyz;

				var uvt: vec4<f32>;
				var back_face: bool;

				if tri_intersection(ray, p0, p1, p2, &uvt, &back_face) && uvt.z < t {
					t = uvt.z;
					bary = uvt.wxy;
					trans = cur_trans;
					is_back_face = back_face;
					material_idx = cur_material_idx;

					verts[0] = v0;
					verts[1] = v1;
					verts[2] = v2;

					intersected = true;
				}
			}
		}
	}

	if !intersected {
		return false;
	}

	var uv0 = vec2<f32>(verts[0].uvx, verts[0].uvy);
	var uv1 = vec2<f32>(verts[1].uvx, verts[1].uvy);
	var uv2 = vec2<f32>(verts[2].uvx, verts[2].uvy);

	surface.dist = t;
	surface.pos = ray.origin + ray.dir * t;
	surface.normal = bary.x * verts[0].normal + bary.y * verts[1].normal + bary.z * verts[2].normal;
	surface.uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

	let e1 = verts[1].pos - verts[0].pos;
	let e2 = verts[2].pos - verts[0].pos;
	let t1 = uv1 - uv0;
	let t2 = uv2 - uv0;
	let inv_det = 1.0 / (t1.x * t2.y - t1.y * t2.x);

	surface.tangent = (e1 * t2.y - e2 * t1.y) * inv_det;
	surface.bitangent = (e2 * t1.x - e1 * t2.x) * inv_det;

	surface.normal = normalize((trans * vec4<f32>(surface.normal, 0.0)).xyz);
	surface.tangent = normalize((trans * vec4<f32>(surface.tangent, 0.0)).xyz);
	surface.bitangent = normalize((trans * vec4<f32>(surface.bitangent, 0.0)).xyz);

	surface.back_face = is_back_face;
	surface.material = materials[material_idx];

	let r = surface.material.roughness;
	surface.material.roughness = max(EPS, r*r);

	if surface.back_face {
		surface.ffnormal = -surface.normal;
		surface.eta = surface.material.ior;
	} else {
		surface.ffnormal = surface.normal;
		surface.eta = 1.0 / surface.material.ior;
	}

	*psurface = surface;

	return true;
}

fn any_hit(ray: Ray, max_dist: f32) -> bool {
	if !ray_aabb_intersects(ray.origin, 1.0 / ray.dir, bvh[0].min, bvh[0].max) {
		return false;
	}

	var surface: SurfaceData;

	var t: f32 = INF;
	var intersected: bool = false;
	var trans: mat4x4<f32>;
	
	var bvh_idx: i32;
	var nodes_to_visit: array<i32, 64>;
	var visit_len = 0;
	var traversing_mesh = false;
	var ray_trans: Ray = ray;
	var ray_inv_dir = 1.0 / ray.dir;
	var cur_trans: mat4x4<f32>;
	
	nodes_to_visit[visit_len] = -1;
	visit_len++;

	nodes_to_visit[visit_len] = 0;
	visit_len++;

	while true {
		visit_len--;
		bvh_idx = nodes_to_visit[visit_len];
		if bvh_idx == -1 {
			if traversing_mesh {
				traversing_mesh = false;
				ray_trans = ray;
				ray_inv_dir = 1.0 / ray.dir;
				continue;
			} else {
				break;
			}
		}

		let node = bvh[bvh_idx];

		// Interior node of either scene/mesh
		if node.param2 == 0 {
			let left_node = bvh[node.param0];
			let right_node = bvh[node.param1];
			let left_hit = ray_aabb_intersects(ray_trans.origin, ray_inv_dir, left_node.min, left_node.max);
			let right_hit = ray_aabb_intersects(ray_trans.origin, ray_inv_dir, right_node.min, right_node.max);

			if left_hit {
				nodes_to_visit[visit_len] = node.param0;
				visit_len++;
			}

			if right_hit {
				nodes_to_visit[visit_len] = node.param1;
				visit_len++;
			}
		}
		// Leaf node of scene bvh
		else if node.param2 < 0 {
			let object_idx = -node.param2 - 1;
			let mesh_root_bvh_idx = node.param0;
			
			traversing_mesh = true;
			let transform = transforms[object_idx];
			ray_trans.origin = (transform.inv_world * vec4<f32>(ray_trans.origin, 1.0)).xyz;
			ray_trans.dir = normalize((transform.inv_world * vec4<f32>(ray_trans.dir, 0.0)).xyz);
			ray_inv_dir = 1.0 / ray_trans.dir;

			cur_trans = transform.world;

			nodes_to_visit[visit_len] = -1;
			visit_len++;
			nodes_to_visit[visit_len] = mesh_root_bvh_idx;
			visit_len++;
		}
		// Leaf node of mesh bvh
		else if node.param2 > 0 {
			for (var i = 0; i < node.param1; i += 1) {
				let v0 = vertices[triangles[node.param0 + i*3+0]];
				let v1 = vertices[triangles[node.param0 + i*3+1]];
				let v2 = vertices[triangles[node.param0 + i*3+2]];

				let p0 = (cur_trans * vec4<f32>(v0.pos, 1.0)).xyz;
				let p1 = (cur_trans * vec4<f32>(v1.pos, 1.0)).xyz;
				let p2 = (cur_trans * vec4<f32>(v2.pos, 1.0)).xyz;

				var uvt: vec4<f32>;
				var back_face: bool;

				if tri_intersection(ray, p0, p1, p2, &uvt, &back_face) && uvt.z < max_dist {
					t = uvt.z;
					trans = cur_trans;
					surface.back_face = back_face;

					intersected = true;
				}
			}
		}
	}

	return intersected;
}