#include "constants.wgsl"
#include "rand.wgsl"
#include "intersection.wgsl"
#include "scene_data.wgsl"
#include "ray.wgsl"
#include "brdf.wgsl"
#include "sample_env.wgsl"
#include "bokeh.wgsl"

@group(3) @binding(0)
var output: texture_storage_2d<rgba32float, write>;

fn sample_env_map(dir: vec3<f32>) -> vec3<f32> {
	let theta = acos(dir.y);
	var uv: vec2<f32>;
	uv.x = atan2(dir.z, dir.x) * INV_PI * 0.5 + 0.5;
	uv.y = theta * INV_PI;

	return textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb;
}

fn calc_camera_ray(coord: vec2<f32>, aspect: f32) -> Ray {
	let clip_pos = coord * 2.0 - 1.0;

	let scale_y = tan(camera.fov * 0.5);
	let scale_x = aspect * scale_y;
	let ray_dir = normalize(
		clip_pos.x * scale_x * camera.right +
		clip_pos.y * scale_y * camera.up +
		camera.forward
	);

	let focal_pos = ray_dir * camera.focal_dist;
	
	var aperture_offset = vec3<f32>(bokeh_offset(camera.bokeh_blades), 0.0) * camera.aperture;
	aperture_offset = aperture_offset.x * camera.right + aperture_offset.y * camera.up;

	let final_ray_dir = normalize(focal_pos - aperture_offset);

	return Ray(camera.position + aperture_offset, final_ray_dir);
}

fn sample_direct_lights(r: Ray, surf: SurfaceData) -> vec3<f32> {
	var Ld: vec3<f32>;
	var Li: vec3<f32>;

	var scatter_dir: vec3<f32>;
	var scatter_pdf: f32;
	var scatter_f: vec3<f32>;
	var scatter_pos = surf.pos + surf.normal * EPS;

	// Environment Light
	{
		let dir_pdf = env_sample_map(&Li);
		let light_dir = dir_pdf.xyz;
		let light_pdf = dir_pdf.w;

		let shadow_ray = Ray(scatter_pos, light_dir);

		let in_shadow = any_hit(shadow_ray, INF);
		if !in_shadow {
			scatter_f = eval_brdf(surf, -r.dir, surf.ffnormal, light_dir, &scatter_pdf);
			if scatter_pdf > 0.0 {
				let misweight = power_heuristic(light_pdf, scatter_pdf);
				if misweight > 0.0 {
					Ld += misweight * Li * scatter_f / light_pdf;
				}
			}
		}
	}

	return Ld;
}

fn sample(ray: Ray) -> vec3<f32> {
	var radiance: vec3<f32> = vec3<f32>(0.);

	var ray: Ray = ray;
	var throughput: vec3<f32> = vec3<f32>(1.);

	var scatter_f: vec3<f32>;
	var scatter_l: vec3<f32>;
	var scatter_pdf: f32;
	
	for (var k = 0; k < 4; k += 1) {
		var surf: SurfaceData;

		if closest_hit(ray, &surf) {
			radiance += surf.material.emission * throughput;
			radiance += sample_direct_lights(ray, surf) * throughput;

			scatter_f = sample_brdf(surf, -ray.dir, surf.ffnormal, &scatter_l, &scatter_pdf);
			if scatter_pdf > 0.0 {
				throughput *= scatter_f / scatter_pdf;
			} else {
				break;
			}

			ray = Ray(surf.pos + scatter_l * EPS, scatter_l);
		} else {
			let env_col_pdf = env_eval_map(ray);
			var misweight = 1.0;
			if k > 0 {
				misweight = power_heuristic(scatter_pdf, env_col_pdf.w);
			}

			if misweight > 0.0 {
				radiance += misweight * env_col_pdf.rgb * throughput;
			}
			
			break;
		}
	}

	return radiance;
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let dim: vec2<i32> = textureDimensions(output);
	if id.x >= u32(dim.x) || id.y >= u32(dim.y) {
		return;
	}

	init_rng(id.xy, frame_idx);

	let fid = vec2<f32>(f32(id.x), f32(id.y));
	let fdim = vec2<f32>(f32(dim.x), f32(dim.y));
	var uv = fid / fdim;
	uv.y = 1.0 - uv.y;

	let jitter = vec2<f32>(rand(), rand()) * 2.0 - 1.0;
	let texel_size = 1.0 / fdim;

	let ray = calc_camera_ray(uv + jitter * texel_size, fdim.x/fdim.y);
	let radiance = sample(ray) * dot(camera.forward, ray.dir);

	textureStore(output, vec2<i32>(id.xy), vec4<f32>(radiance, 1.0));
}