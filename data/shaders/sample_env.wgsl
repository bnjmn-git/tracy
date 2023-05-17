#include "constants.wgsl"
#include "scene_data.wgsl"
#include "ray.wgsl"
#include "rand.wgsl"

fn env_binary_search(value: f32) -> vec2<f32> {
	let dim = textureDimensions(env_cdf);
	var lower = 0;
	var upper = dim.y - 1;
	while lower < upper {
		let mid = (lower + upper) / 2;
		if value < textureLoad(env_cdf, vec2<i32>(dim.x - 1, mid)).r {
			upper = mid;
		} else {
			lower = mid + 1;
		}
	}

	let y = clamp(lower, 0, dim.y - 1);

	lower = 0;
	upper = dim.x - 1;
	while lower < upper {
		let mid = (lower + upper) / 2;
		if value < textureLoad(env_cdf, vec2<i32>(mid, y)).r {
			upper = mid;
		} else {
			lower = mid + 1;
		}
	}

	let x = clamp(lower, 0, dim.x - 1);
	return vec2<f32>(f32(x), f32(y)) / vec2<f32>(dim);
}

fn lum(c: vec3<f32>) -> f32 {
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

fn env_eval_map(ray: Ray) -> vec4<f32> {
	let dim = textureDimensions(env_cdf);
	let total_lum = textureLoad(env_cdf, dim - 1).r;
	let theta = acos(clamp(ray.dir.y, -1.0, 1.0));
	let uv = vec2<f32>((PI + atan2(ray.dir.z, ray.dir.x)) * INV_PI * 0.5, theta * INV_PI);
	let color = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb;
	let pdf = lum(color) / total_lum;

	return vec4<f32>(
		color,
		pdf * f32(dim.x * dim.y) / (TWO_PI * PI * sin(theta))
	);
}

fn env_sample_map(color: ptr<function, vec3<f32>>) -> vec4<f32> {
	let dim = textureDimensions(env_cdf);
	let total_lum = textureLoad(env_cdf, dim - 1).r;
	let uv = env_binary_search(rand() * total_lum);
	*color = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb;
	var pdf = lum(*color) / total_lum;

	let phi = uv.x * TWO_PI;
	let theta = uv.y * PI;

	if sin(theta) == 0.0 {
		pdf = 0.0;
	}

	return vec4<f32>(
		-sin(theta) * cos(phi),
		cos(theta),
		-sin(theta) * sin(phi),
		(pdf * f32(dim.x * dim.y)) / (TWO_PI * PI * sin(theta))
	);
}