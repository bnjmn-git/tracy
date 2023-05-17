#include "constants.wgsl"
#include "rand.wgsl"

fn point_in_triangle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, p: vec2<f32>) -> vec2<f32> {
	return (1.0 - sqrt(p.x)) * a.yx + sqrt(p.x) * (1.0 - p.y) * b.yx + p.y * sqrt(p.x) * c.yx;
}

fn signed_tri_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

fn tri_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
	return abs(signed_tri_area(a, b, c)) * 0.5;
}

fn polygon_bokeh(nblades: i32, r1: f32, r2: f32) -> vec2<f32> {
	let delta_theta = 2.0 * PI / f32(nblades);
	var theta = 0.0;

	let a = vec2<f32>(0.0);

	let tri_choice = i32(rand() * f32(nblades));
	theta += f32(tri_choice - 1) * delta_theta;
	let b = vec2<f32>(cos(theta), sin(theta));
	theta += delta_theta;
	let c = vec2<f32>(cos(theta), sin(theta));

	let offset = point_in_triangle(a, b, c, vec2<f32>(r1, r2));
	return offset;
}

fn bokeh_offset(nblades: i32) -> vec2<f32> {
	switch nblades {
		case 0, 1, 2 {
			let r = sqrt(rand());
			let theta = 2.0 * PI * rand();
			return r * vec2<f32>(cos(theta), sin(theta));
		}
		case 4 {
			var offset = vec2<f32>(rand(), rand()) * 2.0 - 1.0;
			offset *= sqrt(2.0) * 0.5;
			return offset;
		}
		default {
			return polygon_bokeh(nblades, rand(), rand());
		}
	}
}