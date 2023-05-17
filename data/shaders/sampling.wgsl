#include "constants.wgsl"

fn cos_sample_hemisphere(r1: f32, r2: f32) -> vec3<f32> {
	var dir: vec3<f32>;
	
	let cos_theta = sqrt(r1);
	let sin_theta = sqrt(1. - cos_theta*cos_theta);

	let phi = 2. * PI * r2;
	dir.x = sin_theta * cos(phi);
	dir.y = sin_theta * sin(phi);
	dir.z = sqrt(max(0., 1. - dir.x*dir.x - dir.y*dir.y));
	return dir;
}

fn sample_hemisphere(r1: f32, r2: f32) -> vec3<f32> {
	let cos_theta = r1;
	let sin_theta = sqrt(max(0., 1. - r1*r1));
	let phi = r2 * 2. * PI;
	return vec3<f32>(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

fn to_world(x: vec3<f32>, y: vec3<f32>, z: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
	return v.x * x + v.y * y + v.z * z;
}

fn to_local(x: vec3<f32>, y: vec3<f32>, z: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(dot(v, x), dot(v, y), dot(v, z));
}

fn compute_orthonormal(N: vec3<f32>, T: ptr<function, vec3<f32>>, B: ptr<function, vec3<f32>>) {
	var up: vec3<f32>;
	if(abs(N.z) < 0.999) {
		up = vec3<f32>(0., 0., 1.);
	} else {
		up = vec3<f32>(1., 0., 0.);
	}

	*T = normalize(cross(up, N));
	*B = cross(N, *T);
}

fn gtr1(NdotH: f32, a: f32) -> f32 {
	let a2 = a*a;
	let t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (PI * log(a2) * t);
}

fn sample_gtr1(a: f32, r1: f32, r2: f32) -> vec3<f32> {
	let a2 = a*a;
	let phi = r1 * TWO_PI;
	let cos_theta = sqrt((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
	let sin_theta = clamp(sqrt(1.0 - (cos_theta*cos_theta)), 0.0, 1.0);

	return vec3<f32>(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

fn gtr2(NdotH: f32, a: f32) -> f32 {
	let a2 = a*a;
	let t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return a2 / (PI*t*t);
}

fn sample_gtr2(a: f32, r1: f32, r2: f32) -> vec3<f32> {
	let a2 = a*a;
	let phi = r1 * TWO_PI;
	let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
	let sin_theta = clamp(sqrt(1.0 - (cos_theta*cos_theta)), 0.0, 1.0);

	return vec3<f32>(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

fn sample_ggx(V: vec3<f32>, a: f32, r1: f32, r2: f32) -> vec3<f32> {
	let Vh = normalize(vec3<f32>(a * V.x, a * V.y, V.z));
	let len2 = Vh.x * Vh.x + Vh.y * Vh.y;
	var T1: vec3<f32>;
	if len2 > 0.0 {
		T1 = vec3<f32>(-Vh.y, Vh.x, 0.0) * inverseSqrt(len2);
	} else {
		T1 = vec3<f32>(1.0, 0.0, 0.0);
	}

	let T2 = cross(Vh, T1);

	let r = sqrt(r1);
	let phi = TWO_PI * r2;
	let t1 = r * cos(phi);
	var t2 = r * sin(phi);
	let s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

	let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;

	return normalize(vec3<f32>(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
}

// fn sample_ggx(N: vec3<f32>, a: f32, r1: f32, r2: f32) -> vec3<f32> {
// 	let theta = atan(a * sqrt(r1 / (1.0 - r1)));
// 	let phi = 2.0 * PI * r2;
// 	return normalize(vec3<f32>(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)));
// }

fn smith_G(NdotV: f32, alpha: f32) -> f32 {
	let a = alpha*alpha;
	let b = NdotV * NdotV;
	return (2.0 * NdotV) / (NdotV + sqrt(a + b - a * b));
}

fn smith_ggx_G(NdotV: f32, NdotL: f32, a: f32) -> f32 {
	let a2 = a*a;
	let denom_a = NdotV * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
	let denom_b = NdotL * sqrt(a2 + (1.0 - a2) * NdotL * NdotL);
	return 2.0 * NdotV * NdotL / (denom_a + denom_b);
}

fn power_heuristic(a: f32, b: f32) -> f32 {
	let t = a*a;
	return t / (b*b + t);
}