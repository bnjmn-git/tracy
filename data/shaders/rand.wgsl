var<private> seed_: vec4<u32>;

fn init_rng(p: vec2<u32>, frame: u32) {
	seed_ = vec4<u32>(p, frame, p.x + p.y);
}

fn pcg4d_(v: vec4<u32>) -> vec4<u32> {
	var a = v;
	a = a * 1664525u + 1013904223u;
	a.x += a.y * a.w;
	a.y += a.z * a.x;
	a.z += a.x * a.y;
	a.w += a.y * a.z;
	a = a ^ (a >> vec4<u32>(16u));
	a.x += a.y * a.w;
	a.y += a.z * a.x;
	a.z += a.x * a.y;
	a.w += a.y * a.z;
	return a;
}

fn rand() -> f32 {
	seed_ = pcg4d_(seed_);
	return f32(seed_.x) / f32(0xffffffffu);
}