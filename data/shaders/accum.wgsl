// #include "full_screen_vert.wgsl"

@group(0) @binding(0)
var<uniform> iterations: u32;

@group(0) @binding(1)
var src_sampler: sampler;

@group(1) @binding(0)
var src_tex: texture_2d<f32>;

@group(1) @binding(1)
var output: texture_storage_2d<rgba32float, read_write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let dim: vec2<i32> = textureDimensions(output);
	if id.x >= u32(dim.x) || id.y >= u32(dim.y) {
		return;
	}

	let fid = vec2<f32>(f32(id.x), f32(id.y));
	let fdim = vec2<f32>(f32(dim.x), f32(dim.y));
	var uv = fid / fdim;
	// uv.y = 1.0 - uv.y;

	let src = textureSampleLevel(src_tex, src_sampler, uv, 0.0);
	let dst = textureLoad(output, vec2<i32>(id.xy));
	textureStore(output, vec2<i32>(id.xy), (src + dst * max(1.0, f32(iterations - 1u))) / f32(iterations));
}