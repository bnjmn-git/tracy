@group(0) @binding(0)
var src_sampler: sampler;

@group(0) @binding(1)
var src_tex: texture_2d<f32>;

@group(0) @binding(2)
var output: texture_storage_2d<r32float, read_write>;

fn lum(c: vec3<f32>) -> f32 {
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let dim = textureDimensions(output);
	let id = i32(id.x);

	let fdim = vec2<f32>(dim);
	
	if id < dim.y {
		var acc = 0.0;
		for (var i = 0; i < dim.x; i += 1) {
			let uv = vec2<f32>(f32(i), f32(id)) / fdim;
			acc += lum(textureSampleLevel(src_tex, src_sampler, uv, 0.0).xyz);
			textureStore(output, vec2<i32>(i, id), vec4<f32>(acc, 0.0,0.0,0.0));
		}
	}

	storageBarrier();

	if id != 0 {
		return;
	}

	for (var y = 1; y < dim.y; y += 1) {
		let base = textureLoad(output, vec2<i32>(dim.x - 1, y - 1)).r;
		for (var x = 0; x < dim.x; x += 1) {
			let acc = textureLoad(output, vec2<i32>(x, y)).r + base;
			textureStore(output, vec2<i32>(x, y), vec4<f32>(acc, 0.0, 0.0, 0.0));
		}
	}
}