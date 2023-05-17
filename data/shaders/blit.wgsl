@group(0) @binding(0)
var src_sampler: sampler;

@group(0) @binding(1)
var src_texture: texture_2d<f32>;



@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
	return textureSample(src_texture, src_sampler, uv);
}