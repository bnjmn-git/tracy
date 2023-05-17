@group(0) @binding(0)
var<uniform> mvp : mat4x4<f32>;

@vertex
fn vs_main(@location(0) v: vec3<f32>) -> @builtin(position) vec4<f32> {
	return mvp * vec4<f32>(v, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}