struct Attributes {
	@location(0) vertex: vec3<f32>,
	@location(1) normal: vec3<f32>
}

struct VertexOutput {
	@builtin(position) clip_pos: vec4<f32>,
	@location(0) normal: vec3<f32>
}

@group(0) @binding(0)
var<uniform> mvp : mat4x4<f32>;

@vertex
fn vs_main(v: Attributes) -> VertexOutput {
	var o: VertexOutput;
	o.clip_pos = mvp * vec4<f32>(v.vertex, 1.0);
	o.normal = (mvp * vec4<f32>(v.normal, 0.0)).xyz;
	return o;
}

@fragment
fn fs_main(i: VertexOutput) -> @location(0) vec4<f32> {
	let light_dir = normalize(vec3<f32>(1.0, 1.0, -1.0));
	let radiance = vec3<f32>(saturate(dot(light_dir, i.normal)));
	return vec4<f32>(radiance, 1.0);
}