var<private> vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0, -1.0),
	vec2<f32>(-1.0,  1.0),
	vec2<f32>( 1.0,  1.0)
);

struct VertexOutput {
	@builtin(position) clip_pos: vec4<f32>,
	@location(0) uv: vec2<f32>
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
	var o: VertexOutput;
	let pos = vertices[idx];
	o.clip_pos = vec4<f32>(pos, 0.0, 1.0);

	var uv = pos * 0.5 + 0.5;
	uv.y = 1.0 - uv.y;
	o.uv = uv;

	return o;
}