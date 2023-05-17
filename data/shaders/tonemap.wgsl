#include "full_screen_vert.wgsl"

@group(0) @binding(0)
var scene_sampler: sampler;

@group(0) @binding(1)
var scene_tex: texture_2d<f32>;

var<private> ACESInputMat: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.59719, 0.35458, 0.04823),
    vec3<f32>(0.07600, 0.90834, 0.01566),
    vec3<f32>(0.02840, 0.13383, 0.83777)
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
var<private> ACESOutputMat: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.60475, -0.53108, -0.07367),
    vec3<f32>(-0.10208,  1.10813, -0.00605),
    vec3<f32>(-0.00327, -0.07276,  1.07602)
);

fn RRTAndODTFit(v: vec3<f32>) -> vec3<f32>
{
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn aces_map(x: vec3<f32>) -> vec3<f32> {
	let a: f32 = 2.51;
	let b: f32 = 0.03;
	let c: f32 = 2.43;
	let d: f32 = 0.59;
	let e: f32 = 0.14;
	return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
	let uv = vec2<f32>(uv.x, uv.y);
	var col: vec3<f32> = textureSample(scene_tex, scene_sampler, uv).rgb;
	col = col * ACESInputMat;
	col = RRTAndODTFit(col);
	col = col * ACESOutputMat;
	col = saturate(col);

	return vec4<f32>(col, 1.0);
}