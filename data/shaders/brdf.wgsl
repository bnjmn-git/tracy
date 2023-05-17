#include "constants.wgsl"
#include "sampling.wgsl"

fn luminance(c: vec3<f32>) -> f32 {
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

fn schlick_fresnel(u: f32) -> f32 {
	let m = clamp(1.0 - u, 0.0, 1.0);
	return pow(m, 5.0);
}

fn dielectric_fresnel(cos_theta_i: f32, eta: f32) -> f32 {
	let sin_theta_t_sq = eta*eta*(1.0 - cos_theta_i*cos_theta_i);
	if sin_theta_t_sq > 1.0 {
		return 1.0;
	}

	let cos_theta_t = sqrt(max(0.0, 1.0 - sin_theta_t_sq));
	let rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
	let rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

	return 0.5 * (rs*rs + rp*rp);
}

fn disney_fresnel(m: Material, eta: f32, LdotH: f32, VdotH: f32) -> f32 {
	let f_metallic = schlick_fresnel(LdotH);
	let f_dielectric = dielectric_fresnel(abs(VdotH), eta);
	return mix(f_dielectric, f_metallic, m.metallic);
}

fn eval_diffuse(
	m: Material, 
	V: vec3<f32>,
	L: vec3<f32>,
	H: vec3<f32>,
	pdf: ptr<function, f32>
) -> vec3<f32> {
	*pdf = 0.0;
	if(L.z <= 0.) {
		return vec3<f32>(0.);
	}

	// *pdf = L.z * INV_PI;
	// return m.albedo * INV_PI;

	let LdotH = saturate(dot(L, H));
	let fl = schlick_fresnel(L.z);
	let fv = schlick_fresnel(V.z);
	let fh = schlick_fresnel(LdotH);
	let f90 = 0.5 + 2.0 * LdotH * LdotH * m.roughness;
	let fd = mix(1.0, f90, fl) * mix(1.0, f90, fv);

	let fss90 = LdotH * LdotH * m.roughness;
	let fss = mix(1.0, fss90, fl) * mix(1.0, fss90, fv);
	let ss = 1.25 * (fss * (1.0 / (L.z + V.z) - 0.5) + 0.5);

	*pdf = L.z * INV_PI;
	return (1.0 - m.metallic) * (1.0 - m.spec_trans) * mix(fd, ss, 0.0) * m.albedo * INV_PI;
}

fn eval_spec_reflection(
	m: Material,
	eta: f32,
	spec_col: vec3<f32>,
	V: vec3<f32>,
	L: vec3<f32>,
	H: vec3<f32>,
	pdf: ptr<function, f32>
) -> vec3<f32> {
	*pdf = 0.0;
	if L.z < 0.0 {
		return vec3<f32>(0.0);
	}

	let f_mix = disney_fresnel(m, eta, dot(L, H), dot(V, H));
	let F = mix(spec_col, vec3<f32>(1.0), f_mix);
	let D = gtr2(H.z, m.roughness);
	let G1 = smith_G(abs(V.z), m.roughness);
	let G2 = G1 * smith_G(abs(L.z), m.roughness);

	*pdf = D * G1 / (4.0 * V.z);
	return F * D * G2 / (4.0 * L.z * V.z);
}

fn eval_spec_refraction(
	m: Material,
	eta: f32,
	V: vec3<f32>,
	L: vec3<f32>,
	H: vec3<f32>,
	pdf: ptr<function, f32>
) -> vec3<f32> {
	*pdf = 0.0;
	if L.z > 0.0 {
		return vec3<f32>(0.0);
	}

	let F = dielectric_fresnel(abs(dot(V, H)), eta);
	let D = gtr2(H.z, m.roughness);
	var denom = dot(L, H) + dot(V, H) * eta;
	denom *= denom;
	let G1 = smith_G(abs(V.z), m.roughness);
	let G2 = G1 * smith_G(abs(L.z), m.roughness);
	let eta2 = eta*eta;
	let jacobian = abs(dot(L, H)) / denom;

	*pdf = G1 * max(0.0, dot(V, H)) * D * jacobian / V.z;
	return (
		pow(m.albedo, vec3<f32>(0.5)) * (1.0 - m.metallic) * m.spec_trans * (1.0 - F) *
		D * G2 * abs(dot(V, H)) * jacobian * eta2 /
		abs(L.z * V.z)
	);

}

fn get_spec_colour(
	m: Material,
	eta: f32,
) -> vec3<f32> {
	let lum = luminance(m.albedo);
	var tint: vec3<f32>;
	if lum > 0.0 {
		tint = m.albedo / lum;
	} else {
		tint = vec3<f32>(1.0);
	}

	let F0 = (1.0 - eta) / (1.0 + eta);
	return mix(F0 * F0 * mix(vec3<f32>(1.0), tint, m.spec_tint), m.albedo, m.metallic);
}

fn calc_lobe_probabilities(
	m: Material,
	eta: f32,
	spec_col: vec3<f32>,
	fresnel: f32,
	diffuse_wt: ptr<function, f32>,
	spec_reflect_wt: ptr<function, f32>,
	spec_refract_wt: ptr<function, f32>,
) {
	let lum = luminance(m.albedo);
	*diffuse_wt = lum * (1.0 - m.metallic) * (1.0 - m.spec_trans);
	*spec_reflect_wt = luminance(mix(spec_col, vec3<f32>(1.0), fresnel));
	*spec_refract_wt = (1.0 - fresnel) * (1.0 - m.metallic) * m.spec_trans * lum;
	let total = *diffuse_wt + *spec_reflect_wt + *spec_refract_wt;

	*diffuse_wt /= total;
	*spec_reflect_wt /= total;
	*spec_refract_wt /= total;
}

fn refract(I: vec3<f32>, N: vec3<f32>, eta: f32) -> vec3<f32> {
	let k = 1.0 - eta*eta*(1.0 - dot(N, I) * dot(N, I));
	if k < 0.0 {
		return vec3<f32>(0.0);
	} else {
		return eta * I - (eta * dot(N, I) + sqrt(k)) * N;
	}
}

fn sample_brdf(
	surf: SurfaceData,
	V: vec3<f32>,
	N: vec3<f32>,
	pL: ptr<function, vec3<f32>>,
	pdf: ptr<function, f32>
) -> vec3<f32> {
	var L = *pL;
	var f = vec3<f32>(0.0);

	var T: vec3<f32>;
	var B: vec3<f32>;
	compute_orthonormal(N, &T, &B);
	let V = to_local(T, B, N, V);

	var spec_col: vec3<f32> = get_spec_colour(surf.material, surf.eta);

	var diffuse_wt: f32;
	var spec_reflect_wt: f32;
	var spec_refract_wt: f32;
	let approx_fresnel = disney_fresnel(surf.material, surf.eta, V.z, V.z);

	calc_lobe_probabilities(
		surf.material,
		surf.eta,
		spec_col,
		approx_fresnel,
		&diffuse_wt,
		&spec_reflect_wt,
		&spec_refract_wt
	);

	var cdf: array<f32, 3>;
	cdf[0] = diffuse_wt;
	cdf[1] = cdf[0] + spec_reflect_wt;
	cdf[2] = cdf[1] + spec_refract_wt;

	var r1 = rand();
	let r2 = rand();
	if r1 < cdf[0] {
		r1 /= cdf[0];
		L = cos_sample_hemisphere(r1, r2);
		let H = normalize(L + V);
		f = eval_diffuse(surf.material, V, L, H, pdf);
		// f = vec3<f32>(0.0);
		*pdf *= diffuse_wt;
	} else {
		r1 = (r1 - cdf[0]) / (cdf[2] - cdf[0]);
		var H = sample_ggx(V, surf.material.roughness, r1, r2);
		H.z = abs(H.z);

		let fresnel = disney_fresnel(surf.material, surf.eta, dot(V, H), dot(V, H));
		// let fresnel = schlick_fresnel(dot(V, H));
		let F = 1.0 - ((1.0 - fresnel) * surf.material.spec_trans * (1.0 - surf.material.metallic));

		if rand() < F {
			L = normalize(reflect(-V, H));
			f = eval_spec_reflection(surf.material, surf.eta, spec_col, V, L, H, pdf);
			// f = vec3<f32>(0.0);
			*pdf *= F;
		} else {
			L = normalize(refract(-V, H, surf.eta));
			f = eval_spec_refraction(surf.material, surf.eta, V, L, H, pdf);
			*pdf *= 1.0 - F;
		}

		*pdf *= spec_reflect_wt + spec_refract_wt;
	}

	L = to_world(T, B, N, L);
	*pL = L;
	return f * abs(dot(N, L));
}

fn eval_brdf(
	surf: SurfaceData,
	V: vec3<f32>,
	N: vec3<f32>,
	L: vec3<f32>,
	pbsdf_pdf: ptr<function, f32>
) -> vec3<f32> {
	var f = vec3<f32>(0.0);
	var bsdf_pdf = 0.0;

	var T: vec3<f32>;
	var B: vec3<f32>;
	compute_orthonormal(N, &T, &B);
	let V = to_local(T, B, N, V);
	let L = to_local(T, B, N, L);

	var H: vec3<f32>;
	if L.z > 0.0 {
		H = normalize(L + V);
	} else {
		H = normalize(L + V * surf.eta);
	}

	H.z = abs(H.z);

	var spec_col: vec3<f32> = get_spec_colour(surf.material, surf.eta);

	var diffuse_wt: f32;
	var spec_reflect_wt: f32;
	var spec_refract_wt: f32;
	let fresnel = disney_fresnel(surf.material, surf.eta, dot(L, H), dot(V, H));


	calc_lobe_probabilities(
		surf.material,
		surf.eta,
		spec_col,
		fresnel,
		&diffuse_wt,
		&spec_reflect_wt,
		&spec_refract_wt
	);

	let r1 = rand();
	let r2 = rand();
	var pdf: f32;

	if diffuse_wt > 0.0 && L.z > 0.0 {
		f += eval_diffuse(surf.material, V, L, H, &pdf);
		bsdf_pdf += pdf * diffuse_wt;
	}

	if spec_reflect_wt > 0.0 && L.z > 0.0 && V.z > 0.0 {
		f += eval_spec_reflection(surf.material, surf.eta, spec_col, V, L, H, &pdf);
		bsdf_pdf += pdf * spec_reflect_wt;
	}

	if spec_reflect_wt > 0.0 && L.z < 0.0 {
		f += eval_spec_refraction(surf.material, surf.eta, V, L, H, &pdf);
		bsdf_pdf += pdf * spec_refract_wt;
	}

	*pbsdf_pdf = bsdf_pdf;

	return f * abs(L.z);
}