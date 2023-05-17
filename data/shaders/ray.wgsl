struct Ray {
	origin: vec3<f32>,
	dir: vec3<f32>
}

fn ray_point(ray: Ray, t: f32) -> vec3<f32> {
	return ray.origin + ray.dir * t;
}