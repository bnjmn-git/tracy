#![allow(unused)]

use std::path::{PathBuf, Path};

use once_cell::sync::Lazy;

static DATA_PATH: Lazy<PathBuf> = Lazy::new(|| {
	let exe_dir = std::env::current_exe().unwrap();
	let mut dirs = exe_dir.ancestors().skip(1);
	if let Some(config_dir) = dirs.next() {
		let config_name = config_dir.file_name().expect("binary should be in a folder").to_str().unwrap();
		match config_name {
			"debug" | "release" => {},
			_ => panic!("expected binary to be in release or debug folder")
		}
	}

	if let Some(target_dir) = dirs.next() {
		let target_name = target_dir.file_name().expect("binary should be in target/{{config}}").to_str().unwrap();
		if target_name != "target" {
			panic!("expected binary to be in target/{{config}} folder");
		}
	}

	let root = dirs.next().unwrap();
	let data = root.join("data");
	assert!(data.exists(), "data path does not exist!");

	data
});

static SHADERS_PATH: Lazy<PathBuf> = Lazy::new(|| {
	DATA_PATH.join("shaders")
});

static TEXTURES_PATH: Lazy<PathBuf> = Lazy::new(|| {
	DATA_PATH.join("textures")
});

static MESH_PATH: Lazy<PathBuf> = Lazy::new(|| {
	DATA_PATH.join("mesh")
});

pub fn data() -> &'static Path {
	&DATA_PATH
}

pub fn shaders() -> &'static Path {
	&SHADERS_PATH
}

pub fn textures() -> &'static Path {
	&TEXTURES_PATH
}

pub fn mesh() -> &'static Path {
	&MESH_PATH
}