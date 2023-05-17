use std::{path::{PathBuf, Path}, io::{self, Read}, collections::HashMap, fs::File, borrow::Cow};

use egui_wgpu::wgpu;
use once_cell::{sync};
use regex::Regex;

static INCLUDE_REGEX: sync::Lazy<Regex> = sync::Lazy::new(|| {
	Regex::new(r#"^#include "(.*)"$"#).unwrap()
});

static MACRO_REGEX: sync::Lazy<Regex> = sync::Lazy::new(|| {
	Regex::new(r#"^#define (\S*) (.*)$"#).unwrap()
});

#[derive(Debug)]
pub enum ShaderPreprocessError {
	Io(io::Error),
	MultipleDefines(String),
}

pub struct ShaderBuilder {
	source_string: String,
}

impl ShaderBuilder {
	pub fn new(source_path: impl AsRef<Path>) -> Result<Self, ShaderPreprocessError> {
		let mut includes = Vec::new();
		let source_path = source_path.as_ref();
		let module = load_shader_module(source_path, &mut includes)?;

		Ok(Self {
			source_string: module.source
		})
	}

	pub fn build(self) -> wgpu::ShaderModuleDescriptor<'static> {
		wgpu::ShaderModuleDescriptor {
			label: None,
			source: wgpu::ShaderSource::Wgsl(Cow::Owned(self.source_string))
		}
	}
}

struct Module {
	source: String,
	defines: HashMap<String, String>
}

fn load_shader_module(module_path: &Path, includes: &mut Vec<PathBuf>) -> Result<Module, ShaderPreprocessError> {
	let mut source = String::new();
	_ = File::open(module_path)
	.map(|mut f| f.read_to_string(&mut source))
	.map_err(|e| ShaderPreprocessError::Io(e))?;

	includes.push(module_path.to_owned());

	let base_path = module_path.parent().unwrap();
	let mut processed = String::new();

	let mut defines = HashMap::new();

	let mut modules = Vec::new();

	for line in source.lines() {
		if let Some(captures) = INCLUDE_REGEX.captures(line) {
			let include_path = base_path.join(&captures[1]);
			if includes.contains(&include_path) {
				continue;
			}

			let module = load_shader_module(&include_path, includes)?;

			for name in module.defines.keys() {
				if defines.contains_key(name) {
					return Err(ShaderPreprocessError::MultipleDefines(name.to_owned()));
				}
			}
			
			defines.extend(module.defines);
			modules.push(module.source);
		} else if let Some(captures) = MACRO_REGEX.captures(line) {
			let name = captures[1].to_string();
			
			if defines.contains_key(&name) {
				return Err(ShaderPreprocessError::MultipleDefines(name));
			}
			
			let value = captures[2].to_string();
			defines.insert(name, value);
		} else {
			processed.push_str(line);
			processed.push('\n');
		}
	}

	for (name, value) in defines.iter() {
		processed = processed.replace(name, value);
	}

	modules.push(processed);
	processed = modules.concat();

	Ok(Module {
		source: processed,
		defines
	})
}