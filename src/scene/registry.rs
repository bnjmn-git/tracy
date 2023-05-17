use std::{marker::PhantomData, hash::Hash, ptr};

use super::Resource;

pub struct Id<T>(usize, PhantomData<T>);
pub type Index = u32;
pub type Epoch = u32;

pub trait TypedId: Clone + Copy {
	fn zip(index: Index, epoch: Epoch) -> Self;
	fn unzip(&self) -> (Index, Epoch);
}

impl<T> TypedId for Id<T> {
	fn zip(index: Index, epoch: Epoch) -> Id<T> {
		Id((index as usize) | (epoch as usize) << 32, PhantomData)
	}

	fn unzip(&self) -> (u32, u32) {
		let index = self.0 as u32;
		let epoch = (self.0 >> 32) as u32;
		(index, epoch)
	}
}

impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
	fn clone(&self) -> Self {
		Self(self.0, PhantomData)
	}
}

impl<T> Hash for Id<T> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		state.write_usize(self.0);
	}
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

struct IdManager {
	free: Vec<Index>,
	epochs: Vec<Epoch>
}

impl IdManager {
	fn new() -> Self {
		Self {
			epochs: Vec::new(),
			free: Vec::new()
		}
	}

	fn alloc<I: TypedId>(&mut self) -> I {
		if let Some(index) = self.free.pop() {
			let epoch = self.epochs[index as usize];
			I::zip(index, epoch)
		} else {
			let index = self.epochs.len() as Index;
			let epoch = 0;
			self.epochs.push(epoch);
			I::zip(index, epoch)
		}
	}

	fn free<I: TypedId>(&mut self, id: I) {
		let (index, epoch) = id.unzip();
		let pe = &mut self.epochs[index as usize];
		self.free.push(index);

		assert_eq!(*pe, epoch);

		*pe += 1;
	}
}

enum Element<T> {
	Vacant,
	Occupied(T, Epoch)
}

pub(super) struct Registry<T, I> {
	id_manager: IdManager,
	map: Vec<Element<T>>,
	_p: PhantomData<I>
}

impl<T: Resource, I: TypedId> Registry<T, I> {
	pub fn new() -> Self {
		Self {
			id_manager: IdManager::new(),
			map: Vec::new(),
			_p: PhantomData
		}
	}

	pub fn register(&mut self, value: T) -> I {
		let id: I = self.id_manager.alloc();
		let (index, epoch) = id.unzip();
		let index = index as usize;
		if index >= self.map.len() {
			self.map.resize_with(index + 1, || Element::Vacant);
		}

		match std::mem::replace(&mut self.map[index], Element::Occupied(value, epoch)) {
			Element::Vacant => {},
			_ => panic!("Index {} is already occupied", index)
		};

		id
	}

	pub fn unregister(&mut self, id: I) -> T {
		self.id_manager.free(id);
		let (index, epoch) = id.unzip();
		match std::mem::replace(&mut self.map[index as usize], Element::Vacant) {
			Element::Occupied(value, storage_epoch) => {
				assert_eq!(storage_epoch, epoch);
				value
			}
			Element::Vacant => panic!("Cannot remove from empty index {}", index)
		}
	}

	pub fn get(&self, id: I) -> Option<&T> {
		let (index, epoch) = id.unzip();
		match self.map.get(index as usize) {
			Some(Element::Occupied(value, storage_epoch)) => {
				assert_eq!(epoch, *storage_epoch);
				Some(value)
			}
			Some(Element::Vacant) => panic!("Index {} is vacant", index),
			None => None
		}
	}

	pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
		let (index, epoch) = id.unzip();
		match self.map.get_mut(index as usize) {
			Some(Element::Occupied(value, storage_epoch)) => {
				assert_eq!(epoch, *storage_epoch);
				Some(value)
			}
			Some(Element::Vacant) => panic!("Index {} is vacant", index),
			None => None
		}
	}

	pub fn contains(&self, id: I) -> bool {
		let (index, epoch) = id.unzip();
		match self.map.get(index as usize) {
			Some(Element::Occupied(_, element_epoch)) => epoch == *element_epoch,
			_ => false
		}
	}

	pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
		self.map.iter().enumerate().filter_map(|(index, element)| {
			if let Element::Occupied(value, epoch) = element {
				Some((I::zip(index as u32, *epoch), value))
			} else {
				None
			}
		})
	}

	pub fn iter_mut(&mut self) -> impl Iterator<Item = (I, &mut T)> {
		self.map.iter_mut().enumerate().filter_map(|(index, element)| {
			if let Element::Occupied(value, epoch) = element {
				Some((I::zip(index as u32, *epoch), value))
			} else {
				None
			}
		})
	}

	pub fn get_stored(&self, id: I) -> Option<Stored<I>> {
		self.get(id).map(|resource| {
			Stored {
				id,
				ref_count: resource.ref_count().clone()
			}
		})
	}
}

pub struct RefCount(ptr::NonNull<u32>);

impl RefCount {
	pub fn new() -> Self {
		let bx = Box::new(1);
		Self(unsafe { ptr::NonNull::new_unchecked(Box::into_raw(bx)) })
	}

	pub fn load(&self) -> u32 {
		unsafe { *self.0.as_ref() }
	}
}

impl Clone for RefCount {
	fn clone(&self) -> Self {
		unsafe {
			*self.0.as_ptr() += 1;
			Self(self.0)
		}
	}
}

impl Drop for RefCount {
	fn drop(&mut self) {
		unsafe {
			*self.0.as_mut() -= 1;
			if *self.0.as_ref() == 0 {
				drop(Box::from_raw(self.0.as_ptr()));
			}
		}
	}
}

#[derive(Clone)]
pub struct Stored<I: TypedId> {
	pub id: I,
	pub ref_count: RefCount
}