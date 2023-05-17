use std::{sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex, Condvar}, thread, collections::VecDeque};

#[allow(unused)]
#[derive(Clone)]
pub struct ThreadPool {
	workers: Arc<Vec<Worker>>,
	state: Arc<SharedState>
}

impl ThreadPool {
	pub fn new() -> Self {
		let state = Arc::new(SharedState {
			is_running: true.into(),
			jobs: Mutex::new(Jobs::new()),
			cond: Condvar::new()
		});

		let num_threads = 4.min(std::thread::available_parallelism().unwrap().into());
		log::info!("Initializing {} worker threads", num_threads);

		Self {
			workers: Arc::new(
				std::iter::repeat_with(|| {
					Worker::new(state.clone())
				})
				.take(num_threads)
				.collect()
			),
			state
		}
	}

	pub fn add_job(&self, f: impl FnOnce() + Send + 'static) {
		let job = Box::new(f);

		let mut guard = self.state.jobs.lock().unwrap();
		guard.push_back(job);
		drop(guard);

		self.state.cond.notify_all();
	}
}

impl Drop for ThreadPool {
	fn drop(&mut self) {
		self.state.is_running.store(false, Ordering::Release);
		self.state.cond.notify_all();
	}
}

type Job = Box<dyn FnOnce() + Send + 'static>; 
type Jobs = VecDeque<Job>;

struct SharedState {
	jobs: Mutex<Jobs>,
	cond: Condvar,
	is_running: AtomicBool
}

struct Worker {
	thread: Option<thread::JoinHandle<()>>
}

impl Drop for Worker {
	fn drop(&mut self) {
		if let Some(thread) = self.thread.take() {
			thread.join().unwrap();
		}
	}
}

impl Worker {
	fn new(state: Arc<SharedState>) -> Self {
		Self {
			thread: Some(thread::spawn(move || Self::run(state)))
		}
	}

	fn run(state: Arc<SharedState>) {
		log::debug!("Worker {:?} starting", std::thread::current().id());

		'l: loop {
			let mut jobs_guard = state.jobs.lock().unwrap();
			jobs_guard = state.cond.wait_while(jobs_guard, |jobs| {
				state.is_running.load(Ordering::Acquire) && jobs.is_empty()
			}).unwrap();
			
			if !state.is_running.load(Ordering::Acquire) {
				break 'l;
			}

			let job = jobs_guard.pop_front().unwrap();
			drop(jobs_guard);

			job();
		}

		log::debug!("Worker {:?} done", std::thread::current().id());
	}
}