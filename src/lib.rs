use crossbeam::{
  atomic::AtomicCell,
  queue::{ArrayQueue, SegQueue},
};
use evmap::{ReadHandle, ShallowCopy, WriteHandle};
use std::{
  collections::HashMap,
  hash::{Hash, Hasher},
  mem::ManuallyDrop,
  sync::{Arc, Mutex, MutexGuard},
};

pub trait Key: Send + Sync + Hash + Eq + Clone + 'static {}
impl<T: Send + Sync + Hash + Eq + Clone + 'static> Key for T {}

pub trait Value: Send + Sync + Clone + Eq + Hash + 'static {}
impl<T: Send + Sync + Clone + Eq + Hash + 'static> Value for T {}

enum EventualOp<K: Key, V: Value> {
  SetValue(K, Arc<V>),
  Extend(HashMap<K, Arc<V>>),
  InvalidateKey(K),
  GarbageCollect(K, Arc<AtomicCell<usize>>),
  PurgeCache,
}

#[derive(Clone)]
struct ValueBox<T: Value> {
  value: Arc<T>,
  lru_counter: Arc<AtomicCell<usize>>,
}

impl<T> PartialEq for ValueBox<T>
where
  T: Value,
{
  fn eq(&self, other: &Self) -> bool {
    self.value.eq(&other.value)
  }
}

impl<T> Eq for ValueBox<T> where T: Value {}
impl<T> Hash for ValueBox<T>
where
  T: Value,
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.value.hash(state);
  }
}

impl<T> ShallowCopy for ValueBox<T>
where
  T: Value,
{
  unsafe fn shallow_copy(&self) -> ManuallyDrop<Self> {
    ManuallyDrop::new(ValueBox {
      value: ManuallyDrop::into_inner(self.value.shallow_copy()),
      lru_counter: ManuallyDrop::into_inner(self.lru_counter.shallow_copy()),
    })
  }
}

impl<T> From<Arc<T>> for ValueBox<T>
where
  T: Value,
{
  fn from(value: Arc<T>) -> Self {
    let lru_counter = Arc::new(AtomicCell::new(0));
    ValueBox { value, lru_counter }
  }
}

struct EventualWriter<K: Key, V: Value> {
  write_handle: Mutex<WriteHandle<K, ValueBox<V>>>,
  access_log: ArrayQueue<(K, Arc<AtomicCell<usize>>)>,
  recycling_bin: ArrayQueue<(K, Arc<AtomicCell<usize>>)>,
  pending_ops: SegQueue<EventualOp<K, V>>,
}

impl<K, V> EventualWriter<K, V>
where
  K: Key,
  V: Value,
{
  fn new(write_handle: WriteHandle<K, ValueBox<V>>, capacity: usize) -> Arc<Self> {
    let write_handle = Mutex::from(write_handle);
    let access_log = ArrayQueue::new(capacity);
    let recycling_bin = ArrayQueue::new(capacity);
    let pending_ops = SegQueue::new();

    Arc::new(EventualWriter {
      write_handle,
      access_log,
      recycling_bin,
      pending_ops,
    })
  }

  fn set(
    &self,
    key: K,
    value: ValueBox<V>,
    write_handle: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>,
  ) {
    self.push_access_entry(key.clone(), &value);
    write_handle.update(key, value);
  }

  fn push_access_entry(&self, key: K, value: &ValueBox<V>) {
    let lru_counter = value.lru_counter.to_owned();
    (&*lru_counter).fetch_add(1);

    let mut entry = (key, lru_counter);

    loop {
      entry = match self.access_log.push(entry) {
        Ok(()) => return,
        Err(entry) => {
          if let Some(entry) = self.access_log.pop() {
            if (&*entry.1).fetch_sub(1).eq(&1) {
              self.push_recycling_bin(entry);
            }
          }

          entry
        }
      };
    }
  }

  fn push_recycling_bin(&self, mut entry: (K, Arc<AtomicCell<usize>>)) {
    loop {
      entry = match self.recycling_bin.push(entry) {
        Ok(()) => return,

        Err(entry) => {
          if let Some((key, lru_counter)) = self.recycling_bin.pop() {
            if lru_counter.as_ref().load().eq(&0) {
              self
                .pending_ops
                .push(EventualOp::GarbageCollect(key, lru_counter));
            }
          }

          entry
        }
      }
    }
  }

  fn apply_pending_ops(&self, write_handle: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>) {
    let read_handle = write_handle.clone();

    while let Some(op) = self.pending_ops.pop() {
      self.apply_op(op, &read_handle, write_handle);
    }
  }

  fn apply_op(
    &self,
    op: EventualOp<K, V>,
    read_handle: &ReadHandle<K, ValueBox<V>>,
    write_handle: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>,
  ) {
    match op {
      EventualOp::SetValue(key, value) => self.set(key, value.into(), write_handle),
      EventualOp::Extend(data) => {
        for (key, value) in data {
          self.set(key, value.into(), write_handle);
        }
      }
      EventualOp::InvalidateKey(key) => {
        write_handle.clear(key);
      }
      EventualOp::GarbageCollect(key, lru_counter) => {
        if lru_counter.as_ref().load().eq(&0) {
          if let Some(container) = read_handle.get_one(&key) {
            if Arc::ptr_eq(&container.lru_counter, &lru_counter) {
              write_handle.clear(key);
            }
          }
        }
      }
      EventualOp::PurgeCache => {
        while self.access_log.pop().is_some() {}

        write_handle.purge();
      }
    }
  }

  fn is_pending_empty(&self) -> bool {
    self.pending_ops.is_empty()
  }
}

/// An eventually consistent LRU designed for lock-free concurrent reads. This is `!Sync` but can be cloned and used thread local
#[derive(Clone)]
pub struct EVLRU<K: Key, V: Value> {
  reader: ReadHandle<K, ValueBox<V>>,
  writer: Arc<EventualWriter<K, V>>,
}

impl<K, V> EVLRU<K, V>
where
  K: Key,
  V: Value,
{
  /// Create a new ELVRU instance with a flex-capacity.
  pub fn new(flex_capacity: usize) -> Self {
    let (reader, write_handle) = evmap::new();
    let writer = EventualWriter::new(write_handle, flex_capacity);

    EVLRU { reader, writer }
  }

  /// Get current key value as applied and appends a new access counter to the access log. At capacity, the last used access counter is dropped and should that be the last access counter of it's key/value pair, this slow will be evicted during the next apply cycle should no reads/write occurr in the interim.
  pub fn get(&self, key: K) -> Option<Arc<V>> {
    if let Some(container) = self.reader.get_one(&key).map(|guard| guard.clone()) {
      self.writer.push_access_entry(key, &container);

      Some(container.value)
    } else {
      None
    }
  }

  /// Get current key value as applied without updating the access log
  pub fn peek(&self, key: K) -> Option<Arc<V>> {
    if let Some(container) = self.reader.get_one(&key).map(|guard| guard.clone()) {
      Some(container.value)
    } else {
      None
    }
  }

  /// Returns a bool indicating whether the given key is in the cache as currently applied and without updating the access log
  pub fn contains(&self, key: &K) -> bool {
    self.reader.contains_key(key)
  }

  /// Set value on next apply cycle
  pub fn set(&self, key: K, value: Arc<V>) {
    self
      .writer
      .pending_ops
      .push(EventualOp::SetValue(key, value));
  }

  /// Extend data on next apply cycle
  pub fn extend<I: IntoIterator<Item = (K, Arc<V>)>>(&self, iter: I) {
    let data: HashMap<K, Arc<V>> = HashMap::from_iter(iter);

    self.writer.pending_ops.push(EventualOp::Extend(data));
  }

  /// Mark key to be invalidated on next apply cycle
  pub fn invalidate_key(&self, key: K) {
    self.writer.pending_ops.push(EventualOp::InvalidateKey(key));
  }

  /// Clear pending operations and mark cache to be purged on next apply cycle
  pub fn purge(&self) {
    while self.writer.pending_ops.pop().is_some() {}
    self.writer.pending_ops.push(EventualOp::PurgeCache);
  }

  /// Block to acquire write lock and apply all pending changes. It is preferrable to use [`EVLRU::try_apply_blocking`] or [`EVLRU::background_apply_changes`]instead whenever possible because this usage only blocks to apply changes when necessary and otherwise delegates the responsibility to apply changes to the current lock holder.
  pub fn apply_blocking(&self) {
    let mut write_handle = self.writer.write_handle.lock().unwrap();
    self.writer.apply_pending_ops(&mut write_handle);
    write_handle.refresh();
  }

  /// Delegate or directly apply all pending changes in cycles until no more pending ops are present or another writer lock holder takes over between cycles. The ensures all pending work is eventually processed either directly or indirectly and without blocking ever for lock acquisition.
  pub fn try_apply_blocking(&self) {
    while let Ok(mut write_handle) = self.writer.write_handle.try_lock() {
      self.writer.apply_pending_ops(&mut write_handle);
      write_handle.refresh();
      drop(write_handle);

      if self.writer.pending_ops.is_empty() {
        break;
      }
    }
  }

  /// Cooperatively apply pending changes in cycles using Tokio's blocking thread pool. Each cycle the responsibility is delegated via lock acquisition to apply pending ops and then to flush updates to readers and this repeats until it is guaranteed that no pending ops are left unprocessed.
  pub fn background_apply_changes(&self) {
    let writer = self.writer.clone();

    if !writer.is_pending_empty() {
      rayon::spawn(move || {
        // If there is already a lock holder the responsibility of handling pending ops is delegated
        while let Ok(mut write_handle) = writer.write_handle.try_lock() {
          writer.apply_pending_ops(&mut write_handle);
          write_handle.refresh();
          drop(write_handle);

          // This ensures there's exactly one writer driving all pending ops to completion and so that ops pushed while flushing are delegated and not left unprocessed
          if writer.pending_ops.is_empty() {
            break;
          }
        }
      });
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::EVLRU;

  #[test]
  fn it_expires_oldest() {
    let cache: EVLRU<&'static str, &'static str> = EVLRU::new(2);

    cache.set("apple", "red".into());
    cache.set("banana", "yellow".into());

    cache.apply_blocking();

    assert!(cache.get("apple").is_some());
    assert!(cache.get("banana").is_some());

    cache.set("pear", "green".into());
    cache.set("peach", "orange".into());
    cache.set("coconut", "brown".into());

    cache.apply_blocking();

    assert!(cache.get("banana").is_some());
    assert!(cache.get("pear").is_some());
    assert!(cache.get("apple").is_none());
  }

  #[test]
  fn it_purges() {
    let cache: EVLRU<&'static str, &'static str> = EVLRU::new(2);

    cache.set("apple", "red".into());
    cache.set("banana", "yellow".into());

    cache.apply_blocking();

    assert!(cache.get("apple").is_some());
    assert!(cache.get("banana").is_some());

    cache.purge();
    cache.apply_blocking();

    assert!(cache.get("pear").is_none());
    assert!(cache.get("apple").is_none());
  }

  #[test]
  fn it_invalidates() {
    let cache: EVLRU<&'static str, &'static str> = EVLRU::new(2);

    cache.set("apple", "red".into());

    cache.apply_blocking();

    assert!(cache.get("apple").is_some());

    cache.invalidate_key("apple");

    cache.apply_blocking();

    assert!(cache.get("apple").is_none());
  }
}
