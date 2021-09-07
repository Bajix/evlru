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

pub trait Key: Send + Sync + Hash + Eq + Clone {}
impl<T: Send + Sync + Hash + Eq + Clone> Key for T {}

pub trait Value: Send + Sync + Clone + Eq + Hash {}
impl<T: Send + Sync + Clone + Eq + Hash> Value for T {}

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

/// An eventually consistent LRU designed for lock-free concurrent reads
pub struct EVLRU<K: Key, V: Value> {
  reader: ReadHandle<K, ValueBox<V>>,
  writer: Mutex<WriteHandle<K, ValueBox<V>>>,
  access_log: ArrayQueue<(K, Arc<AtomicCell<usize>>)>,
  pending_ops: SegQueue<EventualOp<K, V>>,
}

impl<K, V> EVLRU<K, V>
where
  K: Key,
  V: Value,
{
  /// Create a new ELVRU instance with bounded access log capacity
  pub fn new(capacity: usize) -> Self {
    let (reader, writer) = evmap::new();
    let writer = Mutex::from(writer);
    let access_log = ArrayQueue::new(capacity);
    let pending_ops = SegQueue::new();

    EVLRU {
      reader,
      writer,
      access_log,
      pending_ops,
    }
  }

  /// Get current key value as applied and push a new key access counter to the LRU access log after incrementing. At capacity the last accessed entry is popped and should that be the last access counter for this key/value pair this will be marked for removal the next time changes are applied should no reads/write have since occured.
  pub fn get_value(&self, key: K) -> Option<Arc<V>> {
    if let Some(container) = self.reader.get_one(&key).map(|guard| guard.clone()) {
      self.push_access_entry(key, &container);

      Some(container.value)
    } else {
      None
    }
  }

  /// Set value on next apply cycle
  pub fn set_value(&self, key: K, value: Arc<V>) {
    self.pending_ops.push(EventualOp::SetValue(key, value));
  }

  /// Extend data on next apply cycle
  pub fn extend(&self, data: HashMap<K, Arc<V>>) {
    self.pending_ops.push(EventualOp::Extend(data))
  }

  /// Mark key to be invalidated on next apply cycle
  pub fn invalidate_key(&self, key: K) {
    self.pending_ops.push(EventualOp::InvalidateKey(key));
  }

  /// Clear pending operations and mark cache to be purged on next apply cycle
  pub fn purge(&self) {
    loop {
      if self.pending_ops.pop().is_none() {
        break;
      }
    }

    self.pending_ops.push(EventualOp::PurgeCache);
  }

  fn update_value(
    &self,
    key: K,
    value: ValueBox<V>,
    writer: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>,
  ) {
    self.push_access_entry(key.clone(), &value);
    writer.update(key, value);
  }

  fn push_access_entry(&self, key: K, value: &ValueBox<V>) {
    let lru_counter = value.lru_counter.to_owned();
    (&*lru_counter).fetch_add(1);

    let mut entry = (key, lru_counter);

    loop {
      entry = match self.access_log.push(entry) {
        Ok(()) => return,
        Err(entry) => {
          if let Some((key, lru_counter)) = self.access_log.pop() {
            if (&*lru_counter).fetch_sub(1).eq(&0) {
              self
                .pending_ops
                .push(EventualOp::GarbageCollect(key, lru_counter))
            }
          }

          entry
        }
      };
    }
  }

  fn apply_pending_ops(&self, mut writer: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>) {
    loop {
      match self.pending_ops.pop() {
        Some(EventualOp::SetValue(key, value)) => self.update_value(key, value.into(), &mut writer),
        Some(EventualOp::Extend(data)) => {
          for (key, value) in data {
            self.update_value(key, value.into(), &mut writer);
          }
        }
        Some(EventualOp::InvalidateKey(key)) => {
          writer.clear(key);
        }
        Some(EventualOp::GarbageCollect(key, lru_counter)) => {
          if lru_counter.as_ref().load().eq(&0) {
            if let Some(container) = self.reader.get_one(&key) {
              if Arc::ptr_eq(&container.lru_counter, &lru_counter) {
                writer.clear(key);
              }
            }
          }
        }
        Some(EventualOp::PurgeCache) => {
          loop {
            if self.access_log.pop().is_none() {
              break;
            }
          }

          writer.purge();
        }
        None => break,
      };
    }
  }

  /// Block to acquire writer mutex and apply all pending changes. It is preferrable to use [`EVLRU::try_apply_blocking`] instead whenever possible because this usage only blocks to apply changes when necessary and otherwise delegates the responsibility to apply changes to the current lock holder.
  pub fn apply_blocking(&self) {
    let mut writer = self.writer.lock().unwrap();
    self.apply_pending_ops(&mut writer);
    writer.flush();
  }

  /// Delegate or directly apply all pending changes in cycles until no more pending ops are present or another writer lock holder takes over between cycles. The ensures all pending work is eventually processed either directly or indirectly and without blocking ever for lock acquisition.
  pub fn try_apply_blocking(&self) {
    while let Ok(mut writer) = self.writer.try_lock() {
      self.apply_pending_ops(&mut writer);
      writer.flush();
      drop(writer);

      if self.pending_ops.is_empty() {
        break;
      }
    }
  }
}
