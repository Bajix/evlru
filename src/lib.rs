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
  Extend(Box<HashMap<K, Arc<V>>>),
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
    let pending_ops = SegQueue::new();

    Arc::new(EventualWriter {
      write_handle,
      access_log,
      pending_ops,
    })
  }

  fn set_value(
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

  fn apply_pending_ops(&self, mut write_handle: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>) {
    let read_handle = write_handle.clone();
    loop {
      match self.pending_ops.pop() {
        Some(op) => self.apply_op(op, &read_handle, &mut write_handle),
        None => break,
      };
    }
  }

  fn apply_op(
    &self,
    op: EventualOp<K, V>,
    read_handle: &ReadHandle<K, ValueBox<V>>,
    mut write_handle: &mut MutexGuard<WriteHandle<K, ValueBox<V>>>,
  ) {
    match op {
      EventualOp::SetValue(key, value) => self.set_value(key, value.into(), &mut write_handle),
      EventualOp::Extend(data) => {
        for (key, value) in *data {
          self.set_value(key, value.into(), &mut write_handle);
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
        loop {
          if self.access_log.pop().is_none() {
            break;
          }
        }

        write_handle.purge();
      }
    }
  }

  fn is_pending_empty(&self) -> bool {
    self.pending_ops.is_empty()
  }
}

/// An eventually consistent LRU designed for lock-free concurrent reads
pub struct EVLRU<K: Key, V: Value> {
  reader: ReadHandle<K, ValueBox<V>>,
  writer: Arc<EventualWriter<K, V>>,
}

impl<K, V> EVLRU<K, V>
where
  K: Key,
  V: Value,
{
  /// Create a new ELVRU instance with bounded access log capacity
  pub fn new(capacity: usize) -> Self {
    let (reader, write_handle) = evmap::new();
    let writer = EventualWriter::new(write_handle, capacity);

    EVLRU { reader, writer }
  }

  /// Get current key value as applied and push a new key access counter to the LRU access log after incrementing. At capacity the last accessed entry is popped and should that be the last access counter for this key/value pair this will be marked for removal the next time changes are applied should no reads/write have since occured.
  pub fn get_value(&self, key: K) -> Option<Arc<V>> {
    if let Some(container) = self.reader.get_one(&key).map(|guard| guard.clone()) {
      self.writer.push_access_entry(key, &container);

      Some(container.value)
    } else {
      None
    }
  }

  /// Get current key value as applied without updating the LRU access log
  pub fn peek(&self, key: K) -> Option<Arc<V>> {
    if let Some(container) = self.reader.get_one(&key).map(|guard| guard.clone()) {
      Some(container.value)
    } else {
      None
    }
  }

  /// Set value on next apply cycle
  pub fn set_value(&self, key: K, value: Arc<V>) {
    self
      .writer
      .pending_ops
      .push(EventualOp::SetValue(key, value));
  }

  /// Extend data on next apply cycle
  pub fn extend<T: IntoIterator<Item = (K, Arc<V>)>>(&self, data: T) {
    let data: HashMap<K, Arc<V>> = HashMap::from_iter(data);
    self.writer.pending_ops.push(EventualOp::Extend(Box::from(data)));
  }

  /// Mark key to be invalidated on next apply cycle
  pub fn invalidate_key(&self, key: K) {
    self.writer.pending_ops.push(EventualOp::InvalidateKey(key));
  }

  /// Clear pending operations and mark cache to be purged on next apply cycle
  pub fn purge(&self) {
    loop {
      if self.writer.pending_ops.pop().is_none() {
        break;
      }
    }

    self.writer.pending_ops.push(EventualOp::PurgeCache);
  }

  /// Block to acquire writer mutex and apply all pending changes. It is preferrable to use [`EVLRU::try_apply_blocking`] instead whenever possible because this usage only blocks to apply changes when necessary and otherwise delegates the responsibility to apply changes to the current lock holder.
  pub fn apply_blocking(&self) {
    let mut write_handle = self.writer.write_handle.lock().unwrap();
    self.writer.apply_pending_ops(&mut write_handle);
    write_handle.flush();
  }

  /// Delegate or directly apply all pending changes in cycles until no more pending ops are present or another writer lock holder takes over between cycles. The ensures all pending work is eventually processed either directly or indirectly and without blocking ever for lock acquisition.
  pub fn try_apply_blocking(&self) {
    while let Ok(mut write_handle) = self.writer.write_handle.try_lock() {
      self.writer.apply_pending_ops(&mut write_handle);
      write_handle.flush();
      drop(write_handle);

      if self.writer.pending_ops.is_empty() {
        break;
      }
    }
  }

  /// Apply all pending changes in cycles using Tokio's blocking thread pool
  pub fn background_apply_changes(&self) {
    let writer = self.writer.clone();

    if !self.writer.is_pending_empty() {
      tokio::task::spawn(async move {
        tokio::task::spawn_blocking(move || {
          while let Ok(mut write_handle) = writer.write_handle.try_lock() {
            writer.apply_pending_ops(&mut write_handle);
            write_handle.flush();
            drop(write_handle);

            if writer.pending_ops.is_empty() {
              break;
            }
          }
        })
        .await
        .unwrap();
      });
    }
  }
}
