<!--
  ~ Copyright 2026 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Concurrent Vamana: the `Sync` policy and the seqlock protocol

This document describes how `MutableVamanaIndex` supports a lock-free concurrent search path
alongside concurrent inserts, deletes and consolidation. It is aimed at a reviewer of the design
and at whoever next changes `sync_policy.h`, `dynamic_index.h`, `vamana_build.h`,
`consolidate.h` or `svs/core/graph/graph.h`.

## Table of Contents

- [The `Sync` policy](#the-sync-policy)
- [The seqlock protocol](#the-seqlock-protocol)
- [Preconditions: address-stable storage](#preconditions-address-stable-storage)
- [Index-level locks and their order](#index-level-locks-and-their-order)
- [What is not yet safe](#what-is-not-yet-safe)

## The `Sync` policy

`MutableVamanaIndex<Graph, Data, Dist, Sync = SequentialSync>` takes a fourth, defaulted template
parameter that bundles every synchronization seam the index needs behind one name, rather than
exposing each seam as its own template parameter. `SyncPolicy` (`sync_policy.h`) requires:

- `mutex_type` — the index-level mutex (`slot_alloc_mutex_`, `translator_mutex_`,
  `compact_mutex_`; see below).
- `counter_type` — the slot-allocation counter (`PlainCounter` or `AtomicCounter`), constrained
  by the `SyncCounter` concept.
- `graph_access_type` — how the graph reads and writes an adjacency-list header word
  (`graphs::PlainAccess` or `graphs::SeqlockAccess`).
- `growth_type` — the dataset's growth strategy (`data::Reallocating` or `data::SegmentStable`).
- `container_type<T>` — the slot-metadata container template.
- `reserves_pending_slots` — a `bool` selecting whether a reserved slot is visible to search
  before it is fully built (see `SlotMetadata::Pending`).

`SyncPolicyFor<P, Graph>` additionally requires `P::node_visitor_type` to satisfy `NodeVisitor`
for that graph type (see below). Two more members, `vertex_locks_type` and `has_vertex_locks`,
exist on both concrete policies and are used directly by `dynamic_index.h`, but neither is part
of the `SyncPolicy` concept's `requires`-clause — a future policy that omits them fails at the
use site in `dynamic_index.h` rather than at the concept check.

`SequentialSync` sets every member to exactly the type the index used before it was
parameterized: `lib::NullMutex`, `PlainCounter`, `graphs::PlainAccess`, `data::Reallocating`,
`std::vector<T>`, `VisitOnce`, `EmptyLockArray`, `reserves_pending_slots = false`. Because the
default template argument is `SequentialSync`, `MutableVamanaIndex<G, D, Dist>` still means
exactly what it meant before this parameter existed, and every existing call site compiles
unchanged. `static_assert`s in `sync_policy.cpp` pin this type mapping so a later edit cannot
quietly change the default path, and separate asserts confirm `SequentialSync::mutex_type` is
empty (`std::is_empty_v`) so the default path carries no mutex at all.

`SeqlockSync` selects `lib::MovableMutex<std::shared_mutex>`, `AtomicCounter`,
`graphs::SeqlockAccess`, `data::SegmentStable`, `lib::SegmentedVector<lib::AtomicValue<T>>`,
`SeqlockVisitor`, `lib::SegmentedVector<SpinLock>`, and `reserves_pending_slots = true`.

The alternative this design avoids is a second, parallel implementation of the index in a
separate namespace, selected by which types are duplicated rather than by which types a single
class is instantiated over. One bundled policy makes every synchronization decision a single
name at each use site, and the compiler — not a code reviewer — is what proves the sequential
instantiation is unchanged.

## The seqlock protocol

Each graph node carries a `SeqLockCounter` (`svs/lib/concurrency/seqlock.h`): a `uint32_t`,
odd while a write is in progress, even otherwise. A writer calls `begin_write()` (store `seq+1`,
odd) before mutating the node's adjacency list and `end_write(seq)` (store `seq+2`, even) after.
A reader calls `read_begin()`, which returns the current value if even or `std::nullopt` if a
write is in progress, and later `read_validate(seq)`, which returns whether the counter is still
exactly `seq`.

**A writer must acquire this counter's guard for every mutation of a node's adjacency list, or
the write is invisible to every reader that validates against it.** The counter only changes
inside `begin_write`/`end_write`; a write that bypasses them leaves the counter untouched, so
`read_validate` reports success — the reader believes it validated a consistent read when it
actually observed a partially written, arbitrary mixture of old and new neighbour ids. This is
not merely a race with a chance of being caught; it is invisible to the one mechanism that exists
to catch it.

`SimpleGraphBase<Idx, Data, Access>::write_guard(i)` (`svs/core/graph/graph.h`) is the RAII type
that maintains this invariant: under `SeqlockAccess` it takes the per-node `SpinLock` in
`access_state_.node_locks[i]` and calls `begin_write()` on construction, then calls `end_write()`
and releases the lock on destruction; under `PlainAccess` it compiles to a no-op. Writer-writer
exclusion on the same node is the caller's responsibility via that per-node lock — the counter
alone only orders a single writer against concurrent readers.

On the read side, `SeqlockVisitor` (`svs/index/vamana/greedy_search.h`) is the `NodeVisitor`
`greedy_search` invokes at each expanded node under `SeqlockSync` (the default, `VisitOnce`, just
calls the body once with the graph's own neighbour range). `SeqlockVisitor` copies the adjacency
list into a thread-local scratch buffer, calls `read_validate`, and only then invokes the body
with the validated copy — validation has to precede the body, because a body that has already
inserted a torn neighbour id into the search buffer cannot be undone by a later retry. On
validation failure it retries with a bounded backoff (spin, then yield, then throw
`lib::ANNException` after 1000 yielded rounds without success) rather than spinning indefinitely
against a writer that may itself be waiting.

## Preconditions: address-stable storage

Every lock-free reader — `SeqlockVisitor`, and `graph_access_type::load` more generally —
dereferences storage without holding a lock, so growing that storage must never relocate an
element a reader might be reading. There are **three** independent sites where this has to hold:

1. **The dataset.** Enforced: `MutableVamanaIndex` has an in-class `static_assert`
   (`dynamic_index.h:140-145`) that a `Sync::growth_type` of `data::SegmentStable` requires
   `data::is_dataset_grow_stable_v<Data>`, with the message "SeqlockSync requires a dataset with
   address-stable growth (Blocked<Alloc, SegmentStable>)". `is_dataset_grow_stable_v` is
   specialized for `SimpleData` and (separately) for `SQDataset`, forwarding to whether the
   dataset's allocator is `Blocked<Alloc, SegmentStable>`.
2. **The graph's own adjacency storage.** `SimpleGraphBase`'s backing store is itself a
   `data::MemoryDataset`, so the same `is_dataset_grow_stable_v` trait applies to it — but that
   trait is only checked in test code (`tests/svs/concurrent/concurrency.cpp`,
   `tests/svs/index/vamana/sync_policy.cpp`), never inside `SimpleGraphBase` or
   `MutableVamanaIndex` itself. A caller can instantiate `SeqlockSync` over a graph whose
   adjacency storage reallocates and get no compile-time diagnostic.
3. **The seqlock counters.** `SeqLockArray` (`seqlock.h`) always stores its per-node counters in
   a `lib::SegmentedVector`, unconditionally — there is no allocator template parameter here to
   get wrong, so this site's stability is structural rather than policy-selected, and there is
   nothing to assert.

`growth_type` is the member that *names* the requirement; only site 1 is actually enforced by the
type system. This asymmetry is worth keeping in mind when adding a new grow-stable-sensitive
container: the pattern to follow is the dataset's `static_assert`, not the graph's silence.

## Index-level locks and their order

`MutableVamanaIndex` declares three `[[no_unique_address]]` mutexes of type
`Sync::mutex_type` — `slot_alloc_mutex_`, `translator_mutex_`, `compact_mutex_` — with a
documented order given verbatim in `dynamic_index.h`:

```cpp
// Lock order: compact_mutex_ → slot_alloc_mutex_ and compact_mutex_ →
// translator_mutex_. Violating this order causes deadlock; release compact_mutex_
// before acquiring others.
```

`Sync::mutex_type` is `std::shared_mutex` under `SeqlockSync`, which is not recursive: acquiring
it exclusively and then, on the same thread, shared again is a hard deadlock, and acquiring it
shared recursively can deadlock behind a writer that queued in between the two acquisitions. The
lock order above is what keeps every acquisition path acyclic.

`compact_mutex_` has exactly one exclusive (`std::unique_lock`) acquisition in the whole class,
inside `compact()`; every other acquisition anywhere in the class — the single-query and batch
`search()` overloads, `exhaustive_search()`, and the shared-lock accessors `lock_for_search()`
and `lock_for_translation()` that `BatchIterator` uses — takes it with `std::shared_lock`. **A
shared `compact_mutex_` lock on the search path therefore excludes `compact()` and nothing
else.** In particular:

- `add_points()` takes `slot_alloc_mutex_` and `translator_mutex_` only. It never touches
  `compact_mutex_`, deliberately — concurrent insert during search is the capability this design
  exists to deliver, and excluding it from `compact_mutex_` is what lets it proceed while a
  search holds a shared lock.
- `consolidate()` takes no index-level mutex at all. It is made safe to run alongside search
  purely by the per-node `write_guard` inside the free `consolidate()` function it calls, not by
  any lock declared on `MutableVamanaIndex`.

## What is not yet safe

**`compact()`'s own graph write is unguarded.** `compact()` calls
`graph_.replace_node(new_id, ...)` while holding an exclusive `compact_mutex_`, but it never
calls `graph_.write_guard(new_id)`. That lock excludes `compact()` from concurrent *search*, but
— as noted above — `add_points()` and `consolidate()` never take `compact_mutex_` at all, so
nothing excludes either of them from running at the same time as `compact()`. This is an
unenforced precondition rather than a live defect in the delivered capability: `compact()`
shrinks and renumbers storage, so it cannot be concurrent with anything by construction, and the
concurrency test suite's own comment contrasts `consolidate()` — "allowed to run while search
continues" — with `compact()`, which is not exercised concurrently anywhere in that suite. Still,
nothing states the precondition: there is no assertion and no doc comment on `compact()` itself,
and every public wrapper around it is a plain pass-through.

**`consolidate()`'s graph write is unguarded too.** `GraphConsolidator::apply_updates`
(`consolidate.h`) calls `graph_.replace_node(...)` to install the rewired adjacency list for a
node, with no `write_guard` anywhere in that file — unlike `VamanaBuilder`, which takes
`graph_.write_guard(node_id)` immediately before each of its own `replace_node` calls. The
consequence is the one stated above: a concurrent reader's `read_validate` cannot detect this
write, and two unguarded writers to the same node are not mutually excluded by anything. Fixing
this means adding the same guard `VamanaBuilder` already uses, taken per node around the existing
update loop so nothing that can block is held across it; it is not yet done here.
