/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "svs/lib/relocatable_spinlock.h"
#include "svs/lib/segmented_vector.h"

// external
#include "tsl/robin_set.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <mutex>
#include <vector>

namespace svs::lib {

/// @brief Zero-overhead no-op reverse edge store for sequential graphs.
///
/// Provides the same interface as ReverseEdges but compiles to nothing. Used by
/// PlainAccess graphs to avoid runtime cost for a feature they never use.
struct NoReverseEdges {
    void resize(size_t /*new_size*/) const noexcept {}
    void set_recording(bool /*on*/) const noexcept {}
    bool is_recording() const noexcept { return false; }
    template <typename Idx> void record(Idx /*m*/, Idx /*n*/) const noexcept {}
    template <typename Idx> void record_unchecked(Idx /*m*/, Idx /*n*/) const noexcept {}
    template <typename Idx> void remove(Idx /*m*/, Idx /*n*/) const noexcept {}
    template <typename Idx> void remove_unchecked(Idx /*m*/, Idx /*n*/) const noexcept {}
    template <typename Idx> void reset_node(Idx /*n*/) const noexcept {}
    void reset() const noexcept {}
};

static_assert(
    std::is_empty_v<NoReverseEdges>, "NoReverseEdges must be empty for zero overhead"
);

///
/// @brief Per-node index of in-neighbors: `R(n)` is the list of nodes that point at `n`.
///
/// Stored as one `std::vector<Idx>` per node, indexed by node id in a grow-stable
/// `SegmentedVector`, with a per-node `RelocatableSpinLock`. Every operation touches only
/// the target node's list under its own lock.
///
/// `R(n)` is a complete superset of `n`'s in-neighbors: `record` is called unconditionally
/// on every created edge, so it may hold stale (edge later dropped) or duplicate entries,
/// but never misses a live in-edge. Consolidation reads it to find who points at a deleted
/// node, and prunes stale entries via `remove`/`reset_node`.
///
template <std::unsigned_integral Idx> class ReverseEdges {
  public:
    ReverseEdges() = default;
    explicit ReverseEdges(size_t num_nodes)
        : lists_(num_nodes)
        , locks_(num_nodes) {}

    /// Moving while another thread is recording is undefined behavior.
    ReverseEdges(ReverseEdges&& other) noexcept
        : lists_(std::move(other.lists_))
        , locks_(std::move(other.locks_))
        , recording_(other.recording_.load(std::memory_order_acquire)) {}

    /// Moving while another thread is recording is undefined behavior.
    ReverseEdges& operator=(ReverseEdges&& other) noexcept {
        if (this != &other) {
            lists_ = std::move(other.lists_);
            locks_ = std::move(other.locks_);
            recording_.store(
                other.recording_.load(std::memory_order_acquire), std::memory_order_release
            );
        }
        return *this;
    }

    void resize(size_t new_size) {
        lists_.resize(new_size);
        locks_.resize(new_size);
    }

    void set_recording(bool on) { recording_.store(on, std::memory_order_release); }

    bool is_recording() const { return recording_.load(std::memory_order_acquire); }

    void record(Idx m, Idx n) {
        if (!recording_.load(std::memory_order_acquire)) {
            return;
        }
        record_unchecked(m, n);
    }

    void record_unchecked(Idx m, Idx n) {
        std::lock_guard lock{locks_[n]};
        lists_[n].push_back(m);
    }

    void remove(Idx m, Idx n) {
        if (!recording_.load(std::memory_order_acquire)) {
            return;
        }
        remove_unchecked(m, n);
    }

    void remove_unchecked(Idx m, Idx n) {
        std::lock_guard lock{locks_[n]};
        auto& list = lists_[n];
        list.erase(std::remove(list.begin(), list.end(), m), list.end());
    }

    template <typename Deleted>
    void collect(Idx n, tsl::robin_set<size_t>& out, const Deleted& is_deleted) const {
        std::lock_guard lock{locks_[n]};
        for (auto m : lists_[n]) {
            if (!is_deleted(m)) {
                out.insert(m);
            }
        }
    }

    void reset_node(Idx n) {
        std::lock_guard lock{locks_[n]};
        lists_[n].clear();
    }

    void reset() {
        for (size_t i = 0, imax = lists_.size(); i < imax; ++i) {
            lists_[i].clear();
        }
    }

  private:
    lib::SegmentedVector<std::vector<Idx>> lists_;
    mutable lib::SegmentedVector<RelocatableSpinLock> locks_;
    std::atomic<bool> recording_{true};
};

static_assert(std::is_move_constructible_v<ReverseEdges<uint32_t>>);
static_assert(std::is_move_assignable_v<ReverseEdges<uint32_t>>);

} // namespace svs::lib
