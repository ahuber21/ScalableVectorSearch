/*
 * Copyright 2025 Intel Corporation
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

// The core headers must precede the index headers: index templates issue qualified
// calls into `distance` that bind their candidate set at definition context.
#include "svs/core/data/simple.h"
#include "svs/core/graph/graph.h"
#include "svs/lib/null_mutex.h"
#include "svs/lib/segmented_vector.h"

#include "svs/index/vamana/greedy_search.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <vector>

namespace svs::index::vamana {

/// @brief Counter with no atomicity, for single-threaded use.
class PlainCounter {
  public:
    PlainCounter() = default;
    explicit PlainCounter(size_t value)
        : value_{value} {}

    size_t load() const { return value_; }
    void store(size_t value) { value_ = value; }
    void fetch_max(size_t value) { value_ = std::max(value_, value); }
    size_t fetch_add(size_t value) {
        size_t previous = value_;
        value_ += value;
        return previous;
    }

  private:
    size_t value_ = 0;
};

/// @brief Counter with acquire/release atomic access.
class AtomicCounter {
  public:
    AtomicCounter() = default;
    explicit AtomicCounter(size_t value)
        : value_{value} {}

    // std::atomic is neither copyable nor movable, but the owning index must remain both.
    // These reload the value non-atomically and are unsafe under concurrent access.
    AtomicCounter(const AtomicCounter& other)
        : value_{other.load()} {}
    AtomicCounter& operator=(const AtomicCounter& other) {
        store(other.load());
        return *this;
    }
    AtomicCounter(AtomicCounter&& other) noexcept
        : value_{other.load()} {}
    AtomicCounter& operator=(AtomicCounter&& other) noexcept {
        store(other.load());
        return *this;
    }
    ~AtomicCounter() = default;

    size_t load() const { return value_.load(std::memory_order_acquire); }
    void store(size_t value) { value_.store(value, std::memory_order_release); }

    void fetch_max(size_t value) {
        size_t current = value_.load(std::memory_order_acquire);
        while (current < value &&
               !value_.compare_exchange_weak(
                   current, value, std::memory_order_acq_rel, std::memory_order_acquire
               )) {}
    }

    size_t fetch_add(size_t value) {
        return value_.fetch_add(value, std::memory_order_acq_rel);
    }

  private:
    std::atomic<size_t> value_{0};
};

/// @brief Counter usable for the index's slot bookkeeping.
template <typename C>
concept SyncCounter =
    std::default_initializable<C> && requires(C& counter) {
                                         {
                                             std::as_const(counter).load()
                                             } -> std::convertible_to<size_t>;
                                         counter.store(size_t{});
                                         counter.fetch_max(size_t{});
                                         {
                                             counter.fetch_add(size_t{})
                                             } -> std::convertible_to<size_t>;
                                     };

/// @brief Bundle of the synchronization seams used by the mutable Vamana index.
///
/// A policy supplies the mutex, counter, graph access, node visitor and growth types the
/// index composes, plus flags selecting the slot-lifetime and translator protocols.
template <typename P>
concept SyncPolicy = requires {
                         typename P::mutex_type;
                         typename P::counter_type;
                         typename P::graph_access_type;
                         typename P::node_visitor_type;
                         typename P::growth_type;
                         typename P::template container_type<int>;
                         requires SyncCounter<typename P::counter_type>;
                         requires NodeVisitor<typename P::node_visitor_type>;
                         { P::reserves_pending_slots } -> std::convertible_to<bool>;
                         { P::defers_translator_cleanup } -> std::convertible_to<bool>;
                         { P::supplements_search_buffer } -> std::convertible_to<bool>;
                     };

/// @brief Synchronization policy for single-threaded use.
///
/// Every seam is the identity: the mutexes compile away, the counters are plain values and
/// the containers and access paths are the ones the index used before it was parameterized.
struct SequentialSync {
    using mutex_type = lib::NullMutex;
    using counter_type = PlainCounter;
    using graph_access_type = graphs::PlainAccess;
    using node_visitor_type = VisitOnce;
    using growth_type = data::Reallocating;

    template <typename T> using container_type = std::vector<T>;

    /// Reserved slots are immediately visible to search; there is no Pending state.
    static constexpr bool reserves_pending_slots = false;
    /// Translator entries are erased by `delete_entries` rather than by `consolidate`.
    static constexpr bool defers_translator_cleanup = false;
    /// Search results never need topping up from the translator.
    static constexpr bool supplements_search_buffer = false;
};

static_assert(SyncCounter<PlainCounter>);
static_assert(SyncCounter<AtomicCounter>);
static_assert(SyncPolicy<SequentialSync>);

} // namespace svs::index::vamana
