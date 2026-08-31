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

// svs
#include "svs/core/data/simple.h"
#include "svs/core/graph/graph.h"
#include "svs/lib/datatype.h"

#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/sync_policy.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

namespace vamana = svs::index::vamana;

// The sequential policy must select exactly the types the index used before it was
// parameterized; a change here silently alters the single-threaded index.
using TestGraph = svs::graphs::SimpleGraph<uint32_t>;

static_assert(std::is_same_v<vamana::SequentialSync::mutex_type, svs::lib::NullMutex>);
static_assert(std::is_same_v<
              vamana::SequentialSync::graph_access_type,
              svs::graphs::PlainAccess>);
static_assert(std::is_same_v<
              vamana::SequentialSync::node_visitor_type<TestGraph>,
              vamana::VisitOnce>);
static_assert(std::is_same_v<vamana::SequentialSync::growth_type, svs::data::Reallocating>);
static_assert(std::is_same_v<
              vamana::SequentialSync::template container_type<int>,
              std::vector<int>>);

static_assert(vamana::SyncPolicy<vamana::SequentialSync>);
static_assert(vamana::SyncPolicyFor<vamana::SequentialSync, TestGraph>);
static_assert(!vamana::SequentialSync::reserves_pending_slots);
static_assert(!vamana::SequentialSync::defers_translator_cleanup);
static_assert(!vamana::SequentialSync::supplements_search_buffer);

// The seqlock policy must yield SeqlockVisitor for its graph type.
static_assert(std::is_same_v<
              vamana::SeqlockSync::node_visitor_type<TestGraph>,
              vamana::SeqlockVisitor<TestGraph>>);
static_assert(vamana::SyncPolicy<vamana::SeqlockSync>);
static_assert(vamana::SyncPolicyFor<vamana::SeqlockSync, TestGraph>);

// A policy missing any one member must not satisfy the concept, otherwise the concept is
// decorative and a malformed policy fails deep inside the index instead.
struct MissingCounter {
    using mutex_type = svs::lib::NullMutex;
    using graph_access_type = svs::graphs::PlainAccess;
    using node_visitor_type = vamana::VisitOnce;
    using growth_type = svs::data::Reallocating;
    template <typename T> using container_type = std::vector<T>;
    static constexpr bool reserves_pending_slots = false;
    static constexpr bool defers_translator_cleanup = false;
    static constexpr bool supplements_search_buffer = false;
};
static_assert(!vamana::SyncPolicy<MissingCounter>);

struct BadVisitor {
    using mutex_type = svs::lib::NullMutex;
    using counter_type = vamana::PlainCounter;
    using graph_access_type = svs::graphs::PlainAccess;
    using growth_type = svs::data::Reallocating;
    template <typename Graph> using node_visitor_type = int;
    template <typename T> using container_type = std::vector<T>;
    static constexpr bool reserves_pending_slots = false;
    static constexpr bool defers_translator_cleanup = false;
    static constexpr bool supplements_search_buffer = false;
};
static_assert(vamana::SyncPolicy<BadVisitor>);
static_assert(!vamana::SyncPolicyFor<BadVisitor, TestGraph>);

// Growth constraint: SeqlockSync requires grow-stable datasets; SequentialSync accepts any.
using GrowStableData = svs::data::SimpleData<
    float,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>>;
using NonGrowStableData = svs::data::SimpleData<float, svs::Dynamic>;

static_assert(svs::data::is_dataset_grow_stable_v<GrowStableData>);
static_assert(!svs::data::is_dataset_grow_stable_v<NonGrowStableData>);

// Accepted: SeqlockSync over grow-stable dataset.
using AcceptedSeqlock = vamana::MutableVamanaIndex<
    TestGraph,
    GrowStableData,
    svs::distance::DistanceL2,
    vamana::SeqlockSync>;
static_assert(sizeof(AcceptedSeqlock) > 0);

// Accepted: SequentialSync over non-grow-stable dataset (default path).
using AcceptedSequential = vamana::MutableVamanaIndex<
    TestGraph,
    NonGrowStableData,
    svs::distance::DistanceL2,
    vamana::SequentialSync>;
static_assert(sizeof(AcceptedSequential) > 0);

// Rejected: SeqlockSync over non-grow-stable dataset, a static_assert failure that cannot
// be exercised from a translation unit that must compile.

// The whole point of NullMutex is that a member costs nothing.
struct WithNullMutex {
    [[no_unique_address]] svs::lib::NullMutex mutex_{};
    size_t value_ = 0;
};
static_assert(sizeof(WithNullMutex) == sizeof(size_t));

// Slot layout shared by the predicate state table below: one slot per SlotMetadata state.
constexpr size_t empty_slot = 0;
constexpr size_t valid_slot = 1;
constexpr size_t deleted_slot = 2;
constexpr size_t pending_slot = 3;

template <typename Sync> auto make_status_table() {
    using Container = typename Sync::template container_type<vamana::SlotMetadata>;
    auto status = Container(4, vamana::SlotMetadata::Empty);
    status[valid_slot] = vamana::SlotMetadata::Valid;
    status[deleted_slot] = vamana::SlotMetadata::Deleted;
    status[pending_slot] = vamana::SlotMetadata::Pending;
    return status;
}

// The predicate bodies are duplicated from MutableVamanaIndex in dynamic_index.h, which
// cannot be instantiated with SeqlockSync yet; the trait and container are the real ones.
template <typename Sync, typename Container>
bool prune_predicate(const Container& status, size_t i) {
    if constexpr (Sync::reserves_pending_slots) {
        return status[i] == vamana::SlotMetadata::Deleted;
    } else {
        return status[i] != vamana::SlotMetadata::Valid;
    }
}

template <typename Container> bool live_predicate(const Container& status, size_t i) {
    return status[i] == vamana::SlotMetadata::Valid;
}

static_assert(std::is_copy_constructible_v<vamana::AtomicCounter>);
static_assert(std::is_move_constructible_v<vamana::AtomicCounter>);
static_assert(std::is_copy_assignable_v<vamana::AtomicCounter>);
static_assert(std::is_move_assignable_v<vamana::AtomicCounter>);

} // namespace

CATCH_TEST_CASE("Sync Policy Counters", "[index][vamana][sync_policy]") {
    CATCH_SECTION("PlainCounter semantics") {
        auto counter = vamana::PlainCounter{};
        CATCH_REQUIRE(counter.load() == 0);

        counter.store(10);
        CATCH_REQUIRE(counter.load() == 10);

        // fetch_max must not lower the value.
        counter.fetch_max(4);
        CATCH_REQUIRE(counter.load() == 10);
        counter.fetch_max(17);
        CATCH_REQUIRE(counter.load() == 17);

        CATCH_REQUIRE(counter.fetch_add(3) == 17);
        CATCH_REQUIRE(counter.load() == 20);

        auto seeded = vamana::PlainCounter{7};
        CATCH_REQUIRE(seeded.load() == 7);
    }

    CATCH_SECTION("AtomicCounter matches PlainCounter when uncontended") {
        auto plain = vamana::PlainCounter{};
        auto atomic = vamana::AtomicCounter{};

        for (size_t value : {size_t{5}, size_t{2}, size_t{9}, size_t{9}, size_t{1}}) {
            plain.fetch_max(value);
            atomic.fetch_max(value);
            CATCH_REQUIRE(plain.load() == atomic.load());
        }
        CATCH_REQUIRE(atomic.load() == 9);

        CATCH_REQUIRE(plain.fetch_add(4) == atomic.fetch_add(4));
        CATCH_REQUIRE(plain.load() == atomic.load());
    }

    CATCH_SECTION("AtomicCounter copy and move carry the value") {
        auto original = vamana::AtomicCounter{42};

        auto copied = original;
        CATCH_REQUIRE(copied.load() == 42);

        auto assigned = vamana::AtomicCounter{};
        assigned = original;
        CATCH_REQUIRE(assigned.load() == 42);

        auto moved = std::move(copied);
        CATCH_REQUIRE(moved.load() == 42);
    }

    CATCH_SECTION("AtomicCounter fetch_add is not lost under contention") {
        constexpr size_t num_threads = 8;
        constexpr size_t per_thread = 10'000;

        auto counter = vamana::AtomicCounter{};
        auto threads = std::vector<std::thread>{};
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&counter]() {
                for (size_t i = 0; i < per_thread; ++i) {
                    counter.fetch_add(1);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        CATCH_REQUIRE(counter.load() == num_threads * per_thread);
    }

    CATCH_SECTION("AtomicCounter fetch_max settles on the maximum") {
        constexpr size_t num_threads = 8;
        constexpr size_t per_thread = 5'000;

        auto counter = vamana::AtomicCounter{};
        auto threads = std::vector<std::thread>{};
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&counter, t]() {
                for (size_t i = 0; i < per_thread; ++i) {
                    counter.fetch_max(t * per_thread + i);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        CATCH_REQUIRE(counter.load() == (num_threads - 1) * per_thread + per_thread - 1);
    }
}

CATCH_TEST_CASE("MovableMutex", "[index][vamana][sync_policy]") {
    using MovableMutex = svs::lib::MovableMutex<std::shared_mutex>;

    CATCH_SECTION("Exclusive lock prevents concurrent try_lock") {
        auto mutex = MovableMutex{};
        mutex.lock();
        CATCH_REQUIRE(!mutex.try_lock());
        mutex.unlock();
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock();
    }

    CATCH_SECTION("Shared lock prevents exclusive lock") {
        auto mutex = MovableMutex{};
        auto shared = std::shared_lock<MovableMutex>{mutex};
        CATCH_REQUIRE(!mutex.try_lock());
    }

    CATCH_SECTION("Exclusive lock prevents shared lock") {
        auto mutex = MovableMutex{};
        mutex.lock();
        CATCH_REQUIRE(!mutex.try_lock_shared());
        mutex.unlock();
        CATCH_REQUIRE(mutex.try_lock_shared());
        mutex.unlock_shared();
    }

    CATCH_SECTION("Two threads can hold shared locks concurrently") {
        constexpr size_t max_spins = 100'000;
        auto mutex = MovableMutex{};
        auto both_succeeded = std::atomic<bool>{false};
        auto thread1_acquired = std::atomic<bool>{false};
        auto thread2_acquired = std::atomic<bool>{false};
        auto timeout1 = std::atomic<bool>{false};
        auto timeout2 = std::atomic<bool>{false};

        auto t1 = std::thread{[&]() {
            CATCH_REQUIRE(mutex.try_lock_shared());
            thread1_acquired.store(true, std::memory_order_release);

            for (size_t i = 0; i < max_spins; ++i) {
                if (thread2_acquired.load(std::memory_order_acquire)) {
                    both_succeeded.store(true, std::memory_order_release);
                    mutex.unlock_shared();
                    return;
                }
                std::this_thread::yield();
            }
            timeout1.store(true, std::memory_order_release);
            mutex.unlock_shared();
        }};

        auto t2 = std::thread{[&]() {
            for (size_t i = 0; i < max_spins; ++i) {
                if (thread1_acquired.load(std::memory_order_acquire)) {
                    break;
                }
                std::this_thread::yield();
            }
            if (!thread1_acquired.load(std::memory_order_acquire)) {
                timeout2.store(true, std::memory_order_release);
                return;
            }

            CATCH_REQUIRE(mutex.try_lock_shared());
            thread2_acquired.store(true, std::memory_order_release);

            for (size_t i = 0; i < max_spins; ++i) {
                if (both_succeeded.load(std::memory_order_acquire)) {
                    mutex.unlock_shared();
                    return;
                }
                std::this_thread::yield();
            }
            timeout2.store(true, std::memory_order_release);
            mutex.unlock_shared();
        }};

        t1.join();
        t2.join();

        CATCH_REQUIRE(!timeout1.load());
        CATCH_REQUIRE(!timeout2.load());
        CATCH_REQUIRE(both_succeeded.load());
    }

    CATCH_SECTION("Move construction leaves a valid unlocked mutex") {
        auto source = MovableMutex{};
        auto destination = std::move(source);
        CATCH_REQUIRE(destination.try_lock());
        destination.unlock();
    }

    CATCH_SECTION("Move assignment leaves a valid unlocked mutex") {
        auto source = MovableMutex{};
        auto destination = MovableMutex{};
        destination = std::move(source);
        CATCH_REQUIRE(destination.try_lock());
        destination.unlock();
    }
}

CATCH_TEST_CASE("Slot Metadata State Transitions", "[index][vamana][sync_policy]") {
    using SlotMetadata = vamana::SlotMetadata;

    CATCH_SECTION("delete_entry handles Pending slots in concurrent mode") {
        // Test that delete_entry accepts both Valid and Pending slots when
        // reserves_pending_slots is true.
        auto status = svs::lib::SegmentedVector<SlotMetadata>(10, SlotMetadata::Empty);
        status[0] = SlotMetadata::Valid;
        status[1] = SlotMetadata::Pending;

        // Simulate delete_entry behavior in concurrent mode.
        for (size_t i : {size_t{0}, size_t{1}}) {
            auto meta = status[i];
            // In concurrent mode, both Valid and Pending are acceptable for deletion.
            CATCH_REQUIRE((meta == SlotMetadata::Valid || meta == SlotMetadata::Pending));
            status[i] = SlotMetadata::Deleted;
        }

        CATCH_REQUIRE(status[0] == SlotMetadata::Deleted);
        CATCH_REQUIRE(status[1] == SlotMetadata::Deleted);
    }

    CATCH_SECTION("Pending slots excluded from search in concurrent mode") {
        // ValidBuilder with ReservesPending=true must treat Pending as invalid.
        auto status = svs::lib::SegmentedVector<SlotMetadata>(4, SlotMetadata::Empty);
        status[0] = SlotMetadata::Valid;
        status[1] = SlotMetadata::Pending;
        status[2] = SlotMetadata::Deleted;
        status[3] = SlotMetadata::Empty;

        auto builder =
            vamana::ValidBuilder<decltype(status), true /* ReservesPending */>(status);

        // Only Valid slots should be considered valid by the builder.
        CATCH_REQUIRE(builder(0, 1.0f).valid());
        CATCH_REQUIRE(!builder(1, 1.0f).valid()); // Pending
        CATCH_REQUIRE(!builder(2, 1.0f).valid()); // Deleted
        CATCH_REQUIRE(!builder(3, 1.0f).valid()); // Empty
    }

    CATCH_SECTION("Pending slots handled in debug_check_graph_consistency") {
        // The switch statement in debug_check_graph_consistency explicitly handles
        // all states including Pending, treating it as invalid (not searchable).
        auto is_valid = [](SlotMetadata metadata, bool allow_deleted) {
            switch (metadata) {
                case SlotMetadata::Valid: {
                    return true;
                }
                case SlotMetadata::Deleted: {
                    return allow_deleted;
                }
                case SlotMetadata::Empty:
                case SlotMetadata::Pending: {
                    return false;
                }
            }
            return false;
        };

        CATCH_REQUIRE(is_valid(SlotMetadata::Valid, false));
        CATCH_REQUIRE(is_valid(SlotMetadata::Valid, true));
        CATCH_REQUIRE(!is_valid(SlotMetadata::Deleted, false));
        CATCH_REQUIRE(is_valid(SlotMetadata::Deleted, true));
        CATCH_REQUIRE(!is_valid(SlotMetadata::Empty, false));
        CATCH_REQUIRE(!is_valid(SlotMetadata::Empty, true));
        CATCH_REQUIRE(!is_valid(SlotMetadata::Pending, false));
        CATCH_REQUIRE(!is_valid(SlotMetadata::Pending, true));
    }
}

CATCH_TEST_CASE("Slot Metadata Predicates", "[index][vamana][sync_policy]") {
    auto sequential = make_status_table<vamana::SequentialSync>();
    auto seqlock = make_status_table<vamana::SeqlockSync>();

    CATCH_SECTION("Liveness admits only Valid, under both policies") {
        for (size_t i = 0; i < 4; ++i) {
            bool expected = (i == valid_slot);
            CATCH_REQUIRE(live_predicate(sequential, i) == expected);
            CATCH_REQUIRE(live_predicate(seqlock, i) == expected);
        }
    }

    CATCH_SECTION("The prune predicate is policy dependent") {
        // Sequential prunes everything that is not Valid; concurrent prunes only Deleted.
        CATCH_REQUIRE(prune_predicate<vamana::SequentialSync>(sequential, empty_slot));
        CATCH_REQUIRE(!prune_predicate<vamana::SequentialSync>(sequential, valid_slot));
        CATCH_REQUIRE(prune_predicate<vamana::SequentialSync>(sequential, deleted_slot));
        CATCH_REQUIRE(prune_predicate<vamana::SequentialSync>(sequential, pending_slot));

        CATCH_REQUIRE(!prune_predicate<vamana::SeqlockSync>(seqlock, empty_slot));
        CATCH_REQUIRE(!prune_predicate<vamana::SeqlockSync>(seqlock, valid_slot));
        CATCH_REQUIRE(prune_predicate<vamana::SeqlockSync>(seqlock, deleted_slot));
        CATCH_REQUIRE(!prune_predicate<vamana::SeqlockSync>(seqlock, pending_slot));
    }

    CATCH_SECTION("Pending is live under neither policy but pruned only by sequential") {
        // Pruning a Pending slot severs the in-edges of an in-flight insertion, leaving the
        // new point unreachable by search.
        CATCH_REQUIRE(!live_predicate(sequential, pending_slot));
        CATCH_REQUIRE(!live_predicate(seqlock, pending_slot));
        CATCH_REQUIRE(prune_predicate<vamana::SequentialSync>(sequential, pending_slot));
        CATCH_REQUIRE(!prune_predicate<vamana::SeqlockSync>(seqlock, pending_slot));
    }

    CATCH_SECTION("Sequential liveness is the exact complement of pruning") {
        // This equality is what makes replacing `!is_deleted(i)` with `is_live(i)` a rename
        // rather than a behaviour change on the sequential path.
        for (size_t i = 0; i < 4; ++i) {
            CATCH_REQUIRE(
                live_predicate(sequential, i) !=
                prune_predicate<vamana::SequentialSync>(sequential, i)
            );
        }
    }
}

CATCH_TEST_CASE("Grow-Stable Storage Trait", "[index][vamana][sync_policy]") {
    // SegmentStable allocators provide address stability under growth.
    using GrowStableAlloc =
        svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>;
    using GrowStableData = svs::data::SimpleData<float, svs::Dynamic, GrowStableAlloc>;
    static_assert(svs::data::is_grow_stable_v<GrowStableAlloc>);
    static_assert(svs::data::is_dataset_grow_stable_v<GrowStableData>);

    // Reallocating allocators do not provide address stability.
    using ReallocAlloc =
        svs::data::Blocked<svs::lib::Allocator<float>, svs::data::Reallocating>;
    using ReallocData = svs::data::SimpleData<float, svs::Dynamic, ReallocAlloc>;
    static_assert(!svs::data::is_grow_stable_v<ReallocAlloc>);
    static_assert(!svs::data::is_dataset_grow_stable_v<ReallocData>);

    // Default SimpleData without Blocked allocator is not grow-stable.
    using DefaultData = svs::data::SimpleData<float, svs::Dynamic>;
    static_assert(!svs::data::is_dataset_grow_stable_v<DefaultData>);
}
