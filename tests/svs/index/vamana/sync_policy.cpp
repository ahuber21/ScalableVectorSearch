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

#include "svs/index/vamana/sync_policy.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
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

// The whole point of NullMutex is that a member costs nothing.
struct WithNullMutex {
    [[no_unique_address]] svs::lib::NullMutex mutex_{};
    size_t value_ = 0;
};
static_assert(sizeof(WithNullMutex) == sizeof(size_t));

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
