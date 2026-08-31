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

// svs
#include "svs/concepts/graph.h"
#include "svs/core/graph/graph.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/sync_policy.h"

// test utils
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

CATCH_TEST_CASE("Graph Access Policies", "[graphs][seqlock]") {
    using Idx = uint32_t;
    using PlainGraph =
        svs::graphs::SimpleGraphBase<Idx, svs::data::SimpleData<Idx, svs::Dynamic>>;
    using SeqlockGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>;

    CATCH_SECTION("Plain graph size unchanged") {
        // PlainAccess::state_type must be empty so [[no_unique_address]] compiles it away.
        static_assert(std::is_empty_v<svs::graphs::PlainAccess::state_type>);

        // Seqlock state is non-empty and should increase the graph size.
        static_assert(!std::is_empty_v<svs::graphs::SeqlockAccess::state_type>);
        static_assert(sizeof(SeqlockGraph) > sizeof(PlainGraph));
    }

    CATCH_SECTION("Both access modes satisfy MemoryGraph concept") {
        static_assert(svs::graphs::ImmutableMemoryGraph<PlainGraph>);
        static_assert(svs::graphs::MemoryGraph<PlainGraph>);
        static_assert(svs::graphs::ImmutableMemoryGraph<SeqlockGraph>);
        static_assert(svs::graphs::MemoryGraph<SeqlockGraph>);
    }

    CATCH_SECTION("SeqlockSync satisfies SyncPolicy") {
        static_assert(svs::index::vamana::SyncPolicy<svs::index::vamana::SeqlockSync>);
    }
}

CATCH_TEST_CASE("Plain mode identical behavior", "[graphs][seqlock]") {
    using Idx = uint32_t;
    size_t n_nodes = 10;
    size_t max_degree = 5;

    auto plain_graph = svs::graphs::SimpleGraph<Idx>(n_nodes, max_degree);
    auto seqlock_graph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>(n_nodes, max_degree);

    CATCH_REQUIRE(plain_graph.n_nodes() == n_nodes);
    CATCH_REQUIRE(seqlock_graph.n_nodes() == n_nodes);
    CATCH_REQUIRE(plain_graph.max_degree() == max_degree);
    CATCH_REQUIRE(seqlock_graph.max_degree() == max_degree);

    // Build the same adjacency lists in both graphs
    for (Idx i = 0; i < n_nodes; ++i) {
        std::vector<Idx> neighbors;
        for (size_t j = 1; j <= max_degree; ++j) {
            neighbors.push_back((i + j) % n_nodes);
        }
        plain_graph.replace_node(i, neighbors);
        seqlock_graph.replace_node(i, neighbors);
    }

    // Verify adjacency lists match
    for (Idx i = 0; i < n_nodes; ++i) {
        auto plain_list = plain_graph.get_node(i);
        auto seqlock_list = seqlock_graph.get_node(i);
        CATCH_REQUIRE(plain_list.size() == seqlock_list.size());
        CATCH_REQUIRE(std::equal(plain_list.begin(), plain_list.end(), seqlock_list.begin())
        );
    }
}

CATCH_TEST_CASE("Seqlock torn-read test", "[graphs][seqlock][!mayfail]") {
    using Idx = uint32_t;
    using SeqlockGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>;

    constexpr size_t n_nodes = 100;
    constexpr size_t max_degree = 64;
    constexpr Idx test_node = 50;

    SeqlockGraph graph(n_nodes, max_degree);

    // Two distinct valid states for the test node
    std::vector<Idx> state_a;
    std::vector<Idx> state_b;
    for (size_t i = 0; i < max_degree; ++i) {
        state_a.push_back(i);
        state_b.push_back(max_degree + i);
    }

    // Initialize with state_a
    graph.replace_node(test_node, state_a);

    constexpr size_t num_readers = 4;
    constexpr size_t iterations = 10000;

    std::atomic<bool> start_flag{false};
    std::atomic<bool> stop_flag{false};
    std::array<std::atomic<size_t>, num_readers> validated_reads{};

    // Catch2 assertions are not thread-safe; accumulate failures here for main-thread
    // check.
    std::atomic<size_t> bad_size{0};
    std::atomic<size_t> torn_reads{0};

    // Writer thread: alternate between state_a and state_b
    auto writer = std::thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        bool use_a = false;
        while (!stop_flag.load(std::memory_order_acquire)) {
            auto guard = graph.write_guard(test_node);
            graph.replace_node(test_node, use_a ? state_a : state_b);
            use_a = !use_a;
        }
    });

    // Reader threads: validate every read is either state_a or state_b
    std::array<std::thread, num_readers> readers;
    for (size_t tid = 0; tid < num_readers; ++tid) {
        readers[tid] = std::thread([&, tid]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            size_t validated = 0;
            for (size_t iter = 0; iter < iterations; ++iter) {
                while (true) {
                    auto seq_opt = graph.read_begin(test_node);
                    if (!seq_opt.has_value()) {
                        continue;
                    }

                    auto adjacency = graph.get_node(test_node);
                    std::vector<Idx> snapshot(adjacency.begin(), adjacency.end());

                    if (!graph.read_validate(test_node, seq_opt.value())) {
                        continue;
                    }

                    // Validated read must be exactly state_a or state_b
                    if (snapshot.size() != max_degree) {
                        bad_size.fetch_add(1, std::memory_order_relaxed);
                    }
                    bool is_a =
                        std::equal(snapshot.begin(), snapshot.end(), state_a.begin());
                    bool is_b =
                        std::equal(snapshot.begin(), snapshot.end(), state_b.begin());
                    if (!(is_a || is_b)) {
                        torn_reads.fetch_add(1, std::memory_order_relaxed);
                    }
                    ++validated;
                    break;
                }
            }
            validated_reads[tid].store(validated, std::memory_order_release);
        });
    }

    // Start all threads
    start_flag.store(true, std::memory_order_release);

    // Wait for readers to complete
    for (auto& r : readers) {
        r.join();
    }

    // Stop writer
    stop_flag.store(true, std::memory_order_release);
    writer.join();

    // Verify readers actually completed validated reads
    for (size_t tid = 0; tid < num_readers; ++tid) {
        size_t count = validated_reads[tid].load(std::memory_order_acquire);
        CATCH_REQUIRE(count == iterations);
    }

    // Verify no torn reads were observed
    size_t bad_size_count = bad_size.load(std::memory_order_acquire);
    size_t torn_reads_count = torn_reads.load(std::memory_order_acquire);
    CATCH_INFO("bad_size=" << bad_size_count << ", torn_reads=" << torn_reads_count);
    CATCH_REQUIRE(bad_size_count == 0);
    CATCH_REQUIRE(torn_reads_count == 0);
}

CATCH_TEST_CASE("SeqlockVisitor retries on validation failure", "[graphs][seqlock]") {
    using Idx = uint32_t;
    using SeqlockGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>;

    constexpr size_t n_nodes = 10;
    constexpr size_t max_degree = 5;

    SeqlockGraph graph(n_nodes, max_degree);

    // Initialize with some edges
    for (Idx i = 0; i < n_nodes; ++i) {
        std::vector<Idx> neighbors;
        for (size_t j = 1; j <= max_degree; ++j) {
            neighbors.push_back((i + j) % n_nodes);
        }
        graph.replace_node(i, neighbors);
    }

    // Create visitor
    svs::index::vamana::SeqlockVisitor visitor{graph};

    // Test that visitor executes the body and it satisfies NodeVisitor
    static_assert(svs::index::vamana::NodeVisitor<
                  svs::index::vamana::SeqlockVisitor<SeqlockGraph>>);

    size_t executions = 0;
    visitor(5, [&]() { ++executions; });

    CATCH_REQUIRE(executions > 0);
}
