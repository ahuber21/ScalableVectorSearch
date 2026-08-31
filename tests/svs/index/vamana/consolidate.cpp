/*
 * Copyright 2023 Intel Corporation
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

// header under test
#include "svs/index/vamana/consolidate.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/timing.h"

// test utilities
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <concepts>
#include <functional>
#include <vector>

namespace {

template <typename G>
concept HasReverseEdges = requires(G& g) { g.reverse_edges(); };

template <typename Graph, typename Predicate>
void check_post_conditions(const Graph& graph, Predicate&& predicate) {
    bool contains_deleted = false;
    svs::threads::UnitRange<size_t> node_range{0, graph.n_nodes()};
    for (size_t i : node_range) {
        if (predicate(i)) {
            contains_deleted = true;
            continue;
        }

        const auto& neighbors = graph.get_node(i);
        CATCH_REQUIRE(std::none_of(neighbors.begin(), neighbors.end(), predicate));

        // Don't invent nodes out of thin air.
        CATCH_REQUIRE(std::all_of(neighbors.begin(), neighbors.end(), [&](const auto& i) {
            return node_range.contains(i);
        }));
    }
    CATCH_REQUIRE(contains_deleted);
}

} // namespace

CATCH_TEST_CASE("Graph Consolidation", "[graph_index]") {
    auto graph = test_dataset::graph();
    auto data = test_dataset::data_f32();
    auto threadpool = svs::threads::DefaultThreadPool(2);

    CATCH_SECTION("Partial Consolidation Requires Reverse Edges") {
        // Verify that SimpleGraph does not satisfy the reverse edges requirement.
        // The partial consolidation operator() should not be callable for SimpleGraph.
        using Graph = decltype(graph);

        static_assert(!HasReverseEdges<Graph>);
    }

    CATCH_SECTION("Remove Even Nodes") {
        auto tic = svs::lib::now();
        auto predicate = [](const auto& i) { return (i % 10) == 0; };

        svs::distance::DistanceL2 distance{};
        svs::index::vamana::consolidate(
            graph, data, threadpool, graph.max_degree(), 750, 1.2, distance, predicate
        );
        std::cout << "Pruning took " << svs::lib::time_difference(svs::lib::now(), tic)
                  << std::endl;

        // Ensure that all non-deleted nodes only have non-deleted neighbors.
        check_post_conditions(graph, predicate);
    }

    CATCH_SECTION("SeqlockAccess graph consolidation") {
        using SeqlockGraph = svs::graphs::SimpleGraphBase<
            uint32_t,
            svs::data::SimpleData<uint32_t, svs::Dynamic>,
            svs::graphs::SeqlockAccess>;

        auto plain_graph = test_dataset::graph();
        SeqlockGraph seqlock_graph(plain_graph.n_nodes(), plain_graph.max_degree());

        for (size_t i = 0; i < plain_graph.n_nodes(); ++i) {
            const auto& neighbors = plain_graph.get_node(i);
            std::vector<uint32_t> neighbor_vec(neighbors.begin(), neighbors.end());
            seqlock_graph.replace_node(i, neighbor_vec);
        }

        auto predicate = [](const auto& i) { return (i % 10) == 0; };
        svs::distance::DistanceL2 distance{};
        // Single-threaded for deterministic total edge count comparison.
        auto single_thread = svs::threads::DefaultThreadPool(1);

        svs::index::vamana::consolidate(
            plain_graph,
            data,
            single_thread,
            plain_graph.max_degree(),
            750,
            1.2,
            distance,
            predicate
        );
        svs::index::vamana::consolidate(
            seqlock_graph,
            data,
            single_thread,
            seqlock_graph.max_degree(),
            750,
            1.2,
            distance,
            predicate
        );

        check_post_conditions(plain_graph, predicate);
        check_post_conditions(seqlock_graph, predicate);

        size_t plain_total_edges = 0;
        size_t seqlock_total_edges = 0;
        size_t nodes_with_differing_sets = 0;
        size_t first_mismatch = SIZE_MAX;

        for (size_t i = 0; i < plain_graph.n_nodes(); ++i) {
            const auto& plain_neighbors = plain_graph.get_node(i);
            const auto& seqlock_neighbors = seqlock_graph.get_node(i);

            plain_total_edges += plain_neighbors.size();
            seqlock_total_edges += seqlock_neighbors.size();

            std::unordered_set<uint32_t> plain_set(
                plain_neighbors.begin(), plain_neighbors.end()
            );
            std::unordered_set<uint32_t> seqlock_set(
                seqlock_neighbors.begin(), seqlock_neighbors.end()
            );

            if (plain_set != seqlock_set) {
                ++nodes_with_differing_sets;
                if (first_mismatch == SIZE_MAX) {
                    first_mismatch = i;
                }
            }
        }

        CATCH_INFO("First mismatch at node " << first_mismatch);
        CATCH_REQUIRE(nodes_with_differing_sets == 0);
        CATCH_REQUIRE(plain_total_edges == seqlock_total_edges);
    }
}
