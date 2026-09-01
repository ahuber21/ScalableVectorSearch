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

#pragma once

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/concepts/graph.h"
#include "svs/index/vamana/search_buffer.h"

#include <algorithm>
#include <memory>

namespace svs::index::vamana {

// Include a stat - tracker API.
class NullTracker {
  public:
    NullTracker() = default;

    template <typename I>
    void visited(Neighbor<I> /*neighbor*/, size_t /*num_distance_computations*/) {}
};

template <typename T, typename I>
concept GreedySearchTracker =
    requires(T tracker, Neighbor<I> neighbor, size_t num_distance_computations) {
        tracker.visited(neighbor, num_distance_computations);
    };

struct GreedySearchPrefetchParameters {
  public:
    /// The number of iterations to prefetch ahead.
    size_t lookahead{4};
    /// The number of neighbors to prefetch at a time until `lookahead` is reached.
    size_t step{2};
};

/////
///// Initialization Customization.
/////

/// A greedy search initializer that resets the provided search buffer and appends
/// entry-points to the buffer.
///
/// This feature is low-level and should not be customized lightly.
///
/// This is the default initializer for greedy search.
template <std::integral I> struct EntryPointInitializer {
    template <
        typename Buffer,
        typename Computer,
        graphs::ImmutableMemoryGraph Graph,
        typename Builder,
        typename Tracker>
    void operator()(
        Buffer& buffer,
        const Computer& computer,
        const Graph& graph,
        const Builder& builder,
        Tracker& tracker
    ) const {
        using Idx = typename Buffer::index_type;

        // Reset the buffer for a new search.
        buffer.clear();
        // Add all entry points to the buffer.
        for (I id : entry_points_) {
            auto dist = computer(id);
            buffer.push_back(builder(id, dist));
            graph.prefetch_node(id);
            tracker.visited(Neighbor<Idx>{id, dist}, 1);
        }
        // We've added all the entry points.
        // Finish initializing the search buffer by sorting and preparing for a new run.
        buffer.sort();
    }

    ///// Members
    std::span<const I> entry_points_;
};

// Deduction Guide
template <std::integral I>
EntryPointInitializer(std::span<const I>) -> EntryPointInitializer<I>;

/////
///// Greedy Search
/////

// Default builder for generating neighbor elements.
// Alternative builders that return some builder-like object are supported to enable
// alternative search buffers to be used.
struct NeighborBuilder {
    template <typename I>
    constexpr SearchNeighbor<I> operator()(I i, float distance) const {
        return SearchNeighbor<I>{i, distance};
    }
};

/// Default visitor that invokes the per-node expansion body exactly once.
/// The body receives the graph's own neighbor range with no copy or validation.
struct VisitOnce {
    template <graphs::ImmutableMemoryGraph Graph, typename F>
    void operator()(const Graph& graph, size_t node_id, F&& body) const {
        body(graph.get_node(static_cast<typename Graph::index_type>(node_id)));
    }
};

/// Visitor that wraps node expansion in a seqlock retry loop.
/// Copies the adjacency list, validates, then invokes the body with the validated copy.
/// On validation failure, retries from the beginning. The body is never invoked with
/// torn data.
struct SeqlockVisitor {
    template <graphs::ImmutableMemoryGraph Graph, typename F>
    void operator()(const Graph& graph, size_t node_id, F&& body) const {
        using Idx = typename Graph::index_type;
        constexpr size_t kMaxRetries = 10'000;

        // Thread-local scratch storage for the adjacency list copy.
        thread_local std::vector<Idx> scratch;

        // Ensure scratch is large enough for this graph's max degree.
        if (scratch.size() < graph.max_degree()) {
            scratch.resize(graph.max_degree());
        }

        Idx id = static_cast<Idx>(node_id);
        size_t retry_count = 0;

        while (true) {
            auto seq_opt = graph.read_begin(id);
            if (!seq_opt.has_value()) {
                ++retry_count;
                if (retry_count >= kMaxRetries) {
                    throw lib::ANNException(
                        "SeqlockVisitor: exceeded retry limit for node " +
                        std::to_string(node_id) + " after " + std::to_string(retry_count) +
                        " attempts"
                    );
                }
                continue;
            }

            // Copy the adjacency list into scratch.
            auto neighbors = graph.get_node(id);
            const size_t n = neighbors.size();
            std::copy(neighbors.begin(), neighbors.end(), scratch.begin());

            // Validate the read before publishing anything.
            if (!graph.read_validate(id, seq_opt.value())) {
                ++retry_count;
                if (retry_count >= kMaxRetries) {
                    throw lib::ANNException(
                        "SeqlockVisitor: exceeded retry limit for node " +
                        std::to_string(node_id) + " after " + std::to_string(retry_count) +
                        " attempts"
                    );
                }
                continue;
            }

            // Validation succeeded. Call the body once with the validated copy.
            body(std::span<const Idx>{scratch.data(), n});
            return;
        }
    }
};

/// Concept for per-node visitors.
/// A visitor receives the graph, node ID, and a callable body that accepts the
/// neighbor range, and may invoke that body one or more times.
template <typename V, typename Graph>
concept NodeVisitor = graphs::ImmutableMemoryGraph<Graph> &&
                      requires(const V& visitor, const Graph& graph, size_t node_id) {
                          {
                              visitor(graph, node_id, [](const auto& /*neighbors*/) {})
                              } -> std::same_as<void>;
                      };

template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Dataset,
    data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder,
    GreedySearchTracker<typename Graph::index_type> Tracker,
    NodeVisitor<Graph> Visitor = VisitOnce>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    const QueryType& query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder,
    Tracker& search_tracker,
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>()),
    const Visitor& visit_node = {}
) {
    using I = typename Graph::index_type;

    // Fix the query if needed by the distance function.
    distance::maybe_fix_argument(distance_function, query);

    // Initialize the search buffer.
    {
        // A lambda that wraps the distance computation to avoid propagating everything
        // into the initializer.
        auto computer = [&](std::integral auto id) {
            return distance::compute(distance_function, query, accessor(dataset, id));
        };
        initializer(search_buffer, computer, graph, builder, search_tracker);
    }

    // Main search routine.
    while (!search_buffer.done()) {
        // Check if request to cancel the search
        if (cancel()) {
            return;
        }
        // Get the next unvisited vertex.
        const auto& node = search_buffer.next();
        auto node_id = node.id();

        visit_node(graph, node_id, [&](const auto& neighbors) {
            // The visitor supplies the validated neighbor range.
            const size_t num_neighbors = neighbors.size();
            search_tracker.visited(Neighbor<I>{node}, neighbors.size());

            auto prefetcher = lib::make_prefetcher(
                lib::PrefetchParameters{
                    prefetch_parameters.lookahead, prefetch_parameters.step},
                num_neighbors,
                [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },
                [&](size_t i) {
                    // Perform the visited set enabled check just once.
                    if (search_buffer.visited_set_enabled()) {
                        // Prefetch next bucket so it's (hopefully) in the cache when we
                        // next consult the visited filter.
                        if (i + 1 < num_neighbors) {
                            search_buffer.unsafe_prefetch_visited(neighbors[i + 1]);
                        }
                        return !search_buffer.unsafe_is_visited(neighbors[i]);
                    }

                    // Otherwise, always prefetch the next data item.
                    return true;
                }
            );

            ///// Neighbor expansion.
            prefetcher();
            for (auto id : neighbors) {
                if (search_buffer.emplace_visited(id)) {
                    continue;
                }

                // Run the prefetcher.
                prefetcher();

                // Compute distance and update search buffer.
                auto dist =
                    distance::compute(distance_function, query, accessor(dataset, id));
                search_buffer.insert(builder(id, dist));
            }
        });
    }
}

// Overload to provide a default search tracker because search trackers are taken by
// lvalue reference.
template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Dataset,
    data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder = NeighborBuilder,
    NodeVisitor<Graph> Visitor = VisitOnce>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    QueryType query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder = NeighborBuilder(),
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>()),
    const Visitor& visit_node = {}
) {
    auto null_tracker = NullTracker{};
    greedy_search(
        graph,
        dataset,
        accessor,
        query,
        distance_function,
        search_buffer,
        initializer,
        builder,
        null_tracker,
        prefetch_parameters,
        cancel,
        visit_node
    );
}
} // namespace svs::index::vamana
