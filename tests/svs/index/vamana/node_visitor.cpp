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

// Tests for node visitor implementations: VisitOnce and SeqlockVisitor.
// AR-1: speculative seqlock visitor with bounded retry.

#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/core/graph/graph.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/sync_policy.h"
#include "svs/lib/exception.h"

#include "catch2/catch_test_macros.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace {

using Idx = uint32_t;

// Mock graph that returns different neighbors on first vs. subsequent reads.
// Models a torn read that is corrected on retry.
template <typename Access> class TornReadMockGraph {
  public:
    using index_type = Idx;
    using const_reference = std::span<const Idx>;
    using reference = std::span<Idx>;

    TornReadMockGraph(std::vector<Idx> valid_neighbors, std::vector<Idx> garbage_neighbors)
        : valid_{std::move(valid_neighbors)}
        , garbage_{std::move(garbage_neighbors)}
        , read_count_{0} {}

    size_t n_nodes() const { return 10; }
    size_t max_degree() const { return std::max(valid_.size(), garbage_.size()); }

    const_reference get_node(Idx id) const {
        if (id == 0) {
            // On first read, return garbage. On subsequent reads, return valid neighbors.
            if (read_count_++ == 0) {
                return std::span<const Idx>(garbage_.data(), garbage_.size());
            }
        }
        return std::span<const Idx>(valid_.data(), valid_.size());
    }

    size_t get_node_degree(Idx /*id*/) const { return valid_.size(); }
    void prefetch_node(Idx /*id*/) const {}

    // Seqlock interface: first read (seq=0) will fail validation.
    std::optional<uint32_t> read_begin(Idx id) const {
        if (id == 0) {
            return static_cast<uint32_t>(read_count_.load());
        }
        return 0;
    }

    bool read_validate(Idx id, uint32_t seq) const {
        if (id == 0) {
            // First call has seq=0, read_count_=1 after get_node, so fails.
            // Second call has seq=1, read_count_=2 after get_node, so succeeds.
            return static_cast<uint32_t>(read_count_.load()) == seq;
        }
        return true;
    }

  private:
    std::vector<Idx> valid_;
    std::vector<Idx> garbage_;
    mutable std::atomic<size_t> read_count_;
};

// Mock graph that never validates a specific node, forcing retry exhaustion.
template <typename Access> class NeverValidateMockGraph {
  public:
    using index_type = Idx;
    using const_reference = std::span<const Idx>;
    using reference = std::span<Idx>;

    explicit NeverValidateMockGraph(std::vector<Idx> neighbors)
        : neighbors_{std::move(neighbors)}
        , retry_count_{0} {}

    size_t n_nodes() const { return 10; }
    size_t max_degree() const { return neighbors_.size(); }

    const_reference get_node(Idx /*id*/) const {
        return std::span<const Idx>(neighbors_.data(), neighbors_.size());
    }

    size_t get_node_degree(Idx /*id*/) const { return neighbors_.size(); }
    void prefetch_node(Idx /*id*/) const {}

    // Seqlock interface: always fail validation for node 0.
    std::optional<uint32_t> read_begin(Idx id) const {
        if (id == 0) {
            ++retry_count_;
        }
        return 0;
    }

    bool read_validate(Idx id, uint32_t /*seq*/) const {
        // Node 0 never validates.
        return id != 0;
    }

    size_t get_retry_count() const { return retry_count_.load(); }

  private:
    std::vector<Idx> neighbors_;
    mutable std::atomic<size_t> retry_count_;
};

} // namespace

CATCH_TEST_CASE(
    "VisitOnce calls body with graph neighbors", "[index][vamana][node_visitor]"
) {
    // Test that VisitOnce passes through the graph's neighbors directly.
    using Graph = svs::graphs::SimpleBlockedGraph<Idx>;
    std::vector<std::vector<Idx>> adjacency = {{1, 2, 3}, {0, 2}, {0, 1}};
    auto data = svs::data::SimpleData<Idx>(adjacency.size(), adjacency[0].size());
    for (size_t i = 0; i < adjacency.size(); ++i) {
        data.set_datum(i, std::span<const Idx>(adjacency[i].data(), adjacency[i].size()));
    }
    Graph graph{std::move(data), adjacency[0].size()};

    svs::index::vamana::VisitOnce visitor{};

    bool body_called = false;
    size_t neighbor_count = 0;

    // New API: visitor receives graph and calls body with neighbor range.
    visitor(graph, 0, [&](const auto& neighbors) {
        body_called = true;
        neighbor_count = neighbors.size();
        // Verify we see the expected neighbors.
        CATCH_REQUIRE(neighbor_count == 3);
        CATCH_REQUIRE(neighbors[0] == 1);
        CATCH_REQUIRE(neighbors[1] == 2);
        CATCH_REQUIRE(neighbors[2] == 3);
    });

    CATCH_REQUIRE(body_called);
}

CATCH_TEST_CASE("SeqlockVisitor retries on torn read", "[index][vamana][node_visitor]") {
    // Test that SeqlockVisitor retries when validation fails and eventually succeeds.
    using Graph = TornReadMockGraph<svs::graphs::SeqlockAccess>;

    std::vector<Idx> valid = {1, 2};
    std::vector<Idx> garbage = {9999, 9998, 9997};
    Graph graph{valid, garbage};

    svs::index::vamana::SeqlockVisitor visitor{};

    size_t body_call_count = 0;
    std::vector<Idx> observed_neighbors;

    // Visitor should retry: first attempt sees garbage and fails validation,
    // second attempt sees valid neighbors and succeeds.
    visitor(graph, 0, [&](const auto& neighbors) {
        ++body_call_count;
        observed_neighbors.clear();
        for (const auto& n : neighbors) {
            observed_neighbors.push_back(n);
        }
    });

    // Body should have been called exactly once (only after successful validation).
    CATCH_REQUIRE(body_call_count == 1);
    // And it should have seen the valid neighbors, not the garbage.
    CATCH_REQUIRE(observed_neighbors.size() == 2);
    CATCH_REQUIRE(observed_neighbors[0] == 1);
    CATCH_REQUIRE(observed_neighbors[1] == 2);
}

CATCH_TEST_CASE(
    "SeqlockVisitor throws after max retries", "[index][vamana][node_visitor]"
) {
    // Test that SeqlockVisitor throws ANNException after exceeding kMaxRetries.
    using Graph = NeverValidateMockGraph<svs::graphs::SeqlockAccess>;

    std::vector<Idx> neighbors = {1, 2};
    Graph graph{neighbors};

    svs::index::vamana::SeqlockVisitor visitor{};

    bool threw_exception = false;
    std::string exception_message;

    try {
        visitor(graph, 0, [](const auto& /*neighbors*/) {
            // Body should never actually be called since validation always fails.
        });
    } catch (const svs::ANNException& e) {
        threw_exception = true;
        exception_message = e.what();
    }

    CATCH_REQUIRE(threw_exception);
    // Exception message should mention the node ID and retry count.
    CATCH_INFO("Exception: " << exception_message);
    CATCH_REQUIRE(exception_message.find("node") != std::string::npos);
    // The retry count should have hit the max.
    // Note: get_retry_count counts read_begin calls, which happen on each retry.
    CATCH_REQUIRE(graph.get_retry_count() > 1000);
}

CATCH_TEST_CASE(
    "SeqlockSync instantiates SeqlockVisitor", "[index][vamana][node_visitor][compile]"
) {
    // Test that SeqlockSync's node_visitor_type is SeqlockVisitor.
    using Sync = svs::index::vamana::SeqlockSync;
    using VisitorType = Sync::node_visitor_type;

    // Force instantiation with sizeof.
    static_assert(sizeof(VisitorType) > 0, "Visitor must be complete");

    // The visitor must be SeqlockVisitor, not VisitOnce.
    static_assert(
        std::is_same_v<VisitorType, svs::index::vamana::SeqlockVisitor>,
        "SeqlockSync must use SeqlockVisitor"
    );
}

CATCH_TEST_CASE(
    "SequentialSync instantiates VisitOnce", "[index][vamana][node_visitor][compile]"
) {
    // Test that SequentialSync's node_visitor_type is VisitOnce.
    using Sync = svs::index::vamana::SequentialSync;
    using VisitorType = Sync::node_visitor_type;

    static_assert(sizeof(VisitorType) > 0, "Visitor must be complete");
    static_assert(
        std::is_same_v<VisitorType, svs::index::vamana::VisitOnce>,
        "SequentialSync must use VisitOnce"
    );
}
