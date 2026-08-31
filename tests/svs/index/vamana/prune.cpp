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

// svs/core must come before svs/index for proper name lookup
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"

// header under test
#include "svs/index/vamana/prune.h"

// catch2
#include "catch2/catch_test_macros.hpp"

#include <vector>

namespace {
struct SimpleAccessor {
    template <typename Data> auto operator()(const Data& data, size_t i) const {
        return data.get_datum(i);
    }
    template <typename Data>
    void prefetch(const Data& SVS_UNUSED(data), size_t SVS_UNUSED(i)) const {}
};
} // namespace

CATCH_TEST_CASE("Pruning", "[index][vamana]") {
    namespace v = svs::index::vamana;
    // Protect against changes to the default strategies getting merged.
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceL2>,
                  v::ProgressivePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceIP>,
                  v::IterativePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceCosineSimilarity>,
                  v::IterativePruneStrategy>);

    CATCH_SECTION("Iterative Strategy") {
        CATCH_SECTION("Prune State") {
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Available) == v::PruneState::Available
            );
            CATCH_REQUIRE(v::reenable(v::PruneState::Added) == v::PruneState::Added);
            CATCH_REQUIRE(v::reenable(v::PruneState::Pruned) == v::PruneState::Available);

            CATCH_REQUIRE(v::excluded(v::PruneState::Available) == false);
            CATCH_REQUIRE(v::excluded(v::PruneState::Added) == true);
            CATCH_REQUIRE(v::excluded(v::PruneState::Pruned) == true);
        }
    }

    CATCH_SECTION("Two-Phase Strategy") {
        CATCH_SECTION("Candidate State") {
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Candidate) == v::PruneState::Candidate
            );
            CATCH_REQUIRE(v::excluded(v::PruneState::Candidate) == true);
        }

        CATCH_SECTION("Basic Functionality") {
            constexpr size_t num_points = 5;
            constexpr size_t dims = 2;
            auto dataset = svs::data::SimpleData<float>(num_points, dims);
            dataset.get_datum(0)[0] = 0.0f;
            dataset.get_datum(0)[1] = 0.0f;
            dataset.get_datum(1)[0] = 1.0f;
            dataset.get_datum(1)[1] = 0.0f;
            dataset.get_datum(2)[0] = 0.0f;
            dataset.get_datum(2)[1] = 1.0f;
            dataset.get_datum(3)[0] = 2.0f;
            dataset.get_datum(3)[1] = 0.0f;
            dataset.get_datum(4)[0] = 3.0f;
            dataset.get_datum(4)[1] = 0.0f;

            auto accessor = SimpleAccessor();
            auto distance = svs::distance::DistanceL2();

            std::vector<svs::Neighbor<size_t>> pool = {
                svs::Neighbor<size_t>(1, 1.0f),
                svs::Neighbor<size_t>(2, 1.0f),
                svs::Neighbor<size_t>(3, 4.0f),
                svs::Neighbor<size_t>(4, 9.0f)};

            std::vector<size_t> result;
            v::heuristic_prune_neighbors(
                v::TwoPhasePruneStrategy(),
                3,
                1.2f,
                dataset,
                accessor,
                distance,
                0,
                std::span<const svs::Neighbor<size_t>>(pool),
                result
            );

            CATCH_REQUIRE(result.size() <= 3);
            CATCH_REQUIRE(result.size() > 0);
            for (size_t i = 0; i < result.size(); ++i) {
                for (size_t j = i + 1; j < result.size(); ++j) {
                    CATCH_REQUIRE(result[i] != result[j]);
                }
            }
        }
    }
}
