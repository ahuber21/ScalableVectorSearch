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

#include <algorithm>
#include <random>
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
            // Dataset that forces phase 2 to run by producing Candidate nodes.
            // With alpha=1.2, id2 and id3 fall in the band where cmp(djk, dist)
            // but not cmp(alpha*djk, dist), becoming Candidates rather than Pruned.
            constexpr size_t num_points = 4;
            constexpr size_t dims = 2;
            auto dataset = svs::data::SimpleData<float>(num_points, dims);
            dataset.get_datum(0)[0] = 0.0f;
            dataset.get_datum(0)[1] = 0.0f;
            dataset.get_datum(1)[0] = 1.0f;
            dataset.get_datum(1)[1] = 0.0f;
            dataset.get_datum(2)[0] = 0.6f;
            dataset.get_datum(2)[1] = 1.0f;
            dataset.get_datum(3)[0] = 0.7f;
            dataset.get_datum(3)[1] = 1.5f;

            auto accessor = SimpleAccessor();
            auto distance = svs::distance::DistanceL2();

            // Pool sorted by squared distance from p0: 1.00, 1.36, 2.74.
            std::vector<svs::Neighbor<size_t>> pool = {
                svs::Neighbor<size_t>(1, 1.00f),
                svs::Neighbor<size_t>(2, 1.36f),
                svs::Neighbor<size_t>(3, 2.74f)};

            std::vector<size_t> result_alpha_1_2;
            v::heuristic_prune_neighbors(
                v::TwoPhasePruneStrategy(),
                3,
                1.2f,
                dataset,
                accessor,
                distance,
                0,
                std::span<const svs::Neighbor<size_t>>(pool),
                result_alpha_1_2
            );

            // Phase 1 adds id1, marks id2 and id3 as Candidate.
            // Phase 2 adds id2, prunes id3. Expected: exactly {1, 2}.
            CATCH_REQUIRE(result_alpha_1_2.size() == 2);
            CATCH_REQUIRE(result_alpha_1_2[0] == 1);
            CATCH_REQUIRE(result_alpha_1_2[1] == 2);

            // Contrasting case: alpha=1.0 collapses the band, so id2 and id3
            // are Pruned in phase 1, phase 2 has no Candidates. Expected: {1}.
            std::vector<size_t> result_alpha_1_0;
            v::heuristic_prune_neighbors(
                v::TwoPhasePruneStrategy(),
                3,
                1.0f,
                dataset,
                accessor,
                distance,
                0,
                std::span<const svs::Neighbor<size_t>>(pool),
                result_alpha_1_0
            );

            CATCH_REQUIRE(result_alpha_1_0.size() == 1);
            CATCH_REQUIRE(result_alpha_1_0[0] == 1);

            // Results differ, proving phase 2 executed for alpha=1.2.
            CATCH_REQUIRE(result_alpha_1_2 != result_alpha_1_0);
        }

        CATCH_SECTION("Differential Test: TwoPhase vs Iterative") {
            // Generate deterministic random pools and verify both strategies
            // produce valid results, then confirm they sometimes diverge.
            constexpr size_t num_points = 20;
            constexpr size_t dims = 4;
            auto accessor = SimpleAccessor();
            auto distance = svs::distance::DistanceL2();
            constexpr size_t max_result_size = 5;
            constexpr float alpha = 1.3f;
            constexpr size_t num_trials = 50;

            // RNG and distribution live outside the loop so the sequence advances
            // across trials while the test remains deterministic.
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(0.0f, 10.0f);

            size_t divergence_count = 0;
            for (size_t trial = 0; trial < num_trials; ++trial) {
                // Generate a fresh random dataset for each trial.
                auto dataset = svs::data::SimpleData<float>(num_points, dims);
                for (size_t i = 0; i < num_points; ++i) {
                    for (size_t j = 0; j < dims; ++j) {
                        dataset.get_datum(i)[j] = dist(rng);
                    }
                }

                // Build a random pool of 10 neighbors for node 0.
                std::vector<svs::Neighbor<size_t>> pool;
                for (size_t i = 1; i < std::min(num_points, size_t(11)); ++i) {
                    float d = svs::distance::compute(
                        distance, accessor(dataset, 0), accessor(dataset, i)
                    );
                    pool.emplace_back(i, d);
                }
                std::sort(pool.begin(), pool.end(), svs::distance::comparator(distance));

                std::vector<size_t> result_two_phase;
                v::heuristic_prune_neighbors(
                    v::TwoPhasePruneStrategy(),
                    max_result_size,
                    alpha,
                    dataset,
                    accessor,
                    distance,
                    0,
                    std::span<const svs::Neighbor<size_t>>(pool),
                    result_two_phase
                );

                std::vector<size_t> result_iterative;
                v::heuristic_prune_neighbors(
                    v::IterativePruneStrategy(),
                    max_result_size,
                    alpha,
                    dataset,
                    accessor,
                    distance,
                    0,
                    std::span<const svs::Neighbor<size_t>>(pool),
                    result_iterative
                );

                // Validity checks for both strategies, avoiding copies from
                // initializer_list.
                std::vector<const std::vector<size_t>*> results = {
                    &result_two_phase, &result_iterative};
                for (const auto result_ptr : results) {
                    const auto& result = *result_ptr;
                    CATCH_REQUIRE(result.size() <= max_result_size);
                    for (size_t i = 0; i < result.size(); ++i) {
                        CATCH_REQUIRE(result[i] != 0);
                        for (size_t j = i + 1; j < result.size(); ++j) {
                            CATCH_REQUIRE(result[i] != result[j]);
                        }
                        bool found = false;
                        for (const auto& n : pool) {
                            if (n.id() == result[i]) {
                                found = true;
                                break;
                            }
                        }
                        CATCH_REQUIRE(found);
                    }
                }

                if (result_two_phase != result_iterative) {
                    ++divergence_count;
                }
            }

            CATCH_INFO("divergence_count: " << divergence_count << " / " << num_trials);
            // The two strategies disagree on most inputs, validating the distinct tag.
            CATCH_REQUIRE(divergence_count > num_trials / 2);
        }
    }
}
