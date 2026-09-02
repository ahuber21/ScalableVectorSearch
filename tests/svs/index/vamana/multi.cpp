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

// header under test
#include "svs/index/vamana/multi.h"

// svstest
#include "tests/utils/test_dataset.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// Verify NullMutex is empty and adds zero overhead via [[no_unique_address]].
static_assert(std::is_empty_v<svs::lib::NullMutex>);

template <typename Distance> float pick_alpha(Distance SVS_UNUSED(dist)) {
    if constexpr (std::is_same_v<Distance, svs::DistanceL2>) {
        return 1.2;
    } else if constexpr (std::is_same_v<Distance, svs::DistanceIP>) {
        return 0.95;
    } else if constexpr (std::is_same_v<Distance, svs::DistanceCosineSimilarity>) {
        return 0.95;
    } else {
        throw ANNEXCEPTION("Unsupported distance type!");
    }
}

} // namespace

CATCH_TEMPLATE_TEST_CASE(
    "Multi-vector dynamic vamana index",
    "[long][index][vamana][multi]",
    svs::DistanceL2,
    svs::DistanceIP,
    svs::DistanceCosineSimilarity
) {
    using Eltype = float;
    using Distance = TestType;
    const size_t N = 128;
    const size_t max_degree = 64;
    const float alpha = pick_alpha(Distance());
    const size_t num_threads = 4;
    const size_t num_neighbors = 10;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const auto num_points = data.size();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        alpha, max_degree, 2 * max_degree, 1000, max_degree - 4, true};

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();

    const float epsilon = 0.05f;
    std::vector<size_t> ref_indices(num_points);
    std::iota(ref_indices.begin(), ref_indices.end(), 0);

    auto ref_index = svs::index::vamana::MutableVamanaIndex(
        build_parameters, data, ref_indices, Distance(), num_threads
    );
    auto ref_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    ref_index.search(ref_results.view(), queries.view(), search_parameters);
    auto ref_recall = svs::k_recall_at_n(groundtruth, ref_results);

    // Original data label:
    // 0 1 2 3
    //
    // For each duplicate iteration, insert each vector with label increase by one
    // Suppose we duplicate three times (i.e., num_duplicated = 3):
    //   1 2 3 4
    //     2 3 4 5
    //       3 4 5 6
    //
    // After deleting all the original labels, the remaining
    // number of vectors will be :
    // (num_duplicated * (num_duplicated + 1)) / 2
    //
    // For the above examples, after deleting 0, 1, 2, 3
    // the remaining vectors becomes:
    //         4
    //         4 5
    //         4 5 6
    // And the number of remaining vectors becomes
    // (3 + 4) / 2 = 6 vectors
    CATCH_SECTION("Insertion/Deletion in duplicated test datasets") {
        const size_t num_duplicated = 3;

        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        for (size_t i = 0; i < num_duplicated; ++i) {
            std::iota(test_indices.begin(), test_indices.end(), i + 1);
            test_index.add_points(data, test_indices);
        }
        CATCH_REQUIRE(test_index.labelcount() == ref_index.size() + num_duplicated);
        CATCH_REQUIRE(test_index.size() == ref_index.size() * (num_duplicated + 1));

        std::iota(test_indices.begin(), test_indices.end(), 0);
        test_index.delete_entries(test_indices);
        CATCH_REQUIRE(test_index.labelcount() == num_duplicated);
        CATCH_REQUIRE(test_index.size() == (num_duplicated * (num_duplicated + 1)) / 2);
    }
    CATCH_SECTION("Duplicated vectors with same labels") {
        const size_t num_duplicated = 3;

        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        for (size_t i = 0; i < num_duplicated; ++i) {
            test_index.add_points(data, test_indices);
        }
        CATCH_REQUIRE(test_index.labelcount() == test_indices.size());
        CATCH_REQUIRE(test_index.size() == test_indices.size() * (num_duplicated + 1));

        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(groundtruth, test_results);

        CATCH_REQUIRE(test_recall > ref_recall - epsilon);

        test_index.delete_entries(test_indices);
        CATCH_REQUIRE(test_index.labelcount() == 0);
        CATCH_REQUIRE(test_index.size() == 0);

        test_index.add_points(data, test_indices);
        test_index.consolidate();
        test_index.compact();
        for (size_t i = 0; i < num_duplicated; ++i) {
            test_index.add_points(data, test_indices);
        }

        auto test_results2 = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results2.view(), queries.view(), search_parameters);
        auto test_recall2 = svs::k_recall_at_n(groundtruth, test_results2);

        CATCH_REQUIRE(test_recall2 > test_recall - epsilon);
        CATCH_REQUIRE(test_recall2 < test_recall + epsilon);
    }

    CATCH_SECTION("Step grouping") {
        size_t start = 0;
        size_t step = 4;
        CATCH_REQUIRE(num_points % step == 0);
        size_t num_groups = num_points / step;

        auto remapped_groundtruth = groundtruth;
        CATCH_REQUIRE(remapped_groundtruth.size() == queries.size());

        // It is okay to have duplicated neighbor ids in groundtruth
        // as the recall is checked by counting intersect
        for (size_t i = 0; i < queries.size(); ++i) {
            auto arr = remapped_groundtruth.get_datum(i);
            for (auto& each : arr) {
                each /= step;
            }
        }

        std::vector<size_t> test_indices(num_points);
        for (size_t i = 0; i < num_points; i += step) {
            for (size_t s = 0; s < step; ++s) {
                test_indices[i + s] = start;
            }
            ++start;
        }

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );
        test_index.add_points(data, test_indices);

        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(remapped_groundtruth, test_results);

        CATCH_REQUIRE(test_recall > ref_recall - epsilon);

        // test get_distance
        for (size_t i = 0; i < queries.size(); ++i) {
            size_t k = std::rand() % num_groups;
            double ref_distance = svs::INVALID_DISTANCE;
            for (size_t s = 0; s < step; ++s) {
                if constexpr (std::is_same_v<Distance, svs::distance::DistanceL2>) {
                    ref_distance = std::fmin(
                        ref_distance,
                        ref_index.get_distance(
                            ref_indices[k * step + s], queries.get_datum(i)
                        )
                    );
                } else {
                    ref_distance = std::fmax(
                        ref_distance,
                        ref_index.get_distance(
                            ref_indices[k * step + s], queries.get_datum(i)
                        )
                    );
                }
            }

            double test_distance =
                test_index.get_distance(test_indices[k * step], queries.get_datum(i));
            CATCH_REQUIRE(test_distance == ref_distance);
        }
    }

    CATCH_SECTION("Logging") {
        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        CATCH_REQUIRE(ref_index.get_logger() == test_index.get_logger());
    }

    CATCH_SECTION("Save/Load") {
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        auto config_dir = dir / "config";
        auto graph_dir = dir / "graph";
        auto data_dir = dir / "data";
        std::vector<size_t> test_indices(num_points);
        // Fill the test indices with labels in the range of num_labels
        // to ensure that there are labels mapped to more than 1 vector.
        const size_t per_label = 2;
        const auto num_labels = num_points / per_label;
        for (auto& i : test_indices) {
            i = std::rand() % num_labels;
        }
        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );
        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(groundtruth, test_results);

        test_index.save(config_dir, graph_dir, data_dir);

        auto test_index_2 = svs::index::vamana::auto_multi_dynamic_assemble(
            config_dir,
            svs::GraphLoader(graph_dir),
            svs::VectorDataLoader<float>(data_dir),
            Distance(),
            svs::threads::CppAsyncThreadPool(2)
        );
        auto test_results_2 = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index_2.search(test_results_2.view(), queries.view(), search_parameters);
        auto test_recall_2 = svs::k_recall_at_n(groundtruth, test_results_2);

        // Check that the results are the same
        CATCH_REQUIRE(test_results.n_neighbors() == test_results_2.n_neighbors());
        for (size_t i = 0; i < test_results.n_queries(); ++i) {
            for (size_t j = 0; j < test_results.n_neighbors(); ++j) {
                CATCH_REQUIRE(
                    test_results.indices().at(i, j) == test_results_2.indices().at(i, j)
                );
            }
        }

        CATCH_REQUIRE(test_index.size() == test_index_2.size());
        CATCH_REQUIRE(test_index.dimensions() == test_index_2.dimensions());
        // Index Properties
        CATCH_REQUIRE(test_index.get_alpha() == test_index_2.get_alpha());
        CATCH_REQUIRE(
            test_index.get_construction_window_size() ==
            test_index_2.get_construction_window_size()
        );
        CATCH_REQUIRE(test_index.get_max_candidates() == test_index_2.get_max_candidates());
        CATCH_REQUIRE(test_index.max_degree() == test_index_2.max_degree());
        CATCH_REQUIRE(test_index.get_prune_to() == test_index_2.get_prune_to());
        CATCH_REQUIRE(
            test_index.get_full_search_history() == test_index_2.get_full_search_history()
        );
        CATCH_REQUIRE(test_index.view_data() == test_index_2.view_data());

        CATCH_REQUIRE(test_recall_2 > test_recall - epsilon);
    }
}

CATCH_TEST_CASE(
    "MultiMutableVamana Index Save and Load", "[index][vamana][multi][saveload]"
) {
    using Eltype = float;
    using Distance = svs::DistanceL2;
    const size_t N = 128;
    const size_t num_threads = 4;
    const size_t num_neighbors = 10;
    const size_t max_degree = 64;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const auto num_points = data.size();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        1.2, max_degree, 10, 20, 10, true};

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();

    const float epsilon = 0.05f;

    std::vector<size_t> test_indices(num_points);
    const size_t per_label = 2;
    const auto num_labels = num_points / per_label;
    for (auto& i : test_indices) {
        i = std::rand() % num_labels;
    }

    auto index = svs::index::vamana::MultiMutableVamanaIndex(
        build_parameters, data, test_indices, Distance(), num_threads
    );
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.view(), search_parameters);

    CATCH_SECTION("Load MultiMutableVamana Index being serialized natively to stream") {
        std::stringstream stream;
        index.save(stream);
        {
            auto deserializer = svs::lib::detail::Deserializer::build(stream);
            CATCH_REQUIRE(deserializer.is_native());

            using Data_t = svs::data::SimpleData<Eltype, N>;
            using GraphType = svs::graphs::SimpleBlockedGraph<uint32_t>;

            auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
                stream,
                [&]() -> GraphType { return GraphType::load(stream); },
                [&]() -> Data_t { return svs::lib::load_from_stream<Data_t>(stream); },
                Distance(),
                num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(loaded.max_degree() == index.max_degree());
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            CATCH_REQUIRE(loaded.view_data() == index.view_data());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.view(), search_parameters);
            for (size_t i = 0; i < results.n_queries(); ++i) {
                for (size_t j = 0; j < results.n_neighbors(); ++j) {
                    CATCH_REQUIRE(
                        results.indices().at(i, j) == loaded_results.indices().at(i, j)
                    );
                }
            }

            auto loaded_recall = svs::k_recall_at_n(groundtruth, loaded_results);
            auto test_recall = svs::k_recall_at_n(groundtruth, results);
            CATCH_REQUIRE(loaded_recall > test_recall - epsilon);
        }
    }

    CATCH_SECTION("Load MultiMutableVamana Index being serialized with intermediate files"
    ) {
        std::stringstream stream;
        svs::lib::UniqueTempDirectory tempdir{"svs_multivamana_save"};
        const auto config_dir = tempdir.get() / "config";
        const auto graph_dir = tempdir.get() / "graph";
        const auto data_dir = tempdir.get() / "data";
        std::filesystem::create_directories(config_dir);
        std::filesystem::create_directories(graph_dir);
        std::filesystem::create_directories(data_dir);
        index.save(config_dir, graph_dir, data_dir);
        svs::lib::DirectoryArchiver::pack(tempdir, stream);
        {
            using Data_t = svs::data::SimpleData<Eltype, N>;
            using GraphType = svs::graphs::SimpleBlockedGraph<uint32_t>;

            auto deserializer = svs::lib::detail::Deserializer::build(stream);
            CATCH_REQUIRE(!deserializer.is_native());
            svs::lib::DirectoryArchiver::unpack(stream, tempdir, deserializer.magic());

            auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
                config_dir,
                GraphType::load(graph_dir),
                Data_t::load(data_dir),
                Distance(),
                num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(loaded.max_degree() == index.max_degree());
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            CATCH_REQUIRE(loaded.view_data() == index.view_data());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.view(), search_parameters);
            for (size_t i = 0; i < results.n_queries(); ++i) {
                for (size_t j = 0; j < results.n_neighbors(); ++j) {
                    CATCH_REQUIRE(
                        results.indices().at(i, j) == loaded_results.indices().at(i, j)
                    );
                }
            }

            auto loaded_recall = svs::k_recall_at_n(groundtruth, loaded_results);
            auto test_recall = svs::k_recall_at_n(groundtruth, results);
            CATCH_REQUIRE(loaded_recall > test_recall - epsilon);
        }
    }
}

// AR-10 Tests: Sync policy integration for MultiMutableVamanaIndex

CATCH_TEST_CASE("Multi: Sync policy compile-time routing", "[index][vamana][multi][sync]") {
    // Force instantiation and verify that SeqlockSync is threaded into the parent.
    using Distance = svs::DistanceL2;
    using Eltype = float;
    using DataAlloc =
        svs::data::Blocked<svs::lib::Allocator<Eltype>, svs::data::SegmentStable>;
    using Data = svs::data::SimpleData<Eltype, svs::Dynamic, DataAlloc>;
    using GraphAlloc =
        svs::data::Blocked<svs::lib::Allocator<uint32_t>, svs::data::SegmentStable>;
    using GraphData = svs::data::SimpleData<uint32_t, svs::Dynamic, GraphAlloc>;
    using Graph =
        svs::graphs::SimpleGraphBase<uint32_t, GraphData, svs::graphs::SeqlockAccess>;

    using MultiSeqlock = svs::index::vamana::
        MultiMutableVamanaIndex<Graph, Data, Distance, svs::index::vamana::SeqlockSync>;

    // Force complete instantiation - a type alias proves nothing.
    [[maybe_unused]] auto size = sizeof(MultiSeqlock);
    CATCH_REQUIRE(size > 0);

    // Verify the parent index is instantiated with SeqlockSync.
    using ParentType = typename MultiSeqlock::ParentIndex;
    static_assert(
        std::is_same_v<
            ParentType,
            svs::index::vamana::
                MutableVamanaIndex<Graph, Data, Distance, svs::index::vamana::SeqlockSync>>,
        "Parent index must be instantiated with SeqlockSync"
    );

    // Verify TaggedMutex types are distinct under SeqlockSync.
    using L2EMutex = svs::index::vamana::detail::TaggedMutex<
        typename svs::index::vamana::SeqlockSync::mutex_type,
        svs::index::vamana::detail::L2ETag>;
    using E2LMutex = svs::index::vamana::detail::TaggedMutex<
        typename svs::index::vamana::SeqlockSync::mutex_type,
        svs::index::vamana::detail::E2LTag>;
    static_assert(
        !std::is_same_v<L2EMutex, E2LMutex>,
        "TaggedMutex types must be distinct to prevent mutex collapse"
    );
}

CATCH_TEST_CASE(
    "Multi: Behavior preservation with default policy", "[index][vamana][multi][sync]"
) {
    // Verify that three-argument instantiation still compiles and behaves identically.
    using Distance = svs::DistanceL2;
    using Eltype = float;
    const size_t N = 128;
    const size_t max_degree = 32;
    const float alpha = 1.2f;
    const size_t num_threads = 2;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const size_t num_points = std::min<size_t>(data.size(), 100);

    std::vector<size_t> labels(num_points);
    std::iota(labels.begin(), labels.end(), 0);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        alpha, max_degree, 2 * max_degree, 500, max_degree - 4, true};

    // Three-argument instantiation must still compile and deduce SequentialSync.
    auto index = svs::index::vamana::MultiMutableVamanaIndex(
        build_parameters, data, labels, Distance(), num_threads
    );

    CATCH_REQUIRE(index.size() == num_points);
    CATCH_REQUIRE(index.labelcount() == num_points);

    // Verify the deduced type is sequential by checking the full parent type.
    using IndexType = decltype(index);
    using ParentType = typename IndexType::ParentIndex;
    using ExpectedParentType = svs::index::vamana::MutableVamanaIndex<
        svs::graphs::SimpleBlockedGraph<uint32_t>,
        svs::data::SimpleData<Eltype, N>,
        Distance,
        svs::index::vamana::SequentialSync>;
    static_assert(
        std::is_same_v<ParentType, ExpectedParentType>,
        "Default instantiation must deduce SequentialSync"
    );
}

CATCH_TEST_CASE(
    "Multi: Concurrent add and search under SeqlockSync",
    "[index][vamana][multi][sync][concurrent]"
) {
    // Test concurrent add and search operations on a multi-value index with SeqlockSync.
    // This test is skipped for now because it depends on visitor work happening in parallel
    // on another branch (AR-11). Once that work is integrated, remove this skip.
    CATCH_SKIP("Depends on SeqlockVisitor routing (AR-11)");

    using Distance = svs::distance::DistanceL2;
    using Eltype = float;

#if defined(__SANITIZE_THREAD__) || defined(SVS_THREAD_SANITIZER)
    const size_t kPoints = 500;
    const size_t kDim = 16;
    const size_t kMaxDegree = 16;
    const size_t kNumQueries = 10;
    const size_t kAddBatch = 100;
#else
    const size_t kPoints = 2000;
    const size_t kDim = 32;
    const size_t kMaxDegree = 32;
    const size_t kNumQueries = 50;
    const size_t kAddBatch = 500;
#endif

    using ConcurrentGraphAlloc =
        svs::data::Blocked<svs::lib::Allocator<uint32_t>, svs::data::SegmentStable>;
    using ConcurrentGraphData =
        svs::data::SimpleData<uint32_t, svs::Dynamic, ConcurrentGraphAlloc>;
    using ConcurrentGraph = svs::graphs::
        SimpleGraphBase<uint32_t, ConcurrentGraphData, svs::graphs::SeqlockAccess>;
    using ConcurrentDataAlloc =
        svs::data::Blocked<svs::lib::Allocator<Eltype>, svs::data::SegmentStable>;
    using ConcurrentData = svs::data::SimpleData<Eltype, svs::Dynamic, ConcurrentDataAlloc>;

    using MultiConcurrent = svs::index::vamana::MultiMutableVamanaIndex<
        ConcurrentGraph,
        ConcurrentData,
        Distance,
        svs::index::vamana::SeqlockSync>;

    // Generate random vectors.
    std::mt19937 rng{42};
    std::normal_distribution<Eltype> dist{0.0f, 1.0f};
    auto make_vectors = [&](size_t n) {
        std::vector<Eltype> v(n * kDim);
        for (auto& x : v) {
            x = dist(rng);
        }
        return v;
    };

    auto base_raw = make_vectors(kPoints);
    auto add_raw = make_vectors(kAddBatch);
    auto query_raw = make_vectors(kNumQueries);

    // Build initial index.
    auto base_alloc = ConcurrentDataAlloc{};
    auto base_data = ConcurrentData(kPoints, kDim, base_alloc);
    for (size_t i = 0; i < kPoints; ++i) {
        base_data.set_datum(i, std::span<const Eltype>(base_raw.data() + i * kDim, kDim));
    }

    std::vector<size_t> labels(kPoints);
    std::iota(labels.begin(), labels.end(), 0);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        1.2f, kMaxDegree, 2 * kMaxDegree, 500, kMaxDegree, true};

    auto index = std::make_unique<MultiConcurrent>(
        build_parameters, std::move(base_data), labels, Distance{}, 4
    );

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();

    // Counters for results - atomics because Catch2 macros are not thread-safe.
    std::atomic<size_t> search_successes{0};
    std::atomic<size_t> search_attempts{0};
    std::atomic<size_t> add_successes{0};

    // Prepare addition batch.
    auto add_data = svs::data::SimpleData<Eltype>(kAddBatch, kDim);
    for (size_t i = 0; i < kAddBatch; ++i) {
        add_data.set_datum(i, std::span<const Eltype>(add_raw.data() + i * kDim, kDim));
    }
    std::vector<size_t> add_labels(kAddBatch);
    std::iota(add_labels.begin(), add_labels.end(), kPoints);

    // Launch concurrent operations.
    auto search_worker = std::thread([&]() {
        auto scratch = index->scratchspace(search_parameters);
        for (size_t i = 0; i < kNumQueries; ++i) {
            try {
                auto query = std::span<const Eltype>(query_raw.data() + i * kDim, kDim);
                index->search(query, scratch);
                ++search_successes;
            } catch (...) {
                // Exceptions are allowed during concurrent mutation.
            }
            ++search_attempts;
        }
    });

    auto add_worker = std::thread([&]() {
        try {
            index->add_points(add_data, add_labels);
            ++add_successes;
        } catch (...) {
            // Exceptions allowed.
        }
    });

    search_worker.join();
    add_worker.join();

    // Verify operations completed.
    CATCH_REQUIRE(search_attempts == kNumQueries);
    CATCH_REQUIRE(search_successes > 0);
    CATCH_REQUIRE(add_successes == 1);
}
