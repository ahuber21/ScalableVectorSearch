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

// header under test.
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/consolidate.h"

// stl
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

// svs
#include "svs/core/recall.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/dynamic_vamana.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// Verify that adding the Sync parameter is behaviour-preserving for the sequential path.
namespace {
using TestGraph = svs::graphs::SimpleBlockedGraph<uint32_t>;
using TestData = svs::data::SimpleData<float, svs::Dynamic>;
using TestDist = svs::distance::DistanceL2;

// The three-argument spelling should still work and should default to SequentialSync.
using ThreeArgIndex = svs::index::vamana::MutableVamanaIndex<TestGraph, TestData, TestDist>;
using ExplicitSeqIndex = svs::index::vamana::
    MutableVamanaIndex<TestGraph, TestData, TestDist, svs::index::vamana::SequentialSync>;
static_assert(std::is_same_v<ThreeArgIndex, ExplicitSeqIndex>);

// The mutex must be zero-sized for SequentialSync.
static_assert(std::is_empty_v<svs::index::vamana::SequentialSync::mutex_type>);
} // namespace

// The MutableVamanaIndex "Soft Deletion" test uses outdated API.
#if 0
namespace {
template <typename T> auto copy_dataset(const T& data) {
    auto copy = svs::data::SimplePolymorphicData<typename T::element_type, T::extent>{
        data.size(), data.dimensions()};
    for (size_t i = 0; i < data.size(); ++i) {
        copy.set_datum(i, data.get_datum(i));
    }
    return copy;
}

template <typename T, typename U> void check_results(const T& results, const U& deleted) {
    for (size_t i = 0; i < svs::getsize<0>(results); ++i) {
        for (size_t j = 0; j < svs::getsize<1>(results); ++j) {
            CATCH_REQUIRE(!deleted.contains(results.at(i, j)));
        }
    }
}

template <typename T, typename U>
void check_deleted(const T& index, const U& deleted, size_t imax) {
    for (size_t i = 0; i < imax; ++i) {
        if (deleted.contains(i)) {
            CATCH_REQUIRE(index.is_deleted(i));
        } else {
            CATCH_REQUIRE(!index.is_deleted(i));
        }
    }
}

template <typename Left, typename Right>
void check_equal(const Left& left, const Right& right) {
    CATCH_REQUIRE(left.size() == right.size());
    CATCH_REQUIRE(left.dimensions() == right.dimensions());

    for (size_t i = 0, imax = left.size(); i < imax; ++i) {
        const auto& datum_left = left.get_datum(i);
        const auto& datum_right = right.get_datum(i);
        CATCH_REQUIRE(std::equal(datum_left.begin(), datum_left.end(), datum_right.begin())
        );
    }
}

} // namespace

#if defined(NDEBUG)
const double DELETE_PERCENT = 0.3;
#else
const double DELETE_PERCENT = 0.05;
#endif

CATCH_TEST_CASE("MutableVamanaIndex", "[graph_index]") {
    const size_t num_threads = 2;
    const size_t num_neighbors = 10;

    const auto base_data = test_dataset::data_blocked_f32();
    // const auto base_data = test_dataset::data_f32();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::groundtruth_euclidean();

    CATCH_SECTION("Soft Deletion") {
        // In this section, we test soft deletion.
        // The idea is as follows:
        //
        // (1) Load the test index.
        // (2) Run a round of queries to ensure that everything loading correctly.
        // (3) Set a target deletion percentage where all the neighbors returned by
        //     all results returned by the previous query plus a random collection of extras
        //     are deleted.
        //
        // (4) Rerun queries, make sure accuracy is still high and that no deleted indices
        //     are present in the results.
        auto entry_point = svs::index::load_entry_point(test_dataset::metadata_file());

        auto index = svs::index::MutableVamanaIndex{
            test_dataset::graph_blocked(),
            base_data.copy(),
            entry_point,
            svs::distance::DistanceL2(),
            svs::threads::UnitRange<size_t>(0, base_data.size()),
            num_threads};

        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.set_search_window_size(num_neighbors);

        auto tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto original_time = svs::lib::time_difference(svs::lib::now(), tic);
        auto original_recall = svs::k_recall_at_n(groundtruth, results);
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::unordered_set<uint32_t> ids_to_delete{};
        double delete_percent = DELETE_PERCENT;
        for (size_t i = 0; i < groundtruth.size(); ++i) {
            auto slice = groundtruth.get_datum(i);
            for (size_t j = 0; j < num_neighbors; ++j) {
                auto id = slice[j];

                // For now - don't delete the entry point.
                if (id != entry_point) {
                    ids_to_delete.insert(slice[j]);
                }
            }

            if (ids_to_delete.size() > delete_percent * base_data.size()) {
                break;
            }
        }

        index.set_threadpool(threads::CppAsyncThreadPool(num_threads));

        std::cout << "Deleting " << ids_to_delete.size() << " entries!" << std::endl;
        index.delete_entries(ids_to_delete);
        check_deleted(index, ids_to_delete, base_data.size());
        index.debug_check_graph_consistency(true);
        CATCH_REQUIRE_THROWS_AS(
            index.debug_check_graph_consistency(false), svs::ANNException
        );
        CATCH_REQUIRE(index.entry_point() == entry_point);
        // Make sure the correct points were deleted.
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto new_time = svs::lib::time_difference(tic);

        // Make sure none of the returned results are in the deleted list.
        check_results(results.indices(), ids_to_delete);

        index.set_threadpool(threads::QueueThreadPoolWrapper(num_threads));

        auto results_reference = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.exhaustive_search(queries.view(), num_neighbors, results_reference.view());
        auto new_recall = svs::k_recall_at_n(results_reference.indices(), results);

        // Perform graph consolidation and see how the results are effected.
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_consolidate_time = svs::lib::time_difference(tic);
        auto post_consolidate_recall =
            svs::k_recall_at_n(results_reference.indices(), results);

        // Check deletion again.
        check_deleted(index, ids_to_delete, base_data.size());
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::cout << "Original recall: " << original_recall
                  << ", New Recall: " << new_recall
                  << ", Post Recall: " << post_consolidate_recall << std::endl;
        std::cout << "Original Time: " << original_time << " (s), New Time: " << new_time
                  << " (s) Post Time: " << post_consolidate_time << std::endl;
        CATCH_REQUIRE(new_recall > original_recall);
        check_results(results.indices(), ids_to_delete);

        // Now - delete the entry point and consolidate.
        ids_to_delete.insert(entry_point);
        std::vector<size_t> entry_point_vector{};
        entry_point_vector.push_back(entry_point);
        index.delete_entries(entry_point_vector);
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);

        auto& threadpool =
            index.get_threadpool_handle().get<threads::CppAsyncThreadPool>.get();
        threadpool.resize(3);
        CATCH_REQUIRE(index.get_num_threads() == 3);
        threadpool.resize(num_threads);
        CATCH_REQUIRE(index.get_num_threads() == num_threads);

        CATCH_REQUIRE(index.entry_point() != entry_point);
        index.search(queries.view(), num_neighbors, results.view());
        auto post_entrypoint_recall =
            svs::k_recall_at_n(results_reference.indices(), results);
        std::cout << "Post entry-point deletion recall: " << post_entrypoint_recall
                  << std::endl;

        // Add the deleted points back in.
        auto points = svs::data::SimpleData<float, svs::Dynamic>(
            ids_to_delete.size(), base_data.dimensions()
        );

        size_t i = 0;
        for (const auto& j : ids_to_delete) {
            points.set_datum(i, base_data.get_datum(j));
            ++i;
        }

        index.set_threadpool(threads::DefaultThreadPool(num_threads));
        tic = svs::lib::now();
        index.add_points(points, ids_to_delete);
        auto insert_time = svs::lib::time_difference(tic);
        std::cout << "Insertion took: " << insert_time << " seconds!" << std::endl;

        // Check that the stored dataset and the original dataset are equal.
        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_add_time = svs::lib::time_difference(tic);
        auto post_reinsertion_recall = svs::k_recall_at_n(groundtruth, results);
        std::cout << "Post reinsertion recall: " << post_reinsertion_recall << " in "
                  << post_add_time << " seconds." << std::endl;
    }
}
#endif

CATCH_TEST_CASE(
    "MutableVamana Index Save and Load", "[graph_index][dynamic_index][saveload]"
) {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;

    auto data = test_dataset::data_blocked_f32();
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );

    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), search_params);

    CATCH_SECTION("Load MutableVamana Index being serialized natively to stream") {
        std::stringstream stream;
        index.save(stream);
        {
            using Data_t = svs::data::BlockedData<float>;

            auto loaded = svs::DynamicVamana::assemble<float, Data_t>(
                stream, Distance(), num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(loaded.get_graph_max_degree() == index.get_graph_max_degree());
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            index.on_ids([&](size_t e) { CATCH_REQUIRE(loaded.has_id(e)); });

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < num_neighbors; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Load MutableVamana Index being serialized with intermediate files") {
        std::stringstream stream;
        {
            svs::lib::UniqueTempDirectory tempdir{"svs_dynvamana_save"};
            const auto config_dir = tempdir.get() / "config";
            const auto graph_dir = tempdir.get() / "graph";
            const auto data_dir = tempdir.get() / "data";
            std::filesystem::create_directories(config_dir);
            std::filesystem::create_directories(graph_dir);
            std::filesystem::create_directories(data_dir);
            index.save(config_dir, graph_dir, data_dir);
            svs::lib::DirectoryArchiver::pack(tempdir, stream);
        }
        {
            using Data_t = svs::data::BlockedData<float>;

            auto loaded = svs::DynamicVamana::assemble<float, Data_t>(
                stream, Distance(), num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(loaded.get_graph_max_degree() == index.get_graph_max_degree());
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            index.on_ids([&](size_t e) { CATCH_REQUIRE(loaded.has_id(e)); });

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < num_neighbors; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }
}

namespace {
template <typename Graph, typename Data, typename Sync> void test_locking_impl() {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;
    using Index = svs::index::vamana::MutableVamanaIndex<Graph, Data, Distance, Sync>;

    static_assert(std::is_same_v<
                  decltype(std::declval<const Index&>().lock_for_search()),
                  std::shared_lock<typename Sync::mutex_type>>);
    static_assert(std::is_same_v<
                  decltype(std::declval<const Index&>().lock_for_translation()),
                  std::shared_lock<typename Sync::mutex_type>>);

    auto source_data = test_dataset::data_f32();
    auto data = Data(source_data.size(), source_data.dimensions());
    for (size_t i = 0; i < source_data.size(); ++i) {
        data.set_datum(i, source_data.get_datum(i));
    }
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = Index(parameters, std::move(data), indices, Distance(), num_threads);

    {
        auto lock = index.lock_for_search();
        CATCH_REQUIRE(lock.owns_lock());
    }

    {
        auto lock = index.lock_for_translation();
        CATCH_REQUIRE(lock.owns_lock());
    }

    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto groundtruth = test_dataset::groundtruth_euclidean();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), search_params);

    auto recall = svs::k_recall_at_n(groundtruth, results, num_neighbors, num_neighbors);
    CATCH_REQUIRE(recall >= 0.25);

    auto query_span =
        std::span<const float>(queries.get_datum(0).data(), queries.dimensions());
    auto iterator = index.make_batch_iterator(query_span);
    CATCH_REQUIRE(iterator.batch_number() == 0);
    CATCH_REQUIRE(!iterator.done());

    iterator.next(5);
    CATCH_REQUIRE(iterator.size() <= 5);
    CATCH_REQUIRE(iterator.batch_number() == 1);

    std::vector<size_t> batch_ids;
    for (const auto& neighbor : iterator) {
        batch_ids.push_back(neighbor.id());
    }

    for (const auto& id : batch_ids) {
        CATCH_REQUIRE(index.has_id(id));
    }

    iterator.next(5);
    CATCH_REQUIRE(iterator.batch_number() == 2);
    bool made_progress = (iterator.size() > 0 && iterator.size() <= 5) || iterator.done();
    CATCH_REQUIRE(made_progress);
}
} // namespace

CATCH_TEST_CASE("MutableVamana Index Locking", "[index][vamana]") {
    using SeqSync = svs::index::vamana::SequentialSync;
    using SeqlockSync = svs::index::vamana::SeqlockSync;

    static_assert(std::is_empty_v<SeqSync::mutex_type>);
    static_assert(std::is_empty_v<svs::lib::NullMutex>);
    static_assert(!std::is_empty_v<SeqlockSync::mutex_type>);
    static_assert(!std::is_empty_v<svs::lib::MovableMutex<std::shared_mutex>>);

    CATCH_SECTION("SequentialSync") {
        using Graph = svs::graphs::SimpleBlockedGraph<uint32_t>;
        using Data = svs::data::BlockedData<float>;
        test_locking_impl<Graph, Data, SeqSync>();
    }

    CATCH_SECTION("SeqlockSync") {
        using Idx = uint32_t;
        using Graph = svs::graphs::SimpleGraphBase<
            Idx,
            svs::data::SimpleData<Idx, svs::Dynamic>,
            svs::graphs::SeqlockAccess>;
        using Data = svs::data::SimpleData<float, svs::Dynamic>;
        test_locking_impl<Graph, Data, SeqlockSync>();
    }
}

CATCH_TEST_CASE("MutableVamana Index Memory Usage", "[graph_index][dynamic_index]") {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;

    auto data = test_dataset::data_blocked_f32();
    const size_t data_size = data.size();
    // Expected data bytes are capacity-based; capture them before the dataset is moved
    // into the index so the test can pin the exact value.
    const size_t expected_data_bytes = data.capacity() * data.element_size();
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );

    const size_t expected_graph_bytes = index.view_graph().get_data().capacity() *
                                        index.view_graph().get_data().element_size();
    using Index = decltype(index);
    const size_t expected_metadata_bytes =
        data_size * sizeof(svs::index::vamana::SlotMetadata) +
        sizeof(typename Index::internal_id_type) +
        2 * indices.size() *
            (sizeof(typename Index::external_id_type) +
             sizeof(typename Index::internal_id_type));
    const size_t expected_total_bytes =
        expected_data_bytes + expected_graph_bytes + expected_metadata_bytes;

    // Dynamic get_memory_usage() should exactly match the capacity-based graph and data
    // bytes plus the deterministic metadata implied by the input ids.
    const auto breakdown = index.get_memory_breakdown();
    CATCH_REQUIRE(breakdown.graph_bytes == expected_graph_bytes);
    CATCH_REQUIRE(breakdown.data_bytes == expected_data_bytes);
    CATCH_REQUIRE(breakdown.metadata_bytes == expected_metadata_bytes);
    CATCH_REQUIRE(breakdown.total() == expected_total_bytes);
    const size_t usage = index.get_memory_breakdown().total();
    CATCH_REQUIRE(usage == expected_total_bytes);
}

CATCH_TEST_CASE(
    "MutableVamana Index SeqlockSync Instantiation", "[index][vamana][sync_policy]"
) {
    using Idx = uint32_t;
    using Distance = svs::distance::DistanceL2;
    using SharedGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::PlainAccess>;
    using SharedData = svs::data::SimpleData<float, svs::Dynamic>;

    using SeqSync = svs::index::vamana::SequentialSync;
    using SeqIndex =
        svs::index::vamana::MutableVamanaIndex<SharedGraph, SharedData, Distance, SeqSync>;

    using SeqlockSync = svs::index::vamana::SeqlockSync;
    using SeqlockGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>;
    using SeqlockIndex = svs::index::vamana::
        MutableVamanaIndex<SeqlockGraph, SharedData, Distance, SeqlockSync>;

    static_assert(std::is_move_constructible_v<SeqlockIndex>);

    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto groundtruth = test_dataset::groundtruth_euclidean();
    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};

    auto compare_results = [&](const char* description,
                               svs::QueryResult<size_t>& r1,
                               svs::QueryResult<size_t>& r2,
                               size_t max_differing_queries) {
        auto recall1 = svs::k_recall_at_n(groundtruth, r1, num_neighbors, num_neighbors);
        auto recall2 = svs::k_recall_at_n(groundtruth, r2, num_neighbors, num_neighbors);

        size_t differing_queries = 0;
        size_t total_mismatches = 0;
        for (size_t q = 0; q < queries.size(); ++q) {
            bool query_differs = false;
            for (size_t i = 0; i < num_neighbors; ++i) {
                if (r2.index(q, i) != r1.index(q, i)) {
                    ++total_mismatches;
                    query_differs = true;
                }
            }
            if (query_differs) {
                ++differing_queries;
            }
        }

        if (total_mismatches > 0) {
            CATCH_WARN(
                description << ": " << differing_queries << "/" << queries.size()
                            << " queries differ, " << total_mismatches
                            << " total mismatches. "
                            << "Recall1: " << recall1 << ", Recall2: " << recall2
            );
        }
        CATCH_REQUIRE(differing_queries <= max_differing_queries);
        CATCH_REQUIRE(std::abs(recall1 - recall2) <= 0.01);
    };

    CATCH_SECTION("Same-policy control: two SequentialSync builds") {
        auto data1 = test_dataset::data_f32();
        auto data2 = test_dataset::data_f32();
        const size_t initial_size = data1.size();
        std::vector<size_t> indices(initial_size);
        std::iota(indices.begin(), indices.end(), 0);

        auto index1 =
            SeqIndex(parameters, std::move(data1), indices, Distance(), size_t{2});
        auto index2 =
            SeqIndex(parameters, std::move(data2), indices, Distance(), size_t{2});

        auto results1 = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        auto results2 = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        index1.search(results1.view(), queries.cview(), search_params);
        index2.search(results2.view(), queries.cview(), search_params);

        compare_results(
            "Same-policy control (SequentialSync, num_threads=2)", results1, results2, 700
        );
    }

    CATCH_SECTION("Cross-policy single-threaded") {
        auto data_seq = test_dataset::data_f32();
        auto data_seqlock = test_dataset::data_f32();
        const size_t initial_size = data_seq.size();
        std::vector<size_t> indices(initial_size);
        std::iota(indices.begin(), indices.end(), 0);

        auto seq_index =
            SeqIndex(parameters, std::move(data_seq), indices, Distance(), size_t{1});
        auto seqlock_index = SeqlockIndex(
            parameters, std::move(data_seqlock), indices, Distance(), size_t{1}
        );

        auto seq_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        auto seqlock_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        seq_index.search(seq_results.view(), queries.cview(), search_params);
        seqlock_index.search(seqlock_results.view(), queries.cview(), search_params);

        compare_results(
            "Cross-policy (SequentialSync vs SeqlockSync, num_threads=1)",
            seq_results,
            seqlock_results,
            0
        );
    }

    CATCH_SECTION("Cross-policy multi-threaded") {
        auto data_seq = test_dataset::data_f32();
        auto data_seqlock = test_dataset::data_f32();
        const size_t initial_size = data_seq.size();
        std::vector<size_t> indices(initial_size);
        std::iota(indices.begin(), indices.end(), 0);

        auto seq_index =
            SeqIndex(parameters, std::move(data_seq), indices, Distance(), size_t{2});
        auto seqlock_index = SeqlockIndex(
            parameters, std::move(data_seqlock), indices, Distance(), size_t{2}
        );

        CATCH_REQUIRE(seq_index.size() == initial_size);
        CATCH_REQUIRE(seqlock_index.size() == initial_size);

        auto seq_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        auto seqlock_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        seq_index.search(seq_results.view(), queries.cview(), search_params);
        seqlock_index.search(seqlock_results.view(), queries.cview(), search_params);
        compare_results(
            "Cross-policy multi-threaded: Initial build", seq_results, seqlock_results, 700
        );

        const size_t num_to_add = 5;
        auto points_seq =
            svs::data::SimpleData<float, svs::Dynamic>(num_to_add, seq_index.dimensions());
        auto points_seqlock = svs::data::SimpleData<float, svs::Dynamic>(
            num_to_add, seqlock_index.dimensions()
        );
        for (size_t i = 0; i < num_to_add; ++i) {
            std::vector<float> datum(seq_index.dimensions());
            for (size_t j = 0; j < seq_index.dimensions(); ++j) {
                datum[j] = static_cast<float>(i + j);
            }
            points_seq.set_datum(i, datum);
            points_seqlock.set_datum(i, datum);
        }
        std::vector<size_t> new_ids(num_to_add);
        std::iota(new_ids.begin(), new_ids.end(), initial_size);

        seq_index.add_points(points_seq, new_ids);
        seqlock_index.add_points(points_seqlock, new_ids);

        CATCH_REQUIRE(seq_index.size() == initial_size + num_to_add);
        CATCH_REQUIRE(seqlock_index.size() == initial_size + num_to_add);

        for (const auto& id : new_ids) {
            CATCH_REQUIRE(seq_index.has_id(id));
            CATCH_REQUIRE(seqlock_index.has_id(id));
        }

        seq_index.search(seq_results.view(), queries.cview(), search_params);
        seqlock_index.search(seqlock_results.view(), queries.cview(), search_params);
        compare_results(
            "Cross-policy multi-threaded: After add", seq_results, seqlock_results, 700
        );

        std::vector<size_t> ids_to_delete{new_ids.begin(), new_ids.begin() + 3};
        seq_index.delete_entries(ids_to_delete);
        seqlock_index.delete_entries(ids_to_delete);

        for (const auto& id : ids_to_delete) {
            CATCH_REQUIRE(seq_index.is_deleted(id));
            CATCH_REQUIRE(seqlock_index.is_deleted(id));
        }

        seq_index.search(seq_results.view(), queries.cview(), search_params);
        seqlock_index.search(seqlock_results.view(), queries.cview(), search_params);
        compare_results(
            "Cross-policy multi-threaded: After delete", seq_results, seqlock_results, 700
        );

        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                const auto seqlock_id = seqlock_results.index(q, i);
                CATCH_REQUIRE(
                    std::find(ids_to_delete.begin(), ids_to_delete.end(), seqlock_id) ==
                    ids_to_delete.end()
                );
            }
        }
    }
}
