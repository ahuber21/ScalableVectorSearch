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

// Save/reload coverage for `MutableVamanaIndex` under `SeqlockSync`: delete, consolidate,
// compact, save, and reload, then check that the reloaded index is the same index.
//
// This is single-threaded by design -- it exercises the on-disk round trip, not concurrent
// access. `tests/svs/concurrent/concurrency.cpp` covers overlap between search and
// mutation.

// header under test
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/sync_policy.h"

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/misc.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

namespace {

using Idx = uint32_t;
using Distance = svs::distance::DistanceL2;
using SeqGraph = svs::graphs::SimpleBlockedGraph<Idx>;
// SeqlockSync requires address-stable growth: a resize must not relocate live data out from
// under a lock-free searcher.
using SharedData = svs::data::SimpleData<
    float,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>>;
using SeqlockSync = svs::index::vamana::SeqlockSync;
using SeqlockIndex =
    svs::index::vamana::MutableVamanaIndex<SeqGraph, SharedData, Distance, SeqlockSync>;

const size_t num_threads = 2;
const size_t num_neighbors = 10;

SharedData copy_to_grow_stable(const svs::data::SimpleData<float>& source) {
    SharedData dest(source.size(), source.dimensions());
    for (size_t i = 0; i < source.size(); ++i) {
        dest.set_datum(i, source.get_datum(i));
    }
    return dest;
}

} // namespace

CATCH_TEST_CASE(
    "SeqlockSync MutableVamanaIndex delete, consolidate, compact, save, reload",
    "[concurrent][persistence]"
) {
    auto source = test_dataset::data_f32();
    auto data = copy_to_grow_stable(source);
    const size_t initial_size = data.size();
    std::vector<size_t> indices(initial_size);
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index =
        SeqlockIndex(parameters, std::move(data), indices, Distance(), num_threads);
    CATCH_REQUIRE(index.size() == initial_size);

    // Delete every 10th id, consolidate to reroute in-edges, then compact to shrink
    // storage.
    std::vector<size_t> to_delete;
    for (size_t id = 0; id < initial_size; id += 10) {
        to_delete.push_back(id);
    }
    CATCH_REQUIRE(index.delete_entries(to_delete) == to_delete.size());
    index.consolidate();
    index.compact();

    const size_t expected_size = initial_size - to_delete.size();
    CATCH_REQUIRE(index.size() == expected_size);

    std::unordered_set<size_t> deleted(to_delete.begin(), to_delete.end());
    std::unordered_set<size_t> expected_survivors;
    for (size_t id = 0; id < initial_size; ++id) {
        if (!deleted.contains(id)) {
            expected_survivors.insert(id);
        }
    }
    std::unordered_set<size_t> survivors;
    index.on_ids([&survivors](size_t id) { survivors.insert(id); });
    CATCH_REQUIRE(survivors == expected_survivors);

    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};

    auto original_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(original_results.view(), queries.cview(), search_params);

    svs_test::prepare_temp_directory();
    auto tmp = svs_test::temp_directory();
    index.save(tmp / "config", tmp / "graph", tmp / "data");

    auto graph_loader = SVS_LAZY(SeqGraph::load(tmp / "graph"));
    auto data_loader = SVS_LAZY(SharedData::load(tmp / "data"));
    auto reloaded = svs::index::vamana::auto_dynamic_assemble<
        decltype(graph_loader),
        decltype(data_loader),
        Distance,
        size_t,
        SeqlockSync>(
        tmp / "config",
        std::move(graph_loader),
        std::move(data_loader),
        Distance(),
        num_threads
    );

    CATCH_REQUIRE(reloaded.size() == expected_size);
    CATCH_REQUIRE(reloaded.dimensions() == index.dimensions());

    std::unordered_set<size_t> reloaded_survivors;
    reloaded.on_ids([&reloaded_survivors](size_t id) { reloaded_survivors.insert(id); });
    CATCH_REQUIRE(reloaded_survivors == expected_survivors);

    auto reloaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    reloaded.search(reloaded_results.view(), queries.cview(), search_params);

    // Same graph, same data, same deterministic single-threaded search: the round trip
    // through disk must reproduce identical neighbor lists, not merely similar ones.
    size_t differing = 0;
    for (size_t q = 0; q < queries.size(); ++q) {
        for (size_t i = 0; i < num_neighbors; ++i) {
            if (reloaded_results.index(q, i) != original_results.index(q, i)) {
                ++differing;
            }
        }
    }
    CATCH_INFO("Mismatches: " << differing);
    CATCH_REQUIRE(differing == 0);
}
