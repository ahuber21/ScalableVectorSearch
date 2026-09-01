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

// This file tests the concurrent index (SeqlockSync policy) with scenarios that might
// reveal integration issues: include order effects and quantized storage compatibility.
//
// The original PR #369 needed this to catch two defects only visible from a downstream
// consumer's include order. Our implementation doesn't have the shadow namespace that
// caused those issues, but we retain the scalar quantization test to verify that
// growth and compaction work with quantized datasets under the concurrent policy.

#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/multi.h"
#include "svs/index/vamana/sync_policy.h"

// svs
#include "svs/extensions/vamana/scalar.h"

// stl
#include <cstdint>
#include <numeric>
#include <vector>

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"

namespace {

const size_t num_threads = 2;

// A single-element dataset holding a copy of row `i`, for exercising add_points().
template <typename Data> auto one_point_from(const Data& data, size_t i) {
    auto point = svs::data::SimpleData<typename Data::element_type, svs::Dynamic>(
        1, data.dimensions()
    );
    point.set_datum(0, data.get_datum(i));
    return point;
}

} // namespace

CATCH_TEST_CASE(
    "SeqlockSync index over a scalar-quantized dataset grows and compacts",
    "[concurrent][downstream]"
) {
    using Idx = uint32_t;
    using Distance = svs::distance::DistanceL2;
    using SQAlloc =
        svs::data::Blocked<svs::lib::Allocator<std::int8_t>, svs::data::SegmentStable>;
    using SQData = svs::quantization::scalar::SQDataset<std::int8_t, svs::Dynamic, SQAlloc>;
    using SeqlockSync = svs::index::vamana::SeqlockSync;
    using SeqlockGraph = svs::graphs::SimpleGraphBase<
        Idx,
        svs::data::SimpleData<Idx, svs::Dynamic>,
        svs::graphs::SeqlockAccess>;
    using SeqlockIndex =
        svs::index::vamana::MutableVamanaIndex<SeqlockGraph, SQData, Distance, SeqlockSync>;

    auto base = test_dataset::data_f32();
    auto threadpool = svs::threads::DefaultThreadPool(num_threads);
    auto compressed = SQData::compress(base, threadpool, SQAlloc{});

    const size_t n = compressed.size();
    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index =
        SeqlockIndex(parameters, std::move(compressed), ids, Distance(), num_threads);
    CATCH_REQUIRE(index.size() == n);

    // Growth: no free slot exists yet, so this goes through data_.resize().
    auto point = one_point_from(base, 0);
    index.add_points(point, std::vector<size_t>{n + 100});
    CATCH_REQUIRE(index.has_id(n + 100));
    CATCH_REQUIRE(index.size() == n + 1);

    // Deletion: verify that delete_entries accepts the call.
    std::vector<size_t> to_delete(n / 10);
    std::iota(to_delete.begin(), to_delete.end(), 0);
    CATCH_REQUIRE(index.delete_entries(to_delete) == to_delete.size());

    // The index still answers queries after all of that.
    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), search_params);
    CATCH_REQUIRE(results.n_queries() == queries.size());
}
