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

// Negative control for ThreadSanitizer: concurrent search over an index built with
// the unsafe (sequential) sync policy. TSan must flag data races here.

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/sync_policy.h"

#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace {

constexpr size_t kInitialPoints = 2000;
constexpr size_t kIncrementalPoints = 500;
constexpr size_t kDim = 32;
constexpr size_t kMaxDegree = 32;
constexpr size_t kBuildThreads = 8;

using Idx = uint32_t;
using Distance = svs::distance::DistanceL2;

// Unsafe configuration: plain (unprotected) graph access and SequentialSync.
// Concurrent searches over this policy combination race on adjacency reads.
using UnsafeGraphAlloc =
    svs::data::Blocked<svs::lib::Allocator<Idx>, svs::data::SegmentStable>;
using UnsafeGraphData = svs::data::SimpleData<Idx, svs::Dynamic, UnsafeGraphAlloc>;
using UnsafeGraph = svs::graphs::SimpleGraphBase<Idx, UnsafeGraphData>;
using UnsafeDataAlloc =
    svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>;
using UnsafeData = svs::data::SimpleData<float, svs::Dynamic, UnsafeDataAlloc>;
using UnsafeIndex = svs::index::vamana::MutableVamanaIndex<
    UnsafeGraph,
    UnsafeData,
    Distance,
    svs::index::vamana::SequentialSync>;

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> out(n * dim);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

std::unique_ptr<UnsafeIndex>
build_index(const std::vector<float>& raw, size_t dim, std::span<const size_t> ids) {
    const size_t n = raw.size() / dim;
    auto alloc = UnsafeDataAlloc{};
    auto data = UnsafeData(n, dim, alloc);
    for (size_t i = 0; i < n; ++i) {
        data.set_datum(i, std::span<const float>(raw.data() + i * dim, dim));
    }

    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2f, kMaxDegree, 2 * kMaxDegree, 750, kMaxDegree, true};
    return std::make_unique<UnsafeIndex>(
        parameters,
        std::move(data),
        std::vector<size_t>(ids.begin(), ids.end()),
        Distance{},
        kBuildThreads
    );
}

void add_batch(UnsafeIndex& index, const std::vector<float>& base, size_t first, size_t n) {
    auto batch = svs::data::SimpleData<float>(n, kDim);
    std::vector<size_t> batch_ids(n);
    for (size_t i = 0; i < n; ++i) {
        batch.set_datum(i, std::span<const float>(base.data() + (first + i) * kDim, kDim));
        batch_ids[i] = first + i;
    }
    index.add_points(batch, batch_ids);
}

} // namespace

CATCH_TEST_CASE("Unsafe concurrent index races under TSan", "[concurrent][tsan]") {
    const size_t total = kInitialPoints + kIncrementalPoints;
    auto base = random_vectors(total, kDim, 9999);

    std::vector<size_t> initial_ids(kInitialPoints);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(kInitialPoints * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(50, kDim, 7777);

    std::atomic<bool> writer_done{false};

    auto writer = [&] {
        constexpr size_t kBatch = 250;
        for (size_t offset = 0; offset < kIncrementalPoints; offset += kBatch) {
            const size_t n = std::min(kBatch, kIncrementalPoints - offset);
            add_batch(*index, base, kInitialPoints + offset, n);
        }
        writer_done.store(true);
    };

    auto searcher = [&] {
        auto scratch = index->scratchspace();
        while (!writer_done.load(std::memory_order_relaxed)) {
            for (size_t q = 0; q < 10; ++q) {
                auto query = std::span<const float>(queries_raw.data() + q * kDim, kDim);
                index->search(query, scratch);
            }
        }
    };

    std::thread w{writer};
    std::thread s1{searcher};
    std::thread s2{searcher};

    w.join();
    s1.join();
    s2.join();
}
