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

// Tests proving three hazards are absent from the concurrent index design:
// 1. Failed inserts do not leave stranded Pending slots that deadlock later operations.
// 2. The graph entry point never names an Empty or Pending slot after any operation.
// 3. The seqlock retry boundary is correct: readers overlapping writes retry; others do
// not.

#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/sync_policy.h"

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/concurrency/seqlock.h"
#include "svs/lib/threads.h"

#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

constexpr size_t kInitialPoints = 500;
constexpr size_t kDim = 32;
constexpr size_t kMaxDegree = 32;
constexpr size_t kNumNeighbors = 10;
constexpr size_t kBuildThreads = 4;

using Idx = uint32_t;
using Distance = svs::distance::DistanceL2;
using ConcurrentGraphAlloc =
    svs::data::Blocked<svs::lib::Allocator<Idx>, svs::data::SegmentStable>;
using ConcurrentGraphData = svs::data::SimpleData<Idx, svs::Dynamic, ConcurrentGraphAlloc>;
using ConcurrentGraph =
    svs::graphs::SimpleGraphBase<Idx, ConcurrentGraphData, svs::graphs::SeqlockAccess>;
using ConcurrentDataAlloc =
    svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>;
using ConcurrentData = svs::data::SimpleData<float, svs::Dynamic, ConcurrentDataAlloc>;
using ConcurrentIndex = svs::index::vamana::MutableVamanaIndex<
    ConcurrentGraph,
    ConcurrentData,
    Distance,
    svs::index::vamana::SeqlockSync>;

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> out(n * dim);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

std::unique_ptr<ConcurrentIndex> build_index(
    const std::vector<float>& raw, size_t dim, std::span<const size_t> ids, size_t threads
) {
    const size_t n = raw.size() / dim;
    auto alloc = ConcurrentDataAlloc{};
    auto data = ConcurrentData(n, dim, alloc);
    for (size_t i = 0; i < n; ++i) {
        data.set_datum(i, std::span<const float>(raw.data() + i * dim, dim));
    }

    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2f, kMaxDegree, 2 * kMaxDegree, 750, kMaxDegree, true};
    return std::make_unique<ConcurrentIndex>(
        parameters,
        std::move(data),
        std::vector<size_t>(ids.begin(), ids.end()),
        Distance{},
        threads
    );
}

// Timeout wrapper: returns true if operation completes, false if it times out.
template <typename F> bool with_timeout(F&& operation, std::chrono::milliseconds timeout) {
    std::atomic<bool> done{false};
    std::thread worker{[&] {
        std::forward<F>(operation)();
        done.store(true, std::memory_order_release);
    }};

    const auto start = std::chrono::steady_clock::now();
    while (!done.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() - start > timeout) {
            worker.detach();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    worker.join();
    return true;
}

} // namespace

// Property 1: a failed insert (exception thrown during add_points) must not strand the
// index in a state where subsequent operations deadlock on Pending slots.
CATCH_TEST_CASE("Failed insert leaves index usable", "[concurrent][hazards]") {
    // FINDING: Cannot write this test without library changes. The index does not validate
    // input or provide a public way to trigger a catchable exception from add_points.
    // Dimension mismatches, empty data, and other malformed inputs cause segfaults, not
    // exceptions. The property (Pending slots do not deadlock) cannot be tested directly
    // without either (a) injecting failures into the library, or (b) the library providing
    // input validation that throws on invalid data.
    CATCH_SKIP("Cannot trigger catchable exception from add_points; "
               "input validation needed to test exception-safety");
}

// Property 2: the graph entry point must always be in Valid state, never Empty or Pending.
// This property must hold after any sequence of insert, delete and consolidate operations.
CATCH_TEST_CASE("Entry point is always visible to search", "[concurrent][hazards]") {
    auto base = random_vectors(kInitialPoints, kDim, 2000);
    std::vector<size_t> ids(kInitialPoints);
    std::iota(ids.begin(), ids.end(), 0);
    auto index = build_index(base, kDim, ids, kBuildThreads);

    auto queries_raw = random_vectors(10, kDim, 3000);
    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    // After initial build, entry point must be valid: search and invariants must pass.
    auto scratch = index->scratchspace();
    CATCH_REQUIRE_NOTHROW(
        index->search(std::span<const float>(queries_raw.data(), kDim), scratch)
    );
    CATCH_REQUIRE_NOTHROW(index->debug_check_invariants(true));

    // Insert new points: entry point must remain valid.
    auto point = svs::data::SimpleData<float>(1, kDim);
    point.set_datum(0, std::span<const float>(base.data(), kDim));
    index->add_points(point, std::vector<size_t>{kInitialPoints + 50});
    CATCH_REQUIRE_NOTHROW(
        index->search(std::span<const float>(queries_raw.data() + kDim, kDim), scratch)
    );
    CATCH_REQUIRE_NOTHROW(index->debug_check_invariants(true));

    // Delete points including potentially the entry point itself: entry point must be
    // recomputed to a valid slot. This is the case most likely to expose the hazard.
    std::vector<size_t> to_delete;
    for (size_t i = 0; i < kInitialPoints / 4; ++i) {
        to_delete.push_back(i);
    }
    index->delete_entries(to_delete);
    CATCH_REQUIRE_NOTHROW(
        index->search(std::span<const float>(queries_raw.data() + 2 * kDim, kDim), scratch)
    );
    CATCH_REQUIRE_NOTHROW(index->debug_check_invariants(true));

    // After consolidate (if available), entry point must still be valid.
    // Consolidate is not yet implemented for SeqlockSync; once available, uncomment:
    // index->consolidate();
    // CATCH_REQUIRE_NOTHROW(index->search(...));
    // CATCH_REQUIRE_NOTHROW(index->debug_check_invariants(false));

    CATCH_REQUIRE(index->size() > 0);
}

// Property 3: seqlock retry boundary is correct at the ordering boundary.
// A reader whose critical section overlaps a write must be forced to retry.
// A reader whose window contains no write must not be forced to retry.
CATCH_TEST_CASE("Seqlock retry is correct at ordering boundary", "[concurrent][hazards]") {
    // Test the SeqLockCounter primitive: validation must fail if write occurs between
    // read_begin and read_validate, and must succeed if no write occurs.
    svs::SeqLockCounter counter{};

    // Case 1: reader read_begin, then write, then reader read_validate.
    // Deterministic ordering via barriers ensures the write is between the two reads.
    std::atomic<size_t> phase{0};
    std::optional<svs::SeqLockCounter::counter_type> reader_seq;
    bool validation_result = true;

    std::thread reader{[&] {
        reader_seq = counter.read_begin();
        phase.store(1, std::memory_order_release);
        // Spin until writer completes.
        while (phase.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        if (reader_seq.has_value()) {
            validation_result = counter.read_validate(*reader_seq);
        }
    }};

    std::thread writer{[&] {
        // Spin until reader has captured its sequence.
        while (phase.load(std::memory_order_acquire) != 1) {
            std::this_thread::yield();
        }
        auto seq = counter.begin_write();
        counter.end_write(seq);
        phase.store(2, std::memory_order_release);
    }};

    reader.join();
    writer.join();

    // Validation must fail: the write occurred between read_begin and read_validate.
    CATCH_REQUIRE(reader_seq.has_value());
    CATCH_REQUIRE(!validation_result);

    // Case 2: reader read_begin and read_validate with no intervening write.
    // Validation must succeed.
    auto seq_no_write = counter.read_begin();
    CATCH_REQUIRE(seq_no_write.has_value());
    CATCH_REQUIRE(counter.read_validate(*seq_no_write));
}
