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

// Concurrency tests for ``svs::index::vamana::MutableVamanaIndex`` with
// ``SeqlockSync``: correctness of searches issued while other threads insert, delete,
// and consolidate.
//
// The rest of the suite (``dynamic_index.cpp``, ``dynamic_index_2.cpp``, ``multi.cpp``,
// ``iterator.cpp``) covers single-threaded functional parity with the pre-existing dynamic
// index. This file covers the property that motivates the concurrent synchronization
// policy: searches and mutations may overlap in time.
//
// Recall thresholds here are deliberately coarse. They exist to catch a graph that has been
// corrupted into uselessness, not to track search quality -- that is the job of the
// benchmark suite.
//
// Assertions inside the hot loops accumulate into counters and are checked once at the end.
// Catch2's assertion bookkeeping is not free, and these loops run for millions of
// iterations.

// header under test
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/sync_policy.h"

// For the definition of the `BatchIterator` that `make_batch_iterator` returns.
#include "svs/index/vamana/iterator.h"

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// ThreadSanitizer costs roughly an order of magnitude in both time and memory, and it is
// looking for *races*, which show up just as readily in a small index. Shrink the problem
// rather than skipping the run.
#if defined(__SANITIZE_THREAD__) || defined(SVS_THREAD_SANITIZER)
constexpr size_t kInitialPoints = 4000;
constexpr size_t kIncrementalPoints = 1000;
constexpr size_t kNumQueries = 50;
#else
constexpr size_t kInitialPoints = 20000;
constexpr size_t kIncrementalPoints = 5000;
constexpr size_t kNumQueries = 200;
#endif

constexpr size_t kDim = 32;
constexpr size_t kMaxDegree = 32;
constexpr size_t kNumNeighbors = 10;
constexpr size_t kBuildThreads = 8;

using Idx = uint32_t;
using Distance = svs::distance::DistanceL2;
// Adjacency storage must be grow-stable: concurrent searches hold spans into it without
// locks, so resize must not relocate existing elements or searches segfault.
using ConcurrentGraphAlloc =
    svs::data::Blocked<svs::lib::Allocator<Idx>, svs::data::SegmentStable>;
using ConcurrentGraphData = svs::data::SimpleData<Idx, svs::Dynamic, ConcurrentGraphAlloc>;
using ConcurrentGraph =
    svs::graphs::SimpleGraphBase<Idx, ConcurrentGraphData, svs::graphs::SeqlockAccess>;
// Concurrent searches hold dataset pointers without taking locks, so the dataset must
// provide address-stable storage. A reallocating dataset causes use-after-free.
using ConcurrentDataAlloc =
    svs::data::Blocked<svs::lib::Allocator<float>, svs::data::SegmentStable>;
using ConcurrentData = svs::data::SimpleData<float, svs::Dynamic, ConcurrentDataAlloc>;
using ConcurrentIndex = svs::index::vamana::MutableVamanaIndex<
    ConcurrentGraph,
    ConcurrentData,
    Distance,
    svs::index::vamana::SeqlockSync>;

static_assert(svs::data::is_grow_stable_v<ConcurrentDataAlloc>);
static_assert(svs::data::is_dataset_grow_stable_v<ConcurrentData>);
static_assert(svs::data::is_grow_stable_v<ConcurrentGraphAlloc>);
static_assert(svs::data::is_dataset_grow_stable_v<ConcurrentGraphData>);

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> out(n * dim);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

svs::data::SimpleData<float> make_dataset(const std::vector<float>& raw, size_t dim) {
    const size_t n = raw.size() / dim;
    auto data = svs::data::SimpleData<float>(n, dim);
    for (size_t i = 0; i < n; ++i) {
        data.set_datum(i, std::span<const float>(raw.data() + i * dim, dim));
    }
    return data;
}

// Build a concurrent index over ``raw`` through the ordinary build constructor.
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

// Brute-force ground truth over the given set of live external IDs.
std::vector<std::vector<size_t>> ground_truth(
    const std::vector<float>& base,
    const std::unordered_set<size_t>& live,
    const std::vector<float>& queries,
    size_t dim,
    size_t k
) {
    const size_t nq = queries.size() / dim;
    std::vector<std::vector<size_t>> result(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, size_t>> scored;
        scored.reserve(live.size());
        for (size_t id : live) {
            float d = 0;
            for (size_t j = 0; j < dim; ++j) {
                float diff = queries[q * dim + j] - base[id * dim + j];
                d += diff * diff;
            }
            scored.emplace_back(d, id);
        }
        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<long>(std::min(k, scored.size())),
            scored.end()
        );
        for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
            result[q].push_back(scored[i].second);
        }
    }
    return result;
}

double recall_at_k(
    const svs::QueryResult<size_t>& got, const std::vector<std::vector<size_t>>& expected
) {
    size_t hits = 0, total = 0;
    for (size_t q = 0; q < expected.size(); ++q) {
        std::unordered_set<size_t> truth{expected[q].begin(), expected[q].end()};
        for (size_t j = 0; j < got.n_neighbors(); ++j) {
            if (truth.count(got.index(q, j))) {
                ++hits;
            }
        }
        total += truth.size();
    }
    return total == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(total);
}

// Insert ``[first, first + n)`` of ``base`` as a single batch.
void add_batch(
    ConcurrentIndex& index, const std::vector<float>& base, size_t first, size_t n
) {
    auto batch = svs::data::SimpleData<float>(n, kDim);
    std::vector<size_t> batch_ids(n);
    for (size_t i = 0; i < n; ++i) {
        batch.set_datum(i, std::span<const float>(base.data() + (first + i) * kDim, kDim));
        batch_ids[i] = first + i;
    }
    index.add_points(batch, batch_ids);
}

} // namespace

CATCH_TEST_CASE("Concurrent MutableVamanaIndex quiescent recall", "[concurrent][index]") {
    auto base = random_vectors(kInitialPoints, kDim, 1234);
    std::vector<size_t> ids(kInitialPoints);
    std::iota(ids.begin(), ids.end(), 0);

    auto index = build_index(base, kDim, ids, kBuildThreads);
    CATCH_REQUIRE(index->size() == kInitialPoints);

    auto queries_raw = random_vectors(kNumQueries, kDim, 999);
    auto queries = make_dataset(queries_raw, kDim);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto results = svs::QueryResult<size_t>{kNumQueries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());

    std::unordered_set<size_t> live{ids.begin(), ids.end()};
    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    const double recall = recall_at_k(results, truth);

    // A correctly built Vamana graph at this window size should be well above 0.9.
    CATCH_INFO("quiescent recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.90);
}

// The core test: searches run continuously while writers insert new vectors and delete
// existing ones. Any torn adjacency read, use-after-free from a resize, or missing ID
// translation surfaces as a crash, an exception, or an inconsistent ID round-trip.
CATCH_TEST_CASE(
    "Concurrent MutableVamanaIndex search during mutation", "[concurrent][index]"
) {
    auto envnum = [](const char* name, size_t fallback) {
        const char* v = std::getenv(name);
        return v != nullptr ? static_cast<size_t>(std::strtoul(v, nullptr, 10)) : fallback;
    };
    const size_t local_initial = envnum("SVS_MUT_INITIAL", 200);
    const size_t local_incremental = envnum("SVS_MUT_INCR", 50);
    const size_t local_queries = envnum("SVS_MUT_QUERIES", 10);
    const size_t local_batch = envnum("SVS_MUT_BATCH", 25);
    const size_t local_writers = envnum("SVS_MUT_WRITERS", 2);
    const int local_searchers = static_cast<int>(envnum("SVS_MUT_SEARCHERS", 2));

    const size_t total = local_initial + local_incremental;
    auto base = random_vectors(total, kDim, 4321);

    std::vector<size_t> initial_ids(local_initial);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    // Build over the first ``local_initial`` only; the tail is inserted concurrently
    // below.
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(local_initial * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(local_queries, kDim, 777);
    auto queries = make_dataset(queries_raw, kDim);

    std::atomic<size_t> writers_running{0};
    std::atomic<size_t> searches_completed{0};
    std::atomic<size_t> bad_roundtrips{0};
    std::atomic<size_t> duplicate_ids{0};
    std::atomic<size_t> exceptions{0};
    // Guards exception_log; appended only from catch blocks below, read only after joins.
    std::mutex exception_log_mutex;
    std::vector<std::string> exception_log;

    const size_t per_writer = local_incremental / local_writers;

    auto writer = [&](size_t w) {
        size_t offset = 0;
        try {
            const size_t begin = local_initial + w * per_writer;
            for (; offset < per_writer; offset += local_batch) {
                const size_t n = std::min(local_batch, per_writer - offset);
                add_batch(*index, base, begin + offset, n);

                // Delete a fresh, disjoint set of the original IDs each round, *spread*
                // across the whole ID range with a stride. A contiguous low-ID slice would
                // almost never intersect a query's top-k, so the interesting race -- a
                // result slot retired between selection and ID translation -- would go
                // unexercised. Each (writer, round) pair takes its own residue class mod
                // 40, so the rounds are disjoint and together retire a quarter of the
                // original vectors.
                const size_t round = w * (per_writer / local_batch) + offset / local_batch;
                std::vector<size_t> to_delete;
                for (size_t id = round; id < local_initial; id += 40) {
                    to_delete.push_back(id);
                }
                index->delete_entries(to_delete);
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "writer " + std::to_string(w) + " offset=" + std::to_string(offset) + ": " +
                e.what()
            );
        }
        writers_running.fetch_sub(1);
    };

    // Searchers: hammer the index with single-query searches throughout.
    auto searcher = [&] {
        size_t q = 0;
        try {
            auto scratch = index->scratchspace();
            std::unordered_set<Idx> seen_ids;
            while (writers_running.load(std::memory_order_relaxed) != 0) {
                for (q = 0; q < local_queries; ++q) {
                    auto query =
                        std::span<const float>(queries_raw.data() + q * kDim, kDim);
                    index->search(query, scratch);
                    // ``[0, valid())`` is the region a caller is allowed to read: skipped
                    // (deleted) candidates have been compacted out by this point.
                    const size_t n =
                        std::min<size_t>(kNumNeighbors, scratch.buffer.valid());
                    seen_ids.clear();
                    for (size_t j = 0; j < n; ++j) {
                        auto internal = scratch.buffer[j].id();
                        // A ranked result must never list the same vector twice. Torn
                        // adjacency reads or a botched retry in the seqlock section would
                        // show up here, because the search buffer dedupes by ID and can
                        // only be fooled by inconsistent input.
                        if (!seen_ids.insert(internal).second) {
                            duplicate_ids.fetch_add(1, std::memory_order_relaxed);
                        }
                        // If an ID is still live, its mapping must round-trip exactly.
                        // translate_internal_id throws if the internal ID was retired
                        // concurrently (IDTranslator::get_external uses .at() which throws
                        // on missing keys, per translation.h). That is an expected outcome
                        // of the test's deliberate concurrent deletion, not a failure.
                        try {
                            auto external = index->translate_internal_id(internal);
                            if (index->has_id(external) &&
                                index->translate_external_id(external) != internal) {
                                bad_roundtrips.fetch_add(1, std::memory_order_relaxed);
                            }
                        } catch (const std::out_of_range&) {
                            // Internal ID retired concurrently — expected, not an error.
                        }
                    }
                    searches_completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "searcher query=" + std::to_string(q) + " completed=" +
                std::to_string(searches_completed.load(std::memory_order_relaxed)) + ": " +
                e.what()
            );
        }
    };

    std::vector<std::thread> threads;
    writers_running.store(local_writers);
    for (size_t w = 0; w < local_writers; ++w) {
        threads.emplace_back(writer, w);
    }
    for (int i = 0; i < local_searchers; ++i) {
        threads.emplace_back(searcher);
    }
    for (auto& t : threads) {
        t.join();
    }

    CATCH_INFO("searches completed: " << searches_completed.load());
    // No lock needed: all threads that could append are joined above.
    std::string exception_report;
    for (const auto& msg : exception_log) {
        exception_report += msg + "; ";
    }
    CATCH_INFO("captured exceptions: " << exception_report);
    CATCH_REQUIRE(exceptions.load() == 0);
    CATCH_REQUIRE(bad_roundtrips.load() == 0);
    CATCH_REQUIRE(duplicate_ids.load() == 0);
    CATCH_REQUIRE(searches_completed.load() > 0);

    // Post-mutation the index must still be a correct index.
    index->debug_check_invariants(true);
    std::unordered_set<size_t> live;
    index->on_ids([&live](size_t id) { live.insert(id); });
    CATCH_REQUIRE(live.size() == index->size());

    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    auto results = svs::QueryResult<size_t>{local_queries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());
    const double recall = recall_at_k(results, truth);
    CATCH_INFO("post-mutation recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.85);
}

// `consolidate()` walks the reverse-edge index and rewires in-neighbors of deleted slots in
// place. Unlike `compact()` it does not shrink storage, so it is allowed to run while
// searches are in flight. That is the property under test here.
CATCH_TEST_CASE(
    "Concurrent MutableVamanaIndex consolidate during search", "[concurrent][index]"
) {
    auto base = random_vectors(kInitialPoints, kDim, 8642);
    std::vector<size_t> ids(kInitialPoints);
    std::iota(ids.begin(), ids.end(), 0);
    auto index = build_index(base, kDim, ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(kNumQueries, kDim, 555);

    std::atomic<bool> writer_done{false};
    std::atomic<size_t> searches_completed{0};
    std::atomic<size_t> exceptions{0};

    std::thread writer{[&] {
        try {
            // Five rounds of "delete a stride, then consolidate", retiring 25% overall.
            for (size_t round = 0; round < 5; ++round) {
                std::vector<size_t> to_delete;
                for (size_t id = round; id < kInitialPoints; id += 20) {
                    to_delete.push_back(id);
                }
                index->delete_entries(to_delete);
                index->consolidate();
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
        }
        writer_done.store(true);
    }};

    auto searcher = [&] {
        try {
            auto scratch = index->scratchspace();
            while (!writer_done.load(std::memory_order_relaxed)) {
                for (size_t q = 0; q < kNumQueries; ++q) {
                    auto query =
                        std::span<const float>(queries_raw.data() + q * kDim, kDim);
                    index->search(query, scratch);
                    searches_completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> searchers;
    for (int i = 0; i < 6; ++i) {
        searchers.emplace_back(searcher);
    }
    writer.join();
    for (auto& t : searchers) {
        t.join();
    }

    CATCH_INFO("searches completed: " << searches_completed.load());
    CATCH_REQUIRE(exceptions.load() == 0);
    CATCH_REQUIRE(searches_completed.load() > 0);
    index->debug_check_invariants(false);

    // The surviving 75% must still be reachable at a reasonable rate.
    std::unordered_set<size_t> live;
    index->on_ids([&live](size_t id) { live.insert(id); });
    CATCH_REQUIRE(live.size() == index->size());

    auto queries = make_dataset(queries_raw, kDim);
    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    auto results = svs::QueryResult<size_t>{kNumQueries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());
    const double recall = recall_at_k(results, truth);
    CATCH_INFO("post-consolidate recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.85);
}

// The batch iterator holds a cursor across calls. This checks it works at all, and that it
// keeps working while a writer mutates the index.
CATCH_TEST_CASE("Concurrent MutableVamanaIndex batch iterator", "[concurrent][index]") {
    const size_t total = kInitialPoints + kIncrementalPoints;
    auto base = random_vectors(total, kDim, 24680);

    std::vector<size_t> initial_ids(kInitialPoints);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(kInitialPoints * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto queries_raw = random_vectors(kNumQueries, kDim, 13579);

    CATCH_SECTION("quiescent") {
        // Batches must be non-overlapping, and each batch must be sorted.
        //
        // Batches are *not* globally monotonic and it would be wrong to assert that: the
        // iterator is approximate, so a later batch can surface a vector closer than one
        // the earlier batch's window had already returned. Count those inversions and
        // report them as a quality signal rather than a correctness one -- this is
        // pre-existing behaviour and has nothing to do with concurrency.
        auto query = std::span<const float>(queries_raw.data(), kDim);
        auto it = index->make_batch_iterator(query);
        std::unordered_set<size_t> all;
        float previous_worst = -1.0f;
        size_t batches = 0;
        size_t inversions = 0;
        size_t repeats = 0;
        size_t unsorted = 0;
        for (; batches < 5 && !it.done(); ++batches) {
            it.next(10);
            float last = -1.0f;
            for (const auto& n : it) {
                if (!all.insert(n.id()).second) {
                    ++repeats;
                }
                if (n.distance() < last) {
                    ++unsorted;
                }
                last = n.distance();
                if (n.distance() < previous_worst) {
                    ++inversions;
                }
            }
            if (it.size() != 0) {
                previous_worst = (it.end() - 1)->distance();
            }
        }
        CATCH_INFO(
            batches << " batches, " << all.size() << " distinct vectors, " << inversions
                    << " cross-batch inversions"
        );
        CATCH_REQUIRE(repeats == 0);
        CATCH_REQUIRE(unsorted == 0);
        CATCH_REQUIRE(all.size() >= 40);
    }

    CATCH_SECTION("during mutation") {
        // The claim under test is memory safety and per-batch consistency, *not* that a
        // long-lived cursor sees a stable snapshot.
        std::atomic<bool> writer_done{false};
        std::atomic<size_t> batches_completed{0};
        std::atomic<size_t> exceptions{0};
        std::atomic<size_t> bad_ids{0};

        std::thread writer{[&] {
            try {
                constexpr size_t kBatch = 500;
                for (size_t offset = 0; offset < kIncrementalPoints; offset += kBatch) {
                    const size_t n = std::min(kBatch, kIncrementalPoints - offset);
                    add_batch(*index, base, kInitialPoints + offset, n);
                }
            } catch (const std::exception& e) {
                exceptions.fetch_add(1, std::memory_order_relaxed);
            }
            writer_done.store(true);
        }};

        auto reader = [&] {
            try {
                while (!writer_done.load(std::memory_order_relaxed)) {
                    for (size_t q = 0; q < kNumQueries; ++q) {
                        auto query =
                            std::span<const float>(queries_raw.data() + q * kDim, kDim);
                        auto it = index->make_batch_iterator(query);
                        for (size_t b = 0; b < 3 && !it.done(); ++b) {
                            it.next(10);
                            for (const auto& n : it) {
                                // Any ID the iterator yields must name a slot that either
                                // is live or was retired mid-flight; a bad graph read shows
                                // up as an ID no larger than the highest slot ever handed
                                // out.
                                if (n.id() >= total) {
                                    bad_ids.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                            batches_completed.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            } catch (const std::exception& e) {
                exceptions.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> readers;
        for (int i = 0; i < 4; ++i) {
            readers.emplace_back(reader);
        }
        writer.join();
        for (auto& t : readers) {
            t.join();
        }

        CATCH_INFO("batches completed during mutation: " << batches_completed.load());
        CATCH_REQUIRE(exceptions.load() == 0);
        CATCH_REQUIRE(bad_ids.load() == 0);
        CATCH_REQUIRE(batches_completed.load() > 0);
    }
}

// Reproduce the vertex lock array bounds violation: two concurrent writers, graph size
// crossing the builder's captured array size. Before the fix, VamanaBuilder sized its lock
// array from data_.size() at construction; a second writer growing the graph between that
// capture and add_reverse_edges yielded an out-of-range index into vertex_locks_. The
// out-of-range element reinterpreted heap garbage as std::atomic<bool>; when non-zero, the
// SpinLock hung. With the bounds assertion enabled, this aborts instead of hanging.
//
// At 2000/500 (initial/increment), the defect is deterministic: instrumented measurement
// showed 5 of 5 aborts within 3 seconds, and without instrumentation 5 of 5 hangs beyond
// 120 seconds. The fix inverts the dependency so the builder takes the lock array by
// reference from its caller, and under SeqlockSync the array is grown alongside the graph
// inside slot_alloc_mutex_.
CATCH_TEST_CASE(
    "Vertex lock array covers concurrent graph growth", "[concurrent][index][vamana]"
) {
    constexpr size_t kInitial = 2000;
    constexpr size_t kIncrement = 500;
    constexpr size_t kPerWriter = kIncrement / 2;
    constexpr size_t kBatch = 25;

    auto base = random_vectors(kInitial + kIncrement, kDim, 9876);
    std::vector<size_t> initial_ids(kInitial);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);

    auto initial_slice =
        std::vector<float>(base.begin(), base.begin() + static_cast<long>(kInitial * kDim));
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    std::atomic<size_t> exceptions{0};
    std::atomic<size_t> completed{0};

    auto writer = [&](size_t w) {
        try {
            const size_t begin = kInitial + w * kPerWriter;
            for (size_t offset = 0; offset < kPerWriter; offset += kBatch) {
                const size_t n = std::min(kBatch, kPerWriter - offset);
                add_batch(*index, base, begin + offset, n);
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::thread w0(writer, 0);
    std::thread w1(writer, 1);
    w0.join();
    w1.join();

    CATCH_REQUIRE(exceptions.load() == 0);
    CATCH_REQUIRE(completed.load() == 2);
    CATCH_REQUIRE(index->size() == kInitial + kIncrement);
}

// Footprint measurement for verification
CATCH_TEST_CASE("Report MutableVamanaIndex footprint", "[.][footprint]") {
    using namespace svs;
    using Index = index::vamana::MutableVamanaIndex<
        graphs::SimpleGraph<uint32_t>,
        data::SimpleData<float>,
        distance::DistanceL2,
        index::vamana::SequentialSync>;
    CATCH_INFO("sizeof(MutableVamanaIndex<SequentialSync>)=" << sizeof(Index));
    CATCH_REQUIRE(true);
}

// AR-20 excludes add_points()/consolidate() from compact() via compact_mutex_; without it,
// compact() can resize data_/graph_ underneath an in-flight add_points.
CATCH_TEST_CASE(
    "Concurrent MutableVamanaIndex compact during mutation", "[concurrent][index]"
) {
    auto envnum = [](const char* name, size_t fallback) {
        const char* v = std::getenv(name);
        return v != nullptr ? static_cast<size_t>(std::strtoul(v, nullptr, 10)) : fallback;
    };
    const size_t local_initial = envnum("SVS_CMP_INITIAL", 200);
    const size_t local_incremental = envnum("SVS_CMP_INCR", 10);
    const size_t local_queries = envnum("SVS_CMP_QUERIES", 10);
    const size_t local_writers = envnum("SVS_CMP_WRITERS", 1);
    const int local_searchers = static_cast<int>(envnum("SVS_CMP_SEARCHERS", 1));
    const size_t local_compacts = envnum("SVS_CMP_COMPACTS", 2);
    constexpr size_t local_batch = 10;

    const size_t total = local_initial + local_incremental;
    auto base = random_vectors(total, kDim, 2468);

    std::vector<size_t> initial_ids(local_initial);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(local_initial * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(local_queries, kDim, 8642);
    auto queries = make_dataset(queries_raw, kDim);

    // Reclaim deleted slots once, before workers start: add_points can wire edges into
    // Deleted nodes, so racing it against consolidate()/compact() risks an unrelated throw.
    std::vector<size_t> to_delete;
    for (size_t id = 0; id < local_initial; id += 4) {
        to_delete.push_back(id);
    }
    index->delete_entries(to_delete);
    index->consolidate();
    index->compact();

    std::atomic<size_t> writers_running{0};
    std::atomic<size_t> compactor_running{0};
    std::atomic<size_t> searches_completed{0};
    std::atomic<size_t> compacts_completed{0};
    std::atomic<size_t> exceptions{0};
    // Guards exception_log; appended only from catch blocks below, read only after joins.
    std::mutex exception_log_mutex;
    std::vector<std::string> exception_log;

    const size_t per_writer = local_incremental / local_writers;

    auto writer = [&](size_t w) {
        size_t offset = 0;
        try {
            const size_t begin = local_initial + w * per_writer;
            for (; offset < per_writer; offset += local_batch) {
                const size_t n = std::min(local_batch, per_writer - offset);
                add_batch(*index, base, begin + offset, n);
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "writer " + std::to_string(w) + " offset=" + std::to_string(offset) + ": " +
                e.what()
            );
        }
        writers_running.fetch_sub(1);
    };

    // No further deletions occur, so each compact() below is an identity remap; it still
    // races add_points growing data_/graph_ against a size snapshot taken at its own start.
    auto compactor = [&] {
        size_t c = 0;
        try {
            for (; c < local_compacts; ++c) {
                index->compact();
                compacts_completed.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "compactor round=" + std::to_string(c) + ": " + e.what()
            );
        }
        compactor_running.store(0);
    };

    auto searcher = [&] {
        size_t q = 0;
        try {
            auto scratch = index->scratchspace();
            while (writers_running.load(std::memory_order_relaxed) != 0 ||
                   compactor_running.load(std::memory_order_relaxed) != 0) {
                for (q = 0; q < local_queries; ++q) {
                    auto query =
                        std::span<const float>(queries_raw.data() + q * kDim, kDim);
                    index->search(query, scratch);
                    searches_completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "searcher query=" + std::to_string(q) + " completed=" +
                std::to_string(searches_completed.load(std::memory_order_relaxed)) + ": " +
                e.what()
            );
        }
    };

    std::vector<std::thread> threads;
    writers_running.store(local_writers);
    compactor_running.store(1);
    for (size_t w = 0; w < local_writers; ++w) {
        threads.emplace_back(writer, w);
    }
    threads.emplace_back(compactor);
    for (int i = 0; i < local_searchers; ++i) {
        threads.emplace_back(searcher);
    }
    for (auto& t : threads) {
        t.join();
    }

    CATCH_INFO("searches completed: " << searches_completed.load());
    CATCH_INFO("compact rounds completed: " << compacts_completed.load());
    // No lock needed: all threads that could append are joined above.
    std::string exception_report;
    for (const auto& msg : exception_log) {
        exception_report += msg + "; ";
    }
    CATCH_INFO("captured exceptions: " << exception_report);
    CATCH_REQUIRE(exceptions.load() == 0);
    CATCH_REQUIRE(searches_completed.load() > 0);
    CATCH_REQUIRE(compacts_completed.load() > 0);

    // Post-mutation the index must still be a correct index.
    index->debug_check_invariants(true);
    std::unordered_set<size_t> live;
    index->on_ids([&live](size_t id) { live.insert(id); });
    CATCH_REQUIRE(live.size() == index->size());

    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    auto results = svs::QueryResult<size_t>{local_queries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());
    const double recall = recall_at_k(results, truth);
    CATCH_INFO("post-compact recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.85);
}

// VamanaBuilder's own greedy_search can select an already-Deleted node as a neighbor and
// wire an edge to it. If consolidate() has already rerouted that node's in-edges, the new
// edge survives consolidate() and only surfaces later, when compact() removes the slot.
// Unlike the "compact during mutation" test above, deletions are *not* reclaimed
// quiescently first -- add_points and consolidate() run genuinely concurrently, which is
// the combination this test exists to exercise.
CATCH_TEST_CASE(
    "Concurrent MutableVamanaIndex add_points during consolidate", "[concurrent]"
) {
    auto envnum = [](const char* name, size_t fallback) {
        const char* v = std::getenv(name);
        return v != nullptr ? static_cast<size_t>(std::strtoul(v, nullptr, 10)) : fallback;
    };
    const size_t local_initial = envnum("SVS_ADC_INITIAL", 500);
    const size_t local_incremental = envnum("SVS_ADC_INCR", 200);
    const size_t local_batch = envnum("SVS_ADC_BATCH", 25);
    const size_t local_rounds = envnum("SVS_ADC_ROUNDS", 8);
    const size_t local_queries = envnum("SVS_ADC_QUERIES", 10);

    const size_t total = local_initial + local_incremental;
    auto base = random_vectors(total, kDim, 3691);

    std::vector<size_t> initial_ids(local_initial);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(local_initial * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(local_queries, kDim, 1590);

    std::atomic<size_t> exceptions{0};
    // Guards exception_log; appended only from the two catch blocks below, read only
    // after both threads have joined.
    std::mutex exception_log_mutex;
    std::vector<std::string> exception_log;
    // Written only by the consolidator thread; read only after it joins.
    std::unordered_set<size_t> deleted_ids;

    auto inserter = [&] {
        size_t offset = 0;
        try {
            for (; offset < local_incremental; offset += local_batch) {
                const size_t n = std::min(local_batch, local_incremental - offset);
                add_batch(*index, base, local_initial + offset, n);
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "inserter offset=" + std::to_string(offset) + ": " + e.what()
            );
        }
    };

    // Deletes a fresh residue class of original ids each round, then consolidates
    // immediately, so its in-edge reroute for this round's ids can settle while the
    // inserter above may still be wiring new edges into them.
    auto deleter_consolidator = [&] {
        size_t round = 0;
        try {
            for (; round < local_rounds; ++round) {
                std::vector<size_t> to_delete;
                for (size_t id = round; id < local_initial; id += 40) {
                    to_delete.push_back(id);
                    deleted_ids.insert(id);
                }
                index->delete_entries(to_delete);
                index->consolidate();
            }
        } catch (const std::exception& e) {
            exceptions.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(exception_log_mutex);
            exception_log.push_back(
                "consolidator round=" + std::to_string(round) + ": " + e.what()
            );
        }
    };

    std::thread inserter_thread(inserter);
    std::thread consolidator_thread(deleter_consolidator);
    inserter_thread.join();
    consolidator_thread.join();

    std::string exception_report;
    for (const auto& msg : exception_log) {
        exception_report += msg + "; ";
    }
    CATCH_INFO("captured exceptions: " << exception_report);
    CATCH_REQUIRE(exceptions.load() == 0);

    // Directly detects a valid node wired to a Deleted neighbor -- the corruption this
    // test targets -- without needing compact() to turn it into a thrown exception.
    std::string invariant_exception;
    try {
        index->debug_check_invariants(false);
    } catch (const std::exception& e) { invariant_exception = e.what(); }
    CATCH_INFO("debug_check_invariants(false): " << invariant_exception);
    CATCH_REQUIRE(invariant_exception.empty());

    // The production symptom: compact() removes a Deleted slot, and a stale edge into it
    // (one consolidate() never got a chance to reroute) throws when compact() remaps it.
    std::string compact_exception;
    try {
        index->compact();
    } catch (const std::exception& e) { compact_exception = e.what(); }
    CATCH_INFO("compact() exception: " << compact_exception);
    CATCH_REQUIRE(compact_exception.empty());

    std::unordered_set<size_t> live;
    index->on_ids([&live](size_t id) { live.insert(id); });
    CATCH_REQUIRE(live.size() == index->size());
    CATCH_REQUIRE(live.size() == total - deleted_ids.size());
}
