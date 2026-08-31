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

// header under test
#include "svs/lib/concurrency/seqlock.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <thread>

CATCH_TEST_CASE("SeqLockCounter single-threaded", "[core][seqlock]") {
    svs::SeqLockCounter counter;

    CATCH_SECTION("read_begin succeeds initially") {
        auto seq = counter.read_begin();
        CATCH_REQUIRE(seq.has_value());
        CATCH_REQUIRE(counter.read_validate(*seq) == true);
    }

    CATCH_SECTION("read_begin returns nullopt while write in progress") {
        auto seq = counter.begin_write();
        auto read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value() == false);
        counter.end_write(seq);

        // After write completes, read_begin succeeds again.
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());
    }

    CATCH_SECTION("read_validate detects concurrent write") {
        auto read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        // Simulate a write completing during the read.
        auto write_seq = counter.begin_write();
        counter.end_write(write_seq);

        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);
    }

    CATCH_SECTION("read_validate succeeds without intervening write") {
        auto read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());
        // No write occurred.
        CATCH_REQUIRE(counter.read_validate(*read_seq) == true);
    }

    CATCH_SECTION("read_validate rejects large numbers of intervening writes") {
        // Regression test: ensure counter width is large enough that wrap-around
        // cannot occur in practice. Narrow widths (e.g., uint8_t) allow the counter
        // to return to its original value after a multiple of 2^(width-1) writes,
        // causing read_validate to incorrectly accept a torn read.
        auto read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        // 128 writes (2^7) would wrap an 8-bit counter; must still reject.
        for (int i = 0; i < 128; ++i) {
            auto w = counter.begin_write();
            counter.end_write(w);
        }
        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);

        // Reset for next check.
        counter = svs::SeqLockCounter();
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        // 256 writes (2^8) would wrap an 8-bit counter; must still reject.
        for (int i = 0; i < 256; ++i) {
            auto w = counter.begin_write();
            counter.end_write(w);
        }
        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);

        // Reset for next check.
        counter = svs::SeqLockCounter();
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        // 65536 writes (2^16) would wrap a 16-bit counter; must still reject.
        for (int i = 0; i < 65536; ++i) {
            auto w = counter.begin_write();
            counter.end_write(w);
        }
        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);

        // Positive controls: small counts should still reject.
        counter = svs::SeqLockCounter();
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        // Single write must be detected.
        auto w = counter.begin_write();
        counter.end_write(w);
        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);

        // Two writes must be detected.
        counter = svs::SeqLockCounter();
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());

        for (int i = 0; i < 2; ++i) {
            auto wt = counter.begin_write();
            counter.end_write(wt);
        }
        CATCH_REQUIRE(counter.read_validate(*read_seq) == false);

        // Zero intervening writes: validation should succeed.
        counter = svs::SeqLockCounter();
        read_seq = counter.read_begin();
        CATCH_REQUIRE(read_seq.has_value());
        CATCH_REQUIRE(counter.read_validate(*read_seq) == true);
    }
}

CATCH_TEST_CASE("SeqLockCounter concurrent reader and writer", "[core][seqlock]") {
    svs::SeqLockCounter counter;
    constexpr int kIterations = 10000;
    std::atomic<int> data{0};
    std::atomic<bool> start{false};
    std::atomic<int> failed_reads{0};

    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < kIterations; ++i) {
            auto seq = counter.begin_write();
            data.store(i, std::memory_order_relaxed);
            counter.end_write(seq);
        }
    });

    std::thread reader([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        int successful = 0;
        while (successful < kIterations) {
            auto seq = counter.read_begin();
            if (!seq.has_value()) {
                failed_reads.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            int value = data.load(std::memory_order_relaxed);
            if (!counter.read_validate(*seq)) {
                failed_reads.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            successful++;
            CATCH_REQUIRE(value >= 0);
            CATCH_REQUIRE(value < kIterations);
        }
    });

    start.store(true, std::memory_order_release);
    writer.join();
    reader.join();

    // Reader should have detected some conflicts.
    CATCH_REQUIRE(failed_reads.load(std::memory_order_relaxed) > 0);
}

CATCH_TEST_CASE("SeqLockArray basic operations", "[core][seqlock]") {
    svs::SeqLockArray array(10);
    CATCH_REQUIRE(array.size() == 10);

    for (size_t i = 0; i < 10; ++i) {
        auto seq = array[i].read_begin();
        CATCH_REQUIRE(seq.has_value());
        CATCH_REQUIRE(array[i].read_validate(*seq) == true);
    }

    array.resize(20);
    CATCH_REQUIRE(array.size() == 20);
}
