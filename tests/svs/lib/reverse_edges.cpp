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
#include "svs/lib/reverse_edges.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_set>

CATCH_TEST_CASE("ReverseEdges basic operations", "[core][reverse_edges]") {
    svs::lib::ReverseEdges<uint32_t> edges(10);

    CATCH_SECTION("record and collect") {
        // Node 5 has in-neighbors 1, 2, 3.
        edges.record(1, 5);
        edges.record(2, 5);
        edges.record(3, 5);

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(5, collected, is_deleted);

        CATCH_REQUIRE(collected.size() == 3);
        CATCH_REQUIRE(collected.count(1) == 1);
        CATCH_REQUIRE(collected.count(2) == 1);
        CATCH_REQUIRE(collected.count(3) == 1);
    }

    CATCH_SECTION("remove") {
        edges.record(1, 5);
        edges.record(2, 5);
        edges.record(3, 5);

        edges.remove(2, 5);

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(5, collected, is_deleted);

        CATCH_REQUIRE(collected.size() == 2);
        CATCH_REQUIRE(collected.count(1) == 1);
        CATCH_REQUIRE(collected.count(3) == 1);
        CATCH_REQUIRE(collected.count(2) == 0);
    }

    CATCH_SECTION("reset_node") {
        edges.record(1, 5);
        edges.record(2, 5);

        edges.reset_node(5);

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(5, collected, is_deleted);

        CATCH_REQUIRE(collected.empty());
    }

    CATCH_SECTION("reset all") {
        edges.record(1, 5);
        edges.record(2, 3);

        edges.reset();

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(5, collected, is_deleted);
        CATCH_REQUIRE(collected.empty());

        edges.collect(3, collected, is_deleted);
        CATCH_REQUIRE(collected.empty());
    }

    CATCH_SECTION("recording on/off") {
        edges.set_recording(false);
        edges.record(1, 5);

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(5, collected, is_deleted);
        CATCH_REQUIRE(collected.empty());

        edges.set_recording(true);
        edges.record(2, 5);
        edges.collect(5, collected, is_deleted);
        CATCH_REQUIRE(collected.size() == 1);
        CATCH_REQUIRE(collected.count(2) == 1);
    }

    CATCH_SECTION("resize") {
        edges.resize(20);
        edges.record(15, 18);

        tsl::robin_set<size_t> collected;
        auto is_deleted = [](uint32_t) { return false; };
        edges.collect(18, collected, is_deleted);
        CATCH_REQUIRE(collected.size() == 1);
        CATCH_REQUIRE(collected.count(15) == 1);
    }

    CATCH_SECTION("collect with deleted filter") {
        edges.record(1, 5);
        edges.record(2, 5);
        edges.record(3, 5);

        auto is_deleted = [](uint32_t n) { return n == 2; };

        tsl::robin_set<size_t> collected;
        edges.collect(5, collected, is_deleted);

        CATCH_REQUIRE(collected.size() == 2);
        CATCH_REQUIRE(collected.count(1) == 1);
        CATCH_REQUIRE(collected.count(3) == 1);
        CATCH_REQUIRE(collected.count(2) == 0);
    }
}
