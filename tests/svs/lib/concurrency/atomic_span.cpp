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
#include "svs/lib/concurrency/atomic_span.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <vector>

CATCH_TEST_CASE("AtomicSpan basic operations", "[core][atomic_span]") {
    std::vector<int> data = {1, 2, 3, 4, 5};
    svs::AtomicSpan<int> span(data.data(), data.size());

    CATCH_REQUIRE(span.size() == 5);
    CATCH_REQUIRE(span.empty() == false);
    CATCH_REQUIRE(span.data() == data.data());

    CATCH_SECTION("element access") {
        for (size_t i = 0; i < data.size(); ++i) {
            CATCH_REQUIRE(span[i] == data[i]);
        }
    }

    CATCH_SECTION("iteration") {
        std::vector<int> collected;
        for (int val : span) {
            collected.push_back(val);
        }
        CATCH_REQUIRE(collected == data);
    }

    CATCH_SECTION("empty span") {
        svs::AtomicSpan<int> empty_span(nullptr, 0);
        CATCH_REQUIRE(empty_span.empty());
        CATCH_REQUIRE(empty_span.size() == 0);
    }
}
