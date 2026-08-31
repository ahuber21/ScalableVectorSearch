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
#include "svs/lib/concurrency/atomic_value.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>

enum class TestEnum : uint8_t { Zero = 0, One = 1, Two = 2 };

CATCH_TEST_CASE("AtomicValue basic operations", "[lib][concurrency][atomic_value]") {
    using svs::lib::AtomicValue;

    CATCH_SECTION("default construction yields value-initialized enum") {
        AtomicValue<TestEnum> value;
        CATCH_REQUIRE(value == TestEnum::Zero);
    }

    CATCH_SECTION("construction from value round-trips") {
        AtomicValue<TestEnum> value(TestEnum::One);
        CATCH_REQUIRE(value == TestEnum::One);
    }

    CATCH_SECTION("copy construction round-trips") {
        AtomicValue<TestEnum> value1(TestEnum::Two);
        AtomicValue<TestEnum> value2(value1);
        CATCH_REQUIRE(value2 == TestEnum::Two);
        CATCH_REQUIRE(value1 == TestEnum::Two);
    }

    CATCH_SECTION("copy assignment round-trips") {
        AtomicValue<TestEnum> value1(TestEnum::One);
        AtomicValue<TestEnum> value2(TestEnum::Two);
        value2 = value1;
        CATCH_REQUIRE(value2 == TestEnum::One);
        CATCH_REQUIRE(value1 == TestEnum::One);
    }

    CATCH_SECTION("assignment from value round-trips") {
        AtomicValue<TestEnum> value(TestEnum::Zero);
        value = TestEnum::Two;
        CATCH_REQUIRE(value == TestEnum::Two);
    }

    CATCH_SECTION("comparison against enumerators works") {
        AtomicValue<TestEnum> value(TestEnum::One);
        CATCH_REQUIRE(value == TestEnum::One);
        CATCH_REQUIRE(value != TestEnum::Zero);
        CATCH_REQUIRE(value != TestEnum::Two);
    }
}
