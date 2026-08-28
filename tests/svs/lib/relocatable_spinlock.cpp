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
#include "svs/lib/relocatable_spinlock.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <mutex>

CATCH_TEST_CASE("RelocatableSpinLock basic operations", "[core][relocatable_spinlock]") {
    svs::RelocatableSpinLock lock;
    CATCH_REQUIRE(lock.islocked() == false);
    {
        std::lock_guard guard{lock};
        CATCH_REQUIRE(lock.islocked() == true);
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(lock.try_lock() == false);
        }
    }
    CATCH_REQUIRE(lock.islocked() == false);
}

CATCH_TEST_CASE(
    "RelocatableSpinLock copy is born unlocked", "[core][relocatable_spinlock]"
) {
    svs::RelocatableSpinLock original;
    {
        std::lock_guard guard{original};
        CATCH_REQUIRE(original.islocked() == true);

        // Copy while original is locked.
        svs::RelocatableSpinLock copy(original);
        CATCH_REQUIRE(copy.islocked() == false);

        // Copy assignment while original is locked.
        svs::RelocatableSpinLock assigned;
        assigned = original;
        CATCH_REQUIRE(assigned.islocked() == false);
    }
}

CATCH_TEST_CASE(
    "RelocatableSpinLock move is born unlocked", "[core][relocatable_spinlock]"
) {
    svs::RelocatableSpinLock original;
    {
        std::lock_guard guard{original};
        CATCH_REQUIRE(original.islocked() == true);

        // Move while original is locked.
        svs::RelocatableSpinLock moved(std::move(original));
        CATCH_REQUIRE(moved.islocked() == false);

        // Move assignment while original is locked.
        svs::RelocatableSpinLock assigned;
        assigned = std::move(moved);
        CATCH_REQUIRE(assigned.islocked() == false);
    }
}
