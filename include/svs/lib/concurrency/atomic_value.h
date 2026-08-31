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

#pragma once

#include <atomic>
#include <type_traits>

namespace svs::lib {

///
/// @brief Atomic wrapper for trivially-copyable types.
///
/// Provides a copyable atomic value with acquire/release memory order. Copyability is
/// required for containers that construct elements via `new (p) T(other[i])` or
/// `new (p) T(*fill)`, including `lib::SegmentedVector::resize` with a fill value.
///
template <typename T> class AtomicValue {
    static_assert(
        std::is_trivially_copyable_v<T>, "AtomicValue requires a trivially copyable type"
    );
    static_assert(
        std::atomic<T>::is_always_lock_free,
        "AtomicValue requires a lock-free atomic type; a lock would serialize readers"
    );

  public:
    AtomicValue() = default;

    // Non-explicit constructor from T. Required for `resize(n, fill)` where `fill` is a
    // bare `T` and must convert implicitly to the wrapper.
    AtomicValue(T value)
        : value_{value} {}

    // Copy via load-then-store, because `SegmentedVector` constructs elements with
    // `new (p) T(...)` and rejects a non-copyable element type. Moves fall back to copy.
    AtomicValue(const AtomicValue& other)
        : value_{other.value_.load(std::memory_order_acquire)} {}

    AtomicValue& operator=(const AtomicValue& other) {
        value_.store(
            other.value_.load(std::memory_order_acquire), std::memory_order_release
        );
        return *this;
    }

    // Implicit conversion keeps existing comparisons against an enumerator compiling.
    // Declaring `operator==` instead would be ambiguous with the converting constructor.
    operator T() const { return value_.load(std::memory_order_acquire); }

    // Assignment from T via release store. The lock-free reader sees the write atomically.
    AtomicValue& operator=(T v) {
        value_.store(v, std::memory_order_release);
        return *this;
    }

  private:
    std::atomic<T> value_{};
};

} // namespace svs::lib
