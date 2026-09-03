/*
 * Copyright 2025 Intel Corporation
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

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

namespace svs::lib {

/// @brief Movable wrapper for non-movable mutexes.
///
/// Moving does not transfer lock state; moving while any lock is held results in
/// undefined behavior as the destination has no knowledge of existing holders.
template <typename Mutex> class MovableMutex {
  public:
    MovableMutex() = default;

    MovableMutex(const MovableMutex&) = delete;
    MovableMutex& operator=(const MovableMutex&) = delete;

    MovableMutex(MovableMutex&& other) noexcept
        : mutex_{} {
        // Verify the source is not locked before move-construction.
        // Locked source + move leads to undefined behavior (new owner unaware of holders).
        assert(other.try_lock() && (other.unlock(), true));
        (void)other; // Suppress unused parameter warning when assert compiles out.
    }

    MovableMutex& operator=(MovableMutex&& other) noexcept {
        // Verify the source is not locked before move-assignment.
        // Locked source + move leads to undefined behavior (new owner unaware of holders).
        assert(other.try_lock() && (other.unlock(), true));
        (void)other; // Suppress unused parameter warning when assert compiles out.
        return *this;
    }

    ~MovableMutex() = default;

    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }
    bool try_lock() { return mutex_.try_lock(); }

    void lock_shared()
        requires requires(Mutex& m) { m.lock_shared(); }
    {
        mutex_.lock_shared();
    }

    void unlock_shared()
        requires requires(Mutex& m) { m.unlock_shared(); }
    {
        mutex_.unlock_shared();
    }

    bool try_lock_shared()
        requires requires(Mutex& m) {
                     { m.try_lock_shared() } -> std::convertible_to<bool>;
                 }
    {
        return mutex_.try_lock_shared();
    }

  private:
    Mutex mutex_{};
};

} // namespace svs::lib
