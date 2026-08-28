/*
 * Copyright 2023 Intel Corporation
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

#include "svs/core/logging.h"

#include <algorithm>
#include <memory>

namespace svs_test {

// Pushing a sink with captured local storage onto a process-global logger causes
// use-after-free when that storage outlives the test. Remove on destruction.
class ScopedGlobalSink {
  public:
    ScopedGlobalSink(svs::logging::logger_ptr logger, spdlog::sink_ptr sink)
        : logger_{std::move(logger)}
        , sink_{std::move(sink)} {
        logger_->sinks().push_back(sink_);
    }

    ScopedGlobalSink(const ScopedGlobalSink&) = delete;
    ScopedGlobalSink& operator=(const ScopedGlobalSink&) = delete;
    ScopedGlobalSink(ScopedGlobalSink&&) = delete;
    ScopedGlobalSink& operator=(ScopedGlobalSink&&) = delete;

    ~ScopedGlobalSink() {
        auto& sinks = logger_->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
    }

  private:
    svs::logging::logger_ptr logger_;
    spdlog::sink_ptr sink_;
};

} // namespace svs_test
