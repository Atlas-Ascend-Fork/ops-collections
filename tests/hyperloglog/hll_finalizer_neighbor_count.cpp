/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
// Standalone Host regression; only standard C++ and the actual HLL headers.
// Run: c++ -std=c++17 -I include tests/hyperloglog/hll_finalizer_neighbor_count.cpp -o /tmp/hll-neighbors
//      /tmp/hll-neighbors
#include "detail/hyperloglog/finalizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace hll = aclco::detail::hyperloglog;

double AlphaSquared(int precision)
{
    double const registers = 1 << precision;
    return (0.7213 / (1.0 + 1.079 / registers)) * registers * registers;
}

std::uint64_t NearestSixOracle(int precision, double sum)
{
    double estimate = 0.0;
    if (sum > 0.0 && std::isfinite(sum)) {
        estimate = AlphaSquared(precision) / sum;
    } else {
        throw std::invalid_argument("Oracle sum must be finite and positive");
    }
    auto const raw = hll::raw_estimate_data(precision);
    auto const biases = hll::bias_data(precision);
    auto const size = hll::raw_estimate_data_size(precision);
    std::vector<std::pair<double, std::size_t>> distances;
    for (std::size_t index = 0; index < size; ++index) {
        distances.emplace_back(std::abs(raw[index] - estimate), index);
    }
    std::sort(distances.begin(), distances.end());
    double correction = 0.0;
    for (std::size_t index = 0; index < 6; ++index) {
        correction += biases[distances[index].second];
    }
    return static_cast<std::uint64_t>(std::round(estimate - correction / 6));
}

int InvalidOracleSums()
{
    int rejected = 0;
    for (double const sum :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        try {
            NearestSixOracle(13, sum);
        } catch (std::invalid_argument const&) {
            ++rejected;
        }
    }
    std::cout << "ORACLE invalid sums rejected=" << rejected << '\n';
    return rejected == 4 ? 0 : 1;
}

int TailCases(int precision)
{
    int const registers = 1 << precision;
    auto const raw = hll::raw_estimate_data(precision);
    auto const size = hll::raw_estimate_data_size(precision);
    int checked = 0;
    int wrong = 0;
    for (int rank_two = 0; rank_two <= registers; ++rank_two) {
        double const sum = rank_two / 4.0 + (registers - rank_two) / 8.0;
        double const estimate = AlphaSquared(precision) / sum;
        if (estimate <= raw[size - 1] || estimate >= 5.0 * registers) {
            continue;
        }
        ++checked;
        auto const actual = hll::Finalizer(precision)(sum, 0);
        auto const expected = NearestSixOracle(precision, sum);
        if (actual != expected) {
            ++wrong;
        }
    }
    std::cout << "TAIL precision=" << precision << " checked=" << checked << " wrong=" << wrong << '\n';
    return (checked > 0 && wrong == 0) ? 0 : 1;
}

int NormalControls(int precision)
{
    int const registers = 1 << precision;
    hll::Finalizer const finalizer(precision);
    for (int const rank_two : {registers / 2, 3 * registers / 4, registers}) {
        double const sum = rank_two / 4.0 + (registers - rank_two) / 8.0;
        if (finalizer(sum, 0) != NearestSixOracle(precision, sum)) {
            return 1;
        }
    }
    if (finalizer(registers, registers) != 0) {
        return 1;
    }
    if (finalizer(0.0, 0) != std::numeric_limits<std::uint64_t>::max()) {
        return 1;
    }
    double const large_sum = registers / 16.0;
    auto const large_expected = static_cast<std::uint64_t>(std::round(AlphaSquared(precision) / large_sum));
    return finalizer(large_sum, 0) == large_expected ? 0 : 1;
}
} // namespace

int main()
{
    int failed = InvalidOracleSums();
    for (int precision = 13; precision <= 18; ++precision) {
        failed += TailCases(precision);
        int const controls = NormalControls(precision);
        failed += controls;
        std::cout << "CONTROLS precision=" << precision << " wrong=" << controls << '\n';
    }
    return failed == 0 ? 0 : 1;
}
