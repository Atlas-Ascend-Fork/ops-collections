/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * Portions adapted from NVIDIA cuCollections.
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/hyperloglog/tuning.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <cstddef>
#include <cstdint>

namespace aclco::detail::hyperloglog {

/**
 * @brief Estimate correction algorithm based on HyperLogLog++.
 *
 * @note Variable names correspond to the definitions given in the HLL++ paper:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf
 * @note Precision must be in [13, 18].
 *
 */
class Finalizer {
public:
    constexpr Finalizer(int precision) : precision_{precision}, m_{1 << precision} {}

    /**
     * @brief Compute the bias-corrected cardinality estimate.
     *
     * @param z Sum of 2^(-rank) over all registers
     * @param v Number of 0 registers
     */
    std::uint64_t operator()(double z, int v) const noexcept
    {
        if (z == 0.0) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        double e = this->alpha_mm() / z;

        if (v > 0) {
            // Use linear counting for small cardinality estimates.
            double const h = this->m_ * log(static_cast<double>(this->m_) / v);
            // The threshold `2.5 * m` is from the original HLL algorithm.
            if (e <= 2.5 * this->m_) {
                return std::round(h);
            }

            if (this->precision_ < 19) {
                e = (h <= threshold(this->precision_)) ? h : this->bias_corrected_estimate(e);
            }
        } else {
            // HLL++ is defined only when p < 19, otherwise we need to fallback to HLL.
            if (this->precision_ < 19) {
                e = this->bias_corrected_estimate(e);
            }
        }

        return e >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ?
                   std::numeric_limits<std::uint64_t>::max() :
                   static_cast<std::uint64_t>(std::round(e));
    }

private:
    constexpr double alpha_mm() const noexcept
    {
        switch (this->m_) {
            case 16:
                return 0.673 * this->m_ * this->m_;
            case 32:
                return 0.697 * this->m_ * this->m_;
            case 64:
                return 0.709 * this->m_ * this->m_;
            default:
                return (0.7213 / (1.0 + 1.079 / this->m_)) * this->m_ * this->m_;
        }
    }

    constexpr double bias_corrected_estimate(double e) const noexcept
    {
        return (e < 5.0 * this->m_) ? e - this->bias(e) : e;
    }

    constexpr double bias(double e) const noexcept
    {
        auto const anchor_index = this->interpolation_anchor_index(e);
        int const n = raw_estimate_data_size(this->precision_);

        auto low = std::max(std::min(anchor_index - k + 1, n - k), 0);
        auto high = std::min(low + k, n);
        // Keep moving bounds as long as the (exclusive) high bound is closer to the estimate than
        // the lower (inclusive) bound.
        while (high < n and this->distance(e, high) < this->distance(e, low)) {
            low += 1;
            high += 1;
        }

        auto biases = bias_data(this->precision_);
        double bias_sum = 0.0;
        for (int i = low; i < high; ++i) {
            bias_sum += biases[i];
        }

        return bias_sum / (high - low);
    }

    constexpr double distance(double e, int i) const noexcept
    {
        auto const diff = e - raw_estimate_data(this->precision_)[i];
        return diff * diff;
    }

    constexpr int interpolation_anchor_index(double e) const noexcept
    {
        auto estimates = raw_estimate_data(this->precision_);
        int const n = raw_estimate_data_size(this->precision_);
        int left = 0;
        int right = static_cast<int>(n) - 1;
        int mid = -1;
        while (left <= right) {
            mid = left + (right - left) / 2;

            if (estimates[mid] < e) {
                left = mid + 1;
            } else if (estimates[mid] > e) {
                right = mid - 1;
            } else {
                return mid;
            }
        }

        // Use the insertion point when the estimate is not present.
        return left;
    }

    static constexpr auto k = 6; ///< Number of interpolation points to consider
    int precision_;
    int m_;
};
} // namespace aclco::detail::hyperloglog
