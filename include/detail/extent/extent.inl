/*
 * Copyright (c) 2022-2025， NVIDIA CPRPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <limits>
#include <algorithm>
#include "utility/math_utils.h"
#include "utility/prime.h"
#include "probing_scheme.h"

namespace aclco {

template <int32_t BucketSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr MakeValidExtent(Extent<SizeType, N> ext)
{
    auto constexpr stride = BucketSize;
    auto constexpr maxPrime = primes.back();
    auto constexpr maxValue = (static_cast<uint64_t>(std::numeric_limits<SizeType>::max()) < maxPrime) ?
                                  std::numeric_limits<SizeType>::max() :
                                  static_cast<SizeType>(maxPrime);
    auto const size = IntDivCeil(std::max(static_cast<SizeType>(ext), static_cast<SizeType>(1)), stride);

    if (size > maxValue) {
        throw std::invalid_argument("Invalid input Extent");
    }

    if constexpr (N == dynamicExtent) {
        return Extent<SizeType, dynamicExtent>{
            static_cast<SizeType>(*LowerBound(primes.begin(), primes.end(), static_cast<uint64_t>(size)) * stride)};
    } else {
        return Extent<SizeType, static_cast<std::size_t>(
                                    *LowerBound(primes.begin(), primes.end(), static_cast<uint64_t>(size)) * stride)>{};
    }
}

template <typename ProbingScheme, typename Storage, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr MakeValidExtent(Extent<SizeType, N> ext)
{
    if constexpr (aclco::IsDoubleHashing<ProbingScheme>::value) {
        return MakeValidExtent<Storage::bucketSize, SizeType, N>(ext);
    } else {
        auto constexpr stride = Storage::bucketSize; // 后续新增线程组功能时需要扩展

        if constexpr (N == dynamicExtent) {
            auto const size = IntDivCeil(std::max(static_cast<SizeType>(ext), static_cast<SizeType>(1)), stride) +
                              static_cast<SizeType>(ext == 0);
            return Extent<SizeType, dynamicExtent>(size * stride);
        } else {
            auto constexpr size = IntDivCeil(Max<SizeType, static_cast<SizeType>(ext), static_cast<SizeType>(1)>(),
                                             stride) +
                                  static_cast<SizeType>(ext == 0);
            return Extent<SizeType, size * stride>();
        }
    }
}

} // namespace aclco
