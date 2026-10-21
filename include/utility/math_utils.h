/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdlib>
#include <iterator>
#include <type_traits>

#include "macros.h"

namespace aclco {
template <typename SizeType, typename HashType>
COLLECTION_HOST_DEVICE constexpr SizeType ToPositive(HashType hash)
{
    if constexpr (std::is_signed_v<SizeType>) {
        return std::abs(static_cast<SizeType>(hash));
    } else {
        return static_cast<SizeType>(hash);
    }
}

template <typename SizeType, typename HashType>
COLLECTION_HOST_DEVICE constexpr SizeType SanitizeHash(HashType hash) noexcept
{
    return ToPositive<SizeType>(hash);
}

template <typename T, typename U>
COLLECTION_HOST_DEVICE constexpr T IntDivCeil(T dividend, U divisor) noexcept
{
    static_assert(std::is_integral_v<T>);
    static_assert(std::is_integral_v<U>);
    if (divisor != 0) {
        return dividend / divisor + static_cast<T>(dividend % divisor != 0);
    } else {
        return 0;
    }
}

template <typename T, T lhs, T rhs>
COLLECTION_HOST_DEVICE constexpr T Max() noexcept
{
    if constexpr (lhs < rhs) {
        return rhs;
    } else {
        return lhs;
    }
}
COLLECTION_HOST_DEVICE constexpr std::size_t BitCeil(std::size_t n)
{
    if (n == 0) {
        return 1;
    }
    std::size_t x = 1;
    while (x < n) {
        x <<= 1;
    }
    return x;
}

template <typename T>
struct EqualTo {
    COLLECTION_HOST_DEVICE constexpr bool operator()(T const& lhs, T const& rhs) const noexcept { return lhs == rhs; }
};

template <class Iterator, class Value>
constexpr Iterator LowerBound(Iterator first, Iterator last, const Value& value)
{
    using DiffType = typename std::iterator_traits<Iterator>::difference_type;

    Iterator it{};
    DiffType count = std::distance(first, last);
    DiffType step{};

    while (count > 0) {
        it = first;
        step = count / 2;
        std::advance(it, step);

        if (static_cast<Value>(*it) < value) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }

    return first;
}
} // namespace aclco
