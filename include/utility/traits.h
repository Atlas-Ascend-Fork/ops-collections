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

#include <type_traits>
#include "pair.h"
#include "utility/is_same.h"


namespace aclco {

template <typename T, typename = void>
struct IsBitwiseComparable : FalseType {};

template <typename T>
struct IsBitwiseComparable<T, std::enable_if_t<std::has_unique_object_representations_v<T>>>
  : TrueType {};

template <typename T>
struct IsBitwiseComparable<T, std::enable_if_t<std::is_floating_point_v<T>>> : TrueType {};

template <typename T>
inline constexpr bool isBitwiseComparableV = IsBitwiseComparable<T>::value;

template <std::size_t Size>
struct UintBySize {
    static_assert(Size == 1 || Size == 2 || Size == 4 || Size == 8,
                  "UintBySize only supports sizes 1, 2, 4, and 8");
};

template <> struct UintBySize<1> { using type = uint8_t; };
template <> struct UintBySize<2> { using type = uint16_t; };
template <> struct UintBySize<4> { using type = uint32_t; };
template <> struct UintBySize<8> { using type = uint64_t; };

template <std::size_t Size>
using UintBySizeT = typename UintBySize<Size>::type;

template <typename T>
struct DeviceType { using type = std::conditional_t<std::is_arithmetic_v<T>, T, UintBySizeT<sizeof(T)>>; };

template <typename F, typename S>
struct DeviceType<Pair<F, S>> { using type = Pair<typename DeviceType<F>::type, typename DeviceType<S>::type>; };

template <typename T>
using DeviceTypeT = typename DeviceType<T>::type;
}
