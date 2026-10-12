/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "utility/traits.h"

namespace aclco {

namespace detail {

template <typename T>
COLLECTION_SIMT_DEVICE T AtomicCas2Byte(__gm__ T* addr, T expected, T desired)
{
    using WideType = uint32_t;
    uintptr_t addrVal = reinterpret_cast<uintptr_t>(addr);
    uintptr_t alignedAddrVal = addrVal & ~3ull;
    __gm__ WideType* alignedAddr = reinterpret_cast<__gm__ WideType*>(alignedAddrVal);
    int shift = static_cast<int>((addrVal & 2ull) * 8); // 0 or 16

    while (true) {
        WideType currentWide = *alignedAddr;
        T currentNarrow = static_cast<T>(currentWide >> shift);

        if (currentNarrow != expected) {
            return currentNarrow;
        }

        WideType mask = static_cast<WideType>(0xFFFFu) << shift;
        WideType desiredWide = (currentWide & ~mask) | ((static_cast<WideType>(desired) & 0xFFFFu) << shift);
        WideType oldWide = AscendC::Simt::AtomicCas(alignedAddr, currentWide, desiredWide);

        if (oldWide == currentWide) {
            return expected;
        }
    }
}

} // namespace detail

template <typename T>
COLLECTION_SIMT_DEVICE T AtomicCasWrap(__gm__ T* addr, T expected, T desired)
{
    if constexpr (sizeof(T) == 2) {
        return detail::AtomicCas2Byte(addr, expected, desired);
    } else {
        return AscendC::Simt::AtomicCas(addr, expected, desired);
    }
}

} // namespace aclco
