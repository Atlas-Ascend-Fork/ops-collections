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

namespace aclco::detail::multimap {

// One warp scans and inserts one 256-key chunk. A ballot/popcount pair gives
// each lane its rank within a 32-key part; parts and lanes keep key order, so
// the admitted prefix is identical to the serial path.
template <class Key, class Value, class Stencil, class Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(INSERT_THREADS) inline void InsertWarpSimt(
    __gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity, Key emptyKey, Value emptyValue,
    __gm__ Pair<Key, Value>* input, __gm__ Stencil* stencil, std::size_t n, __gm__ uint64_t* offsets,
    std::size_t accepted, std::size_t size, __gm__ uint32_t* failed)
{
    constexpr std::uint32_t WARP = 32;
    StaticMultimapRef<Key, Value> ref{table, capacity, emptyKey, emptyValue, indices};
    auto const lane = static_cast<std::uint32_t>(AscendC::Simt::GetThreadIdx()) % WARP;
    auto const chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex() / WARP; c < chunks; c += ThreadStride() / WARP) {
        std::uint64_t carry = 0;
        auto const chunkEnd = (c + 1) * CHUNK < n ? (c + 1) * CHUNK : n;
        for (std::uint32_t part = 0; part < CHUNK; part += WARP) {
            auto const i = c * CHUNK + part + lane;
            bool const selected = i < chunkEnd && Predicate{}(stencil[i]);
            unsigned int const mask = __ballot(static_cast<int>(selected));
            unsigned int const laneMask = (1U << lane) - 1U;
            std::uint32_t const prefix = static_cast<std::uint32_t>(__popc(mask & laneMask));
            std::uint32_t const total = static_cast<std::uint32_t>(__popc(mask));
            auto const rank = offsets[c] + carry + prefix;
            if (selected && rank < accepted) {
                if (!InsertSelected(ref, table, indices, input, i, size + rank)) {
                    AscendC::Simt::AtomicCas(failed, uint32_t(0), uint32_t(1));
                }
            }
            carry += total;
        }
    }
}

template <class Key, class Value, class Stencil, class Predicate>
COLLECTION_AIV_GLOBAL void InsertWarp(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices,
                                      std::size_t capacity, Key emptyKey, Value emptyValue,
                                      __gm__ Pair<Key, Value>* input, __gm__ Stencil* stencil, std::size_t n,
                                      __gm__ uint64_t* offsets, std::size_t accepted, std::size_t size,
                                      __gm__ uint32_t* failed)
{
    AscendC::Simt::VF_CALL<InsertWarpSimt<Key, Value, Stencil, Predicate>>(
        AscendC::Simt::Dim3{INSERT_THREADS}, table, indices, capacity, emptyKey, emptyValue, input, stencil, n, offsets,
        accepted, size, failed);
}

} // namespace aclco::detail::multimap
