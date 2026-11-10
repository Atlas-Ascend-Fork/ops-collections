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

#include "detail/bitwise_compare.h"
#include "detail/open_addressing/kernels.h"
#include "kernel_operator.h"

namespace aclco
{
/** InsertIf/IoA 用 stencil 谓词：mask 字节为 0（不在任何旧子表）才执行。 */
struct MaskIsZero
{
    COLLECTION_HOST_DEVICE constexpr bool operator()(uint8_t v) const { return v == 0; }
};

/** InsertOrAssign 用 stencil 谓词：mask 字节非 0（命中旧子表）才执行 assign。 */
struct MaskNonZero
{
    COLLECTION_HOST_DEVICE constexpr bool operator()(uint8_t v) const { return v != 0; }
};

// ExtractKeys worker：从 Pair<Key,T> 数组抽取 key 列（旧子表 Contains 过滤需要连续 key）。
template <typename Key, typename PairT>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ExtractKeysSimt(__gm__ uint8_t* keys,
                                                                                     __gm__ uint8_t* pairs, uint32_t n)
{
    uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint32_t blockNumber = AscendC::Simt::GetBlockNum();
    uint32_t globalIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t totalThreads = blockNumber * AscendC::Simt::GetThreadNum();
    for (uint32_t i = globalIdx; i < n; i += totalThreads)
    {
        *((__gm__ Key*)keys + i) = ((__gm__ PairT*)pairs + i)->first;
    }
}

// CountZero worker：统计 mask 中 0 字节个数（= 不在旧子表、应插入活跃子表的元素数）。
// 寄存器累加 + 每线程一次全局原子加。
template <typename Dummy = void>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void CountZeroSimt(__gm__ uint8_t* mask, uint32_t n,
                                                                                   __gm__ uint32_t* zeroNum)
{
    uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint32_t blockNumber = AscendC::Simt::GetBlockNum();
    uint32_t globalIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t totalThreads = blockNumber * AscendC::Simt::GetThreadNum();
    uint32_t localZero = 0;
    for (uint32_t i = globalIdx; i < n; i += totalThreads)
    {
        if (*((__gm__ uint8_t*)mask + i) == 0)
        {
            ++localZero;
        }
    }
    if (localZero != 0)
    {
        AscendC::Simt::AtomicAdd(zeroNum, localZero);
    }
}

// SIMT worker：每线程处理一段下标。
// Find worker：若 out[i] 仍为空值则用本子表查询结果 tmp[i] 覆盖（key 至多命中一个子表）。
template <typename Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void MergeFindSimt(__gm__ uint8_t* out,
                                                                                   __gm__ uint8_t* tmp,
                                                                                   __gm__ uint8_t* emptyValue,
                                                                                   uint32_t n)
{
    uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint32_t blockNumber = AscendC::Simt::GetBlockNum();
    uint32_t globalIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t totalThreads = blockNumber * AscendC::Simt::GetThreadNum();
    Value empty = *((__gm__ Value*)emptyValue);
    for (uint32_t i = globalIdx; i < n; i += totalThreads)
    {
        Value cur = *((__gm__ Value*)out + i);
        bool isEmpty = cur == empty;
        if constexpr (std::is_floating_point_v<Value>) {
            // NaN sentinels match the original object representation, including the payload bits.
            isEmpty = isEmpty || (empty != empty && detail::BitWiseCompare(cur, empty));
        }
        if (isEmpty)
        {
            *((__gm__ Value*)out + i) = *((__gm__ Value*)tmp + i);
        }
    }
}

// Contains worker：对每个下标做按位 OR（uint8 bool）。
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void MergeContainsSimt(__gm__ uint8_t* out,
                                                                                       __gm__ uint8_t* tmp, uint32_t n)
{
    uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint32_t blockNumber = AscendC::Simt::GetBlockNum();
    uint32_t globalIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t totalThreads = blockNumber * AscendC::Simt::GetThreadNum();
    for (uint32_t i = globalIdx; i < n; i += totalThreads)
    {
        if (*((__gm__ uint8_t*)tmp + i) != 0)
        {
            *((__gm__ uint8_t*)out + i) = 1;
        }
    }
}

// __global__ 启动包装：用 VF_CALL 启动上面的 SIMT worker（与 StaticMap kernel 同构）。
template <typename Value>
COLLECTION_AIV_GLOBAL void MergeFind(__gm__ uint8_t* out, __gm__ uint8_t* tmp, __gm__ uint8_t* emptyValue, uint32_t n)
{
    AscendC::Simt::VF_CALL<MergeFindSimt<Value>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM}, out, tmp, emptyValue, n);
}

template <typename Dummy = void>
COLLECTION_AIV_GLOBAL void MergeContains(__gm__ uint8_t* out, __gm__ uint8_t* tmp, uint32_t n)
{
    AscendC::Simt::VF_CALL<MergeContainsSimt>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM}, out, tmp, n);
}

template <typename Key, typename PairT>
COLLECTION_AIV_GLOBAL void ExtractKeys(__gm__ uint8_t* keys, __gm__ uint8_t* pairs, uint32_t n)
{
    AscendC::Simt::VF_CALL<ExtractKeysSimt<Key, PairT>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM}, keys, pairs, n);
}

template <typename Dummy = void>
COLLECTION_AIV_GLOBAL void CountZero(__gm__ uint8_t* mask, uint32_t n, __gm__ uint32_t* zeroNum)
{
    AscendC::Simt::VF_CALL<CountZeroSimt<>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM}, mask, n, zeroNum);
}

}  // namespace aclco
