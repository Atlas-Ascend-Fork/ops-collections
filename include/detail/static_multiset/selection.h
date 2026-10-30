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

#include "static_multiset_ref.h"
#include "simt_api/device_sync_functions.h"

namespace aclco::detail::static_multiset {
constexpr uint32_t THREADS = 1024;
constexpr uint32_t QUERY_THREADS = 1024;
constexpr uint32_t BLOCKS = 56;
constexpr uint32_t WORKERS = THREADS * BLOCKS;
constexpr uint32_t PARTITIONS = WORKERS / 32;
constexpr uint32_t BLOCK_PREFIX = 2 * PARTITIONS + 2;
constexpr uint32_t RANGE_MIN = BLOCK_PREFIX + 5 * BLOCKS + 4;
constexpr uint32_t RANGE_MAX = RANGE_MIN + PARTITIONS;
constexpr uint32_t ORDER_ERRORS = RANGE_MAX + PARTITIONS;
constexpr uint32_t WORK_SIZE = ORDER_ERRORS + PARTITIONS;
constexpr uint32_t SMALL_INSERT_LIMIT = 4096;
struct AcceptAll {
    template <typename T>
    COLLECTION_SIMT_DEVICE bool operator()(T) const
    {
        return true;
    }
};
COLLECTION_SIMT_DEVICE uint64_t ThreadId()
{
    return AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
}
// Distribute complete warp-sized tiles, preserving stable input/output order.
// Aligned starts avoid splitting every warp's contiguous memory accesses.
COLLECTION_SIMT_DEVICE uint64_t Begin(uint64_t n, uint64_t id)
{
    uint64_t tiles = n / 32 + (n % 32 != 0);
    uint64_t first = (tiles / PARTITIONS) * id + (id < tiles % PARTITIONS ? id : tiles % PARTITIONS);
    return first > n / 32 ? n : first * 32;
}

template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_SIMT_DEVICE void SelectPartition(__gm__ Key* keys, __gm__ Stencil* stencil, uint64_t n, Key empty,
                                            __gm__ uint64_t* work, Predicate pred, __gm__ Key* ordered, uint64_t size,
                                            uint64_t available, uint64_t id)
{
    uint64_t lane = AscendC::Simt::GetThreadIdx() % 32;
    uint64_t valid = 0, invalid = 0, low = UINT64_MAX, high = 0, orderErrors = 0;
    for (uint64_t i = Begin(n, id) + lane; i < Begin(n, id + 1); i += 32) {
        // Preserve the raw accepted prefix while reading it for statistics.
        // If invalid keys occur, stable compaction subsequently overwrites this
        // private tail before the logical size is published.
        if constexpr (!Conditional) {
            if (i < available) {
                ordered[size + i] = keys[i];
            }
        }
        if (i && keys[i] <= keys[i - 1]) {
            ++orderErrors;
        }
        if constexpr (Conditional) {
            if (!pred(stencil[i])) {
                continue;
            }
        }
        if (keys[i] == empty) {
            ++invalid;
        } else {
            ++valid;
            uint64_t encoded = StaticMultisetRef<Key>::Encode(keys[i]);
            low = encoded < low ? encoded : low;
            high = encoded > high ? encoded : high;
        }
    }
    for (uint32_t delta = 16; delta > 0; delta >>= 1) {
        valid += AscendC::Simt::WarpShflDownSync(valid, delta);
        orderErrors += AscendC::Simt::WarpShflDownSync(orderErrors, delta);
        invalid += AscendC::Simt::WarpShflDownSync(invalid, delta);
        uint64_t otherLow = AscendC::Simt::WarpShflDownSync(low, delta);
        uint64_t otherHigh = AscendC::Simt::WarpShflDownSync(high, delta);
        low = otherLow < low ? otherLow : low;
        high = otherHigh > high ? otherHigh : high;
    }
    if (lane == 0) {
        work[id] = valid;
        work[PARTITIONS + 1 + id] = invalid;
        work[RANGE_MIN + id] = low;
        work[RANGE_MAX + id] = high;
        work[ORDER_ERRORS + id] = orderErrors;
    }
}
template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void SelectSimt(__gm__ Key* keys, __gm__ Stencil* stencil, uint64_t n,
                                                                Key empty, __gm__ uint64_t* work, Predicate pred,
                                                                __gm__ Key* ordered, uint64_t size, uint64_t available)
{
    SelectPartition<Key, Stencil, Predicate, Conditional>(keys, stencil, n, empty, work, pred, ordered, size, available,
                                                          ThreadId() / 32);
}
// Small batches occupy at most 128 warp partitions. One block collects their
// statistics and builds global prefixes without two additional scan launches.
template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void SmallSelectSimt(__gm__ Key* keys, __gm__ Stencil* stencil,
                                                                     uint64_t n, Key empty, __gm__ uint64_t* work,
                                                                     Predicate pred, __gm__ Key* ordered, uint64_t size,
                                                                     uint64_t available)
{
    uint32_t const lane = AscendC::Simt::GetThreadIdx();
    uint32_t const activePartitions = (n + 31) / 32;
    for (uint32_t id = lane / 32; id < activePartitions; id += THREADS / 32) {
        SelectPartition<Key, Stencil, Predicate, Conditional>(keys, stencil, n, empty, work, pred, ordered, size,
                                                              available, id);
    }
    asc_syncthreads();
    if (lane != 0) {
        return;
    }
    uint64_t sum = 0, invalid = 0, low = UINT64_MAX, high = 0, orderErrors = 0;
    for (uint32_t id = 0; id < activePartitions; ++id) {
        uint64_t const count = work[id];
        work[id] = sum;
        sum += count;
        invalid += work[PARTITIONS + 1 + id];
        low = work[RANGE_MIN + id] < low ? work[RANGE_MIN + id] : low;
        high = work[RANGE_MAX + id] > high ? work[RANGE_MAX + id] : high;
        orderErrors += work[ORDER_ERRORS + id];
    }
    // InsertSimt adds the block prefix to each warp prefix. The small path
    // already produced global warp prefixes, so all block prefixes are zero.
    for (uint32_t block = 0; block < BLOCKS; ++block) {
        work[BLOCK_PREFIX + block] = 0;
    }
    work[PARTITIONS] = sum;
    work[PARTITIONS + 1] = invalid;
    work[BLOCK_PREFIX + 5 * BLOCKS] = low;
    work[BLOCK_PREFIX + 5 * BLOCKS + 1] = high;
    work[BLOCK_PREFIX + 5 * BLOCKS + 2] = orderErrors;
    // Partition statistics are no longer needed. Reuse their storage for one
    // contiguous host read, retaining the layout consumed by InsertSimt above.
    work[RANGE_MIN] = sum;
    work[RANGE_MIN + 1] = invalid;
    work[RANGE_MIN + 2] = low;
    work[RANGE_MIN + 3] = high;
    work[RANGE_MIN + 4] = orderErrors;
}

template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_AIV_GLOBAL void SmallSelectKernel(GM_ADDR keys, GM_ADDR stencil, uint64_t n, Key empty, GM_ADDR work,
                                             Predicate pred, GM_ADDR ordered, uint64_t size, uint64_t available)
{
    AscendC::Simt::VF_CALL<SmallSelectSimt<Key, Stencil, Predicate, Conditional>>(
        AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)keys, (__gm__ Stencil*)stencil, n, empty, (__gm__ uint64_t*)work,
        pred, (__gm__ Key*)ordered, size, available);
}
template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_AIV_GLOBAL void SelectKernel(GM_ADDR keys, GM_ADDR stencil, uint64_t n, Key empty, GM_ADDR work,
                                        Predicate pred, GM_ADDR ordered, uint64_t size, uint64_t available)
{
    AscendC::Simt::VF_CALL<SelectSimt<Key, Stencil, Predicate, Conditional>>(
        AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)keys, (__gm__ Stencil*)stencil, n, empty, (__gm__ uint64_t*)work,
        pred, (__gm__ Key*)ordered, size, available);
}

// Scan each block's contiguous warp totals, then scan the small block array.
template <bool Invalid>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ScanBlocksSimt(__gm__ uint64_t* work)
{
    if (AscendC::Simt::GetThreadIdx() != 0) {
        return;
    }
    uint32_t block = AscendC::Simt::GetBlockIdx();
    uint64_t sum = 0, invalid = 0, low = UINT64_MAX, high = 0, orderErrors = 0;
    constexpr uint32_t LOCAL_PARTITIONS = THREADS / 32;
    for (uint32_t i = block * LOCAL_PARTITIONS; i < (block + 1) * LOCAL_PARTITIONS; ++i) {
        uint64_t count = work[i];
        work[i] = sum;
        sum += count;
        if constexpr (Invalid) {
            invalid += work[PARTITIONS + 1 + i];
            orderErrors += work[ORDER_ERRORS + i];
            low = work[RANGE_MIN + i] < low ? work[RANGE_MIN + i] : low;
            high = work[RANGE_MAX + i] > high ? work[RANGE_MAX + i] : high;
        }
    }
    work[BLOCK_PREFIX + block] = sum;
    if constexpr (Invalid) {
        work[BLOCK_PREFIX + BLOCKS + block] = invalid;
        work[BLOCK_PREFIX + 2 * BLOCKS + block] = low;
        work[BLOCK_PREFIX + 3 * BLOCKS + block] = high;
        work[BLOCK_PREFIX + 4 * BLOCKS + block] = orderErrors;
    }
}
template <bool Invalid>
COLLECTION_AIV_GLOBAL void ScanBlocksKernel(GM_ADDR work)
{
    AscendC::Simt::VF_CALL<ScanBlocksSimt<Invalid>>(AscendC::Simt::Dim3{THREADS}, (__gm__ uint64_t*)work);
}
template <bool Invalid>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ScanSimt(__gm__ uint64_t* work)
{
    if (AscendC::Simt::GetThreadIdx() != 0) {
        return;
    }
    uint64_t sum = 0, invalid = 0, low = UINT64_MAX, high = 0, orderErrors = 0;
    for (uint32_t i = 0; i < BLOCKS; ++i) {
        uint64_t count = work[BLOCK_PREFIX + i];
        work[BLOCK_PREFIX + i] = sum;
        sum += count;
        if constexpr (Invalid) {
            invalid += work[BLOCK_PREFIX + BLOCKS + i];
            orderErrors += work[BLOCK_PREFIX + 4 * BLOCKS + i];
            low = work[BLOCK_PREFIX + 2 * BLOCKS + i] < low ? work[BLOCK_PREFIX + 2 * BLOCKS + i] : low;
            high = work[BLOCK_PREFIX + 3 * BLOCKS + i] > high ? work[BLOCK_PREFIX + 3 * BLOCKS + i] : high;
        }
    }
    work[PARTITIONS] = sum;
    if constexpr (Invalid) {
        work[PARTITIONS + 1] = invalid;
        work[BLOCK_PREFIX + 5 * BLOCKS] = low;
        work[BLOCK_PREFIX + 5 * BLOCKS + 1] = high;
        work[BLOCK_PREFIX + 5 * BLOCKS + 2] = orderErrors;
    }
}
template <bool Invalid>
COLLECTION_AIV_GLOBAL void ScanKernel(GM_ADDR work)
{
    AscendC::Simt::VF_CALL<ScanSimt<Invalid>>(AscendC::Simt::Dim3{THREADS}, (__gm__ uint64_t*)work);
}
} // namespace aclco::detail::static_multiset
