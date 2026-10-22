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
#include "utility/atomic_cas_wrap.h"

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

template <typename Key>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ClearSimt(__gm__ Key* table, __gm__ uint64_t* counts,
                                                               uint64_t capacity, Key empty)
{
    for (uint64_t i = ThreadId(); i < capacity; i += WORKERS) {
        if (table) {
            table[i * StaticMultisetRef<Key>::KEY_STRIDE] = empty;
        }
        // Empty slots start at one: the successful key claimant needs no counter
        // atomic; only later duplicates increment it. Queries check the key first.
        counts[i * StaticMultisetRef<Key>::COUNT_STRIDE] = 1;
    }
}
template <typename Key>
COLLECTION_AIV_GLOBAL void ClearKernel(GM_ADDR table, GM_ADDR counts, uint64_t capacity, Key empty)
{
    AscendC::Simt::VF_CALL<ClearSimt<Key>>(AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)table, (__gm__ uint64_t*)counts,
                                           capacity, empty);
}

template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void SelectSimt(__gm__ Key* keys, __gm__ Stencil* stencil, uint64_t n,
                                                                Key empty, __gm__ uint64_t* work, Predicate pred,
                                                                __gm__ Key* ordered, uint64_t size, uint64_t available)
{
    uint64_t id = ThreadId() / 32, lane = AscendC::Simt::GetThreadIdx() % 32;
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

template <typename Key>
COLLECTION_SIMT_DEVICE void InsertOne(Key key, Key empty, __gm__ Key* table, __gm__ uint64_t* counts,
                                      uint64_t tableCapacity, bool direct, uint64_t base)
{
    if (direct) {
        uint64_t slot = StaticMultisetRef<Key>::Encode(key) - base;
        AscendC::Simt::AtomicAdd(counts + slot * StaticMultisetRef<Key>::COUNT_STRIDE, uint64_t{1});
        return;
    }
    uint64_t slot = StaticMultisetRef<Key>::Hash(key) & (tableCapacity - 1);
    for (uint64_t visited = 0; visited < tableCapacity; ++visited) {
        Key old = AtomicCasWrap(table + slot * StaticMultisetRef<Key>::KEY_STRIDE, empty, key);
        if (old == empty) {
            return;
        }
        if (old == key) {
            AscendC::Simt::AtomicAdd(counts + slot * StaticMultisetRef<Key>::COUNT_STRIDE, uint64_t{1});
            return;
        }
        if (++slot == tableCapacity) {
            slot = 0;
        }
    }
}

template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void InsertSimt(__gm__ Key* keys, __gm__ Stencil* stencil, uint64_t n,
                                                                Key empty, __gm__ Key* table, __gm__ uint64_t* counts,
                                                                __gm__ Key* ordered, uint64_t capacity,
                                                                uint64_t tableCapacity, uint64_t size,
                                                                __gm__ uint64_t* work, Predicate pred, bool direct,
                                                                uint64_t base)
{
    // Selection established that every element is valid. Its stable position is
    // then the input index, including partial insertion when logical capacity fills.
    if (work[PARTITIONS] == n) {
        uint64_t accepted = n < capacity - size ? n : capacity - size;
        for (uint64_t i = ThreadId(); i < accepted; i += WORKERS) {
            Key key = keys[i];
            if constexpr (Conditional) {
                ordered[size + i] = key;
            }
            if (direct && work[BLOCK_PREFIX + 5 * BLOCKS + 2] == 0) {
                // A full input ordering check proved that this batch has unique keys.
                // With direct addressing, each counter has exactly one writer.
                uint64_t slot = StaticMultisetRef<Key>::Encode(key) - base;
                ++counts[slot * StaticMultisetRef<Key>::COUNT_STRIDE];
            } else {
                InsertOne(key, empty, table, counts, tableCapacity, direct, base);
            }
        }
        return;
    }
    uint64_t id = ThreadId() / 32, lane = AscendC::Simt::GetThreadIdx() % 32;
    uint64_t offset = work[id] + work[BLOCK_PREFIX + AscendC::Simt::GetBlockIdx()];
    uint64_t end = Begin(n, id + 1);
    for (uint64_t tile = Begin(n, id); tile < end; tile += 32) {
        uint64_t i = tile + lane;
        Key key = i < end ? keys[i] : empty;
        bool selected = key != empty;
        if constexpr (Conditional) {
            selected = selected && pred(stencil[i]);
        }
        uint64_t prefix = selected ? 1 : 0;
        for (uint32_t delta = 1; delta < 32; delta <<= 1) {
            uint64_t previous = AscendC::Simt::WarpShflUpSync(prefix, delta);
            if (lane >= delta) {
                prefix += previous;
            }
        }
        uint64_t position = offset + prefix - 1;
        if (selected && position < capacity - size) {
            ordered[size + position] = key;
            InsertOne(key, empty, table, counts, tableCapacity, direct, base);
        }
        offset += AscendC::Simt::WarpShflSync(prefix, 31);
    }
}
template <typename Key, typename Stencil, typename Predicate, bool Conditional>
COLLECTION_AIV_GLOBAL void InsertKernel(GM_ADDR keys, GM_ADDR stencil, uint64_t n, Key empty, GM_ADDR table,
                                        GM_ADDR counts, GM_ADDR ordered, uint64_t capacity, uint64_t tableCapacity,
                                        uint64_t size, GM_ADDR work, Predicate pred, bool direct, uint64_t base)
{
    AscendC::Simt::VF_CALL<InsertSimt<Key, Stencil, Predicate, Conditional>>(
        AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)keys, (__gm__ Stencil*)stencil, n, empty, (__gm__ Key*)table,
        (__gm__ uint64_t*)counts, (__gm__ Key*)ordered, capacity, tableCapacity, size, (__gm__ uint64_t*)work, pred,
        direct, base);
}

template <typename Key>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void RebuildSimt(__gm__ Key* ordered, uint64_t size, Key empty,
                                                                 __gm__ Key* table, __gm__ uint64_t* counts,
                                                                 uint64_t capacity)
{
    for (uint64_t i = ThreadId(); i < size; i += WORKERS) {
        InsertOne(ordered[i], empty, table, counts, capacity, false, 0);
    }
}
template <typename Key>
COLLECTION_AIV_GLOBAL void RebuildKernel(GM_ADDR ordered, uint64_t size, Key empty, GM_ADDR table, GM_ADDR counts,
                                         uint64_t capacity)
{
    AscendC::Simt::VF_CALL<RebuildSimt<Key>>(AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)ordered, size, empty,
                                             (__gm__ Key*)table, (__gm__ uint64_t*)counts, capacity);
}

// Materialize an exact interval representation before a general insertion.
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void MaterializeIntervalSimt(__gm__ uint64_t* counts, uint64_t length,
                                                                             uint64_t count)
{
    for (uint64_t i = ThreadId(); i < length; i += WORKERS) {
        counts[i] = count + 1;
    }
}
template <typename Key>
COLLECTION_AIV_GLOBAL void MaterializeIntervalKernel(GM_ADDR counts, uint64_t length, uint64_t count)
{
    AscendC::Simt::VF_CALL<MaterializeIntervalSimt>(AscendC::Simt::Dim3{THREADS}, (__gm__ uint64_t*)counts, length,
                                                    count);
}

// Mode 0: contains, 1: find, 2: count each, 3: count each outer.
template <typename Key, typename Stencil, typename Predicate, bool Conditional, int Mode>
COLLECTION_SIMT_VF LAUNCH_BOUND(QUERY_THREADS) inline void QuerySimt(__gm__ Key* keys, __gm__ Stencil* stencil,
                                                                     uint64_t n, Key empty, __gm__ Key* table,
                                                                     __gm__ uint64_t* counts, uint64_t capacity,
                                                                     __gm__ uint8_t* output, Predicate pred,
                                                                     bool direct, uint64_t base, uint64_t uniformCount,
                                                                     uint64_t uniformLength)
{
    StaticMultisetRef<Key> ref(table, counts, capacity, empty, direct, base, uniformCount, uniformLength);
    for (uint64_t i = ThreadId(); i < n; i += AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum()) {
        bool selected = true;
        if constexpr (Conditional) {
            selected = pred(stencil[i]);
        }
        if constexpr (Mode <= 1) {
            bool found = selected && ref.Contains(keys[i]);
            if constexpr (Mode == 0) {
                output[i] = found;
            } else {
                ((__gm__ Key*)output)[i] = found ? keys[i] : empty;
            }
        } else {
            uint64_t count = selected ? ref.Count(keys[i]) : 0;
            if constexpr (Mode == 2) {
                ((__gm__ uint64_t*)output)[i] = count;
            } else {
                ((__gm__ uint64_t*)output)[i] = count ? count : 1;
            }
        }
    }
}
template <typename Key, typename Stencil, typename Predicate, bool Conditional, int Mode>
COLLECTION_AIV_GLOBAL void QueryKernel(GM_ADDR keys, GM_ADDR stencil, uint64_t n, Key empty, GM_ADDR table,
                                       GM_ADDR counts, uint64_t capacity, GM_ADDR output, Predicate pred, bool direct,
                                       uint64_t base, uint64_t uniformCount, uint64_t uniformLength)
{
    AscendC::Simt::VF_CALL<QuerySimt<Key, Stencil, Predicate, Conditional, Mode>>(
        AscendC::Simt::Dim3{QUERY_THREADS}, (__gm__ Key*)keys, (__gm__ Stencil*)stencil, n, empty, (__gm__ Key*)table,
        (__gm__ uint64_t*)counts, capacity, (__gm__ uint8_t*)output, pred, direct, base, uniformCount, uniformLength);
}

template <typename Key>
COLLECTION_SIMT_VF LAUNCH_BOUND(QUERY_THREADS) inline void CountSimt(__gm__ Key* keys, uint64_t n, Key empty,
                                                                     __gm__ Key* table, __gm__ uint64_t* counts,
                                                                     uint64_t capacity, __gm__ uint64_t* total,
                                                                     bool direct, uint64_t base, uint64_t uniformCount,
                                                                     uint64_t uniformLength)
{
    StaticMultisetRef<Key> ref(table, counts, capacity, empty, direct, base, uniformCount, uniformLength);
    uint64_t sum = 0;
    for (uint64_t i = ThreadId(); i < n; i += AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum()) {
        sum += ref.Count(keys[i]);
    }
    for (uint32_t delta = 16; delta > 0; delta >>= 1) {
        sum += AscendC::Simt::WarpShflDownSync(sum, delta);
    }
    if ((AscendC::Simt::GetThreadIdx() % 32) == 0) {
        auto warp = AscendC::Simt::GetBlockIdx() * (QUERY_THREADS / 32)
                  + AscendC::Simt::GetThreadIdx() / 32;
        total[warp] = sum;
    }
}
template <typename Key>
COLLECTION_AIV_GLOBAL void CountKernel(GM_ADDR keys, uint64_t n, Key empty, GM_ADDR table, GM_ADDR counts,
                                       uint64_t capacity, GM_ADDR total, bool direct, uint64_t base,
                                       uint64_t uniformCount, uint64_t uniformLength)
{
    AscendC::Simt::VF_CALL<CountSimt<Key>>(AscendC::Simt::Dim3{QUERY_THREADS}, (__gm__ Key*)keys, n, empty,
                                           (__gm__ Key*)table, (__gm__ uint64_t*)counts, capacity,
                                           (__gm__ uint64_t*)total, direct, base, uniformCount, uniformLength);
}


COLLECTION_SIMT_VF LAUNCH_BOUND(32)
inline void CountFinalizeSimt(__gm__ uint64_t* partial,
                              __gm__ uint64_t* total,
                              uint32_t warps)
{
    uint32_t lane = AscendC::Simt::GetThreadIdx();
    uint64_t sum = 0;
    for (uint32_t i = lane; i < warps; i += 32) {
        sum += partial[i];
    }
    for (uint32_t delta = 16; delta > 0; delta >>= 1) {
        sum += AscendC::Simt::WarpShflDownSync(sum, delta);
    }
    if (lane == 0) {
        total[0] = sum;
    }
}

COLLECTION_AIV_GLOBAL void CountFinalizeKernel(
    GM_ADDR partial, GM_ADDR total, uint32_t warps)
{
    AscendC::Simt::VF_CALL<CountFinalizeSimt>(
        AscendC::Simt::Dim3{32},
        (__gm__ uint64_t*)partial,
        (__gm__ uint64_t*)total, warps);
}

COLLECTION_SIMT_DEVICE uint64_t InclusiveWarpSum(uint64_t prefix, uint64_t lane)
{
    for (uint32_t delta = 1; delta < 32; delta <<= 1) {
        uint64_t previous = AscendC::Simt::WarpShflUpSync(prefix, delta);
        if (lane >= delta) {
            prefix += previous;
        }
    }
    return prefix;
}

template <typename Key, bool Write>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void RetrieveSimt(__gm__ Key* keys, uint64_t n, Key empty,
                                                                  __gm__ Key* table, __gm__ uint64_t* counts,
                                                                  uint64_t capacity, __gm__ uint64_t* work,
                                                                  __gm__ Key* probes, __gm__ Key* matches, bool direct,
                                                                  uint64_t base, uint64_t uniformCount,
                                                                  uint64_t uniformLength)
{
    uint64_t id = ThreadId() / 32, lane = AscendC::Simt::GetThreadIdx() % 32;
    uint64_t offset = 0;
    if constexpr (Write) {
        offset = work[id] + work[BLOCK_PREFIX + AscendC::Simt::GetBlockIdx()];
    }
    StaticMultisetRef<Key> ref(table, counts, capacity, empty, direct, base, uniformCount, uniformLength);
    uint64_t end = Begin(n, id + 1);
    for (uint64_t tile = Begin(n, id); tile < end; tile += 32) {
        uint64_t i = tile + lane;
        Key key = i < end ? keys[i] : empty;
        uint64_t count = ref.Count(key);
        if constexpr (Write) {
            uint64_t prefix = count, tileCount = 0;
            if (uniformCount) {
                // Every match has the same multiplicity: a membership bit mask is an
                // exact compact representation of the per-lane count vector.
                uint32_t mask = AscendC::Simt::WarpBallotSync(count != 0);
                uint32_t inclusiveMask = UINT32_MAX >> (31 - lane);
                prefix = static_cast<uint64_t>(AscendC::Simt::Popc(mask & inclusiveMask)) * uniformCount;
                tileCount = static_cast<uint64_t>(AscendC::Simt::Popc(mask)) * uniformCount;
            } else {
                prefix = InclusiveWarpSum(prefix, lane);
                tileCount = AscendC::Simt::WarpShflSync(prefix, 31);
            }
            for (uint64_t j = 0; j < count; ++j) {
                probes[offset + prefix - count + j] = key;
                matches[offset + prefix - count + j] = key;
            }
            offset += tileCount;
        } else {
            offset += count;
        }
    }
    if constexpr (!Write) {
        for (uint32_t delta = 16; delta > 0; delta >>= 1) {
            offset += AscendC::Simt::WarpShflDownSync(offset, delta);
        }
        if (lane == 0) {
            work[id] = offset;
        }
    }
}
template <typename Key, bool Write>
COLLECTION_AIV_GLOBAL void RetrieveKernel(GM_ADDR keys, uint64_t n, Key empty, GM_ADDR table, GM_ADDR counts,
                                          uint64_t capacity, GM_ADDR work, GM_ADDR probes, GM_ADDR matches, bool direct,
                                          uint64_t base, uint64_t uniformCount, uint64_t uniformLength)
{
    AscendC::Simt::VF_CALL<RetrieveSimt<Key, Write>>(AscendC::Simt::Dim3{THREADS}, (__gm__ Key*)keys, n, empty,
                                                     (__gm__ Key*)table, (__gm__ uint64_t*)counts, capacity,
                                                     (__gm__ uint64_t*)work, (__gm__ Key*)probes, (__gm__ Key*)matches,
                                                     direct, base, uniformCount, uniformLength);
}
template <typename Key, typename Equal, typename Hash, bool Outer>
COLLECTION_SIMT_VF LAUNCH_BOUND(QUERY_THREADS) inline void ProbeCountSimt(__gm__ Key* keys, uint64_t n, Key empty,
                                                                          __gm__ Key* table, __gm__ uint64_t* counts,
                                                                          uint64_t capacity, __gm__ uint64_t* output,
                                                                          Equal equal, Hash hash, bool direct,
                                                                          uint64_t base, uint64_t uniformCount,
                                                                          uint64_t uniformLength)
{
    StaticMultisetRef<Key> ref(table, counts, capacity, empty, direct, base, uniformCount, uniformLength);
    for (uint64_t i = ThreadId(); i < n; i += AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum()) {
        uint64_t count = ref.Count(keys[i], equal, hash);
        output[i] = Outer && count == 0 ? 1 : count;
    }
}
template <typename Key, typename Equal, typename Hash, bool Outer>
COLLECTION_AIV_GLOBAL void ProbeCountKernel(GM_ADDR keys, uint64_t n, Key empty, GM_ADDR table, GM_ADDR counts,
                                            uint64_t capacity, GM_ADDR output, Equal equal, Hash hash, bool direct,
                                            uint64_t base, uint64_t uniformCount, uint64_t uniformLength)
{
    AscendC::Simt::VF_CALL<ProbeCountSimt<Key, Equal, Hash, Outer>>(
        AscendC::Simt::Dim3{QUERY_THREADS}, (__gm__ Key*)keys, n, empty, (__gm__ Key*)table, (__gm__ uint64_t*)counts,
        capacity, (__gm__ uint64_t*)output, equal, hash, direct, base, uniformCount, uniformLength);
}
} // namespace aclco::detail::static_multiset
