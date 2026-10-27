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

#include <cstdint>
#include <type_traits>

#include "detail/bloom_filter/kernels.h"

namespace aclco::detail::bloom_filter {

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_DEVICE void RoutePrivateKeyIf(__gm__ std::uint64_t* pairs, __gm__ Key* input, std::uint32_t index,
                                              __gm__ std::uint64_t* records, __ubuf__ std::uint32_t* localCounters,
                                              Policy const& policy, std::uint64_t numBlocks, std::uint32_t producerCount,
                                              std::uint32_t producer) noexcept
{
    Key const key = input[index];
    std::uint64_t const hash = policy.HashKey(key);
    std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
    std::uint32_t const bucket = upperHash >> (32U - BucketBits);
    std::uint32_t const ticket = PrivateRouteTicket<PackedCounters, RecordsPerSegment>(localCounters, bucket);
    if (ticket < RecordsPerSegment) {
        std::uint64_t const segment = static_cast<std::uint64_t>(bucket) * producerCount + producer;
        std::uint64_t const recordOffset = segment * RecordsPerSegment + ticket;
        asc_stcg(records + recordOffset, hash);
    } else {
        std::uint32_t const block = policy.BlockIndex(upperHash, numBlocks);
        AddPackedPairs<0>(pairs, policy, block, static_cast<std::uint32_t>(hash));
    }
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_DEVICE bool RoutePrivateStencilBlock(
    __gm__ std::uint64_t* pairs, __gm__ Key* input, __gm__ std::uint8_t* stencil, __gm__ std::uint64_t* records,
    __ubuf__ std::uint32_t* localCounters, Policy const& policy, std::uint64_t numBlocks,
    std::uint32_t producerCount, std::uint32_t producer, std::uint32_t base, std::uint32_t keyNum) noexcept
{
    constexpr std::uint32_t KeysPerThread = 4;
    std::uint32_t const remaining = keyNum - base;
    bool anySelected = false;
    // Four zero stencil bytes are the common all-zero case; one 32-bit load
    // rejects the whole block before touching keys or counters.
    if (stencil != nullptr && remaining >= KeysPerThread &&
        (reinterpret_cast<std::uintptr_t>(stencil) & (KeysPerThread - 1U)) == 0) {
        std::uint32_t const word = *reinterpret_cast<__gm__ std::uint32_t*>(stencil + base);
        if (word == 0) {
            return false;
        }
    }
    std::uint32_t const limit = remaining < KeysPerThread ? remaining : KeysPerThread;
    for (std::uint32_t j = 0; j < limit; ++j) {
        std::uint32_t const index = base + j;
        if (stencil == nullptr || stencil[index] != 0) {
            anySelected = true;
            RoutePrivateKeyIf<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>(
                pairs, input, index, records, localCounters, policy, numBlocks, producerCount, producer);
        }
    }
    return anySelected;
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_DEVICE void RoutePrivateH8V1If(__gm__ std::uint64_t* pairs, __gm__ Key* input,
                                               __gm__ std::uint8_t* stencil, __gm__ std::uint32_t* workspace,
                                               __gm__ std::uint8_t* selectedFlag,
                                               __ubuf__ std::uint32_t* localCounters, Policy const& policy,
                                               std::uint64_t numBlocks, std::uint32_t keyNum,
                                               std::uint32_t producerCount) noexcept
{
    constexpr std::uint32_t localCounterWords = PrivateRouteStorage<BucketBits, PackedCounters>::requiredWords;
    std::uint32_t const thread = AscendC::Simt::GetThreadIdx();
    std::uint32_t const threadCount = AscendC::Simt::GetThreadNum();
    std::uint32_t const producer = AscendC::Simt::GetBlockIdx();
    std::uint32_t const blockCount = AscendC::Simt::GetBlockNum();
    for (std::uint32_t word = thread; word < localCounterWords; word += threadCount) {
        localCounters[word] = 0U;
    }
    asc_syncthreads();

    std::uint64_t const countWords = static_cast<std::uint64_t>(localCounterWords) * producerCount;
    __gm__ std::uint64_t* records = reinterpret_cast<__gm__ std::uint64_t*>(workspace + countWords);
    constexpr std::uint32_t KeysPerThread = 4;
    std::uint32_t base = (producer * threadCount + thread) * KeysPerThread;
    std::uint32_t const stride = blockCount * threadCount * KeysPerThread;
    bool anySelected = false;
    while (base < keyNum) {
        anySelected |= RoutePrivateStencilBlock<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>(
            pairs, input, stencil, records, localCounters, policy, numBlocks, producerCount, producer, base, keyNum);
        if (keyNum - base <= stride) {
            break;
        }
        base += stride;
    }

    asc_syncthreads();
    for (std::uint32_t word = thread; word < localCounterWords; word += threadCount) {
        std::uint64_t const countOffset = static_cast<std::uint64_t>(word) * producerCount + producer;
        asc_stcg(workspace + countOffset, localCounters[word]);
    }
    if (anySelected && selectedFlag != nullptr) {
        asc_stcg(reinterpret_cast<__gm__ std::uint32_t*>(selectedFlag), 1U);
    }
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_ROUTE_THREAD_NUM) inline void AddPrivateRouteIfSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* stencil,
    __gm__ std::uint8_t* workspaceStorage, __gm__ std::uint8_t* selectedFlag, __ubuf__ std::uint32_t* localCounters,
    std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed, std::uint32_t producerCount)
{
    __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});
    RoutePrivateH8V1If<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>(
        pairs, input, stencil, workspace, selectedFlag, localCounters, policy, numBlocks,
        static_cast<std::uint32_t>(keyNum), producerCount);
}

template <class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock, std::uint32_t FixedBlockShift, std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket, bool PreserveExisting>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddPrivateApplySkipSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspaceStorage, __gm__ std::uint8_t* selectedFlag,
    __ubuf__ std::uint32_t* localWorkspace, std::uint64_t hashSeed, std::uint32_t producerCount)
{
  if (selectedFlag != nullptr && asc_ldcg(reinterpret_cast<__gm__ std::uint32_t*>(selectedFlag)) == 0) {
    return;
  }
  __gm__ ulonglong4* skipBlocks = reinterpret_cast<__gm__ ulonglong4*>(filter);
  __gm__ std::uint32_t* skipWorkspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
  Policy skipPolicy(typename Policy::Hasher{hashSeed});
  ApplyPrivateH8V1<Policy, BucketBits, RecordsPerSegment, PackedCounters, FixedSlotsPerBlock, FixedBlockShift,
                   BlocksPerTile, TilesPerBucket, PreserveExisting>(
    skipBlocks, skipWorkspace, localWorkspace, skipPolicy, producerCount);
}

template <class Policy, bool PreserveExisting, std::uint32_t BucketBits = finePrivateRouteBucketBits,
          std::uint32_t RecordsPerSegment = finePrivateRouteRecordsPerSegment, std::uint32_t FixedBlockShift = 6,
          std::uint32_t BlocksPerBucket = compactPrivateSparseBlocksPerBucket,
          std::uint32_t FixedSlots = compactPrivateSparseSlotsPerBlock>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddPrivateCompactSparseApplySkipSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspaceStorage, __gm__ std::uint8_t* selectedFlag,
    __ubuf__ std::uint32_t* localWorkspace, std::uint64_t hashSeed, std::uint32_t producerCount)
{
  if (selectedFlag != nullptr && asc_ldcg(reinterpret_cast<__gm__ std::uint32_t*>(selectedFlag)) == 0) {
    return;
  }
  __gm__ ulonglong4* compactBlocks = reinterpret_cast<__gm__ ulonglong4*>(filter);
  __gm__ std::uint32_t* compactCounters = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
  Policy compactPolicy(typename Policy::Hasher{hashSeed});
  ApplyPrivateCompactSparseH8V1<Policy, PreserveExisting, BucketBits, RecordsPerSegment, FixedBlockShift,
                                BlocksPerBucket, FixedSlots>(
    compactBlocks, compactCounters, localWorkspace, compactPolicy, producerCount);
}

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ACLCO_PRIVATE_ROUTE_COUNTERS(n) __ubuf__ std::uint32_t localCounters[n]
#else
#define ACLCO_PRIVATE_ROUTE_COUNTERS(n) \
    __ubuf__ std::uint32_t* localCounters = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0))
#endif


template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateRouteIf(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                                        __gm__ std::uint8_t* stencil, __gm__ std::uint8_t* workspace,
                                                        __gm__ std::uint8_t* selectedFlag, std::uint64_t numBlocks,
                                                        std::uint64_t keyNum, std::uint64_t hashSeed,
                                                        std::uint32_t producerCount)
{
    constexpr std::uint32_t
        localCounterWords = PrivateRouteStorage<BucketBits, PackedCounters>::requiredWords;
    ACLCO_PRIVATE_ROUTE_COUNTERS(localCounterWords);
    AscendC::Simt::VF_CALL<
        AddPrivateRouteIfSimt<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_ROUTE_THREAD_NUM}, filter, keys, stencil, workspace, selectedFlag,
        localCounters, numBlocks, keyNum, hashSeed, producerCount);
}

#undef ACLCO_PRIVATE_ROUTE_COUNTERS

template <class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock, std::uint32_t FixedBlockShift, std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket, bool PreserveExisting>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateApplySkip(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspace,
                                                          __gm__ std::uint8_t* selectedFlag, std::uint64_t hashSeed,
                                                          std::uint32_t producerCount)
{
  constexpr std::uint32_t applyWorkspaceWords =
    PrivateApplyStorage<FixedSlotsPerBlock, BlocksPerTile>::requiredWords;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
  __ubuf__ std::uint32_t applyWorkspace[applyWorkspaceWords];
#else
  __ubuf__ std::uint32_t* applyWorkspace = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
#endif
  AscendC::Simt::VF_CALL<AddPrivateApplySkipSimt<Policy, BucketBits, RecordsPerSegment, PackedCounters,
                                                 FixedSlotsPerBlock, FixedBlockShift, BlocksPerTile, TilesPerBucket,
                                                 PreserveExisting>>(
    AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, workspace, selectedFlag, applyWorkspace, hashSeed,
    producerCount);
}

template <class Policy, bool PreserveExisting, std::uint32_t BucketBits = finePrivateRouteBucketBits,
          std::uint32_t RecordsPerSegment = finePrivateRouteRecordsPerSegment, std::uint32_t FixedBlockShift = 6,
          std::uint32_t BlocksPerBucket = compactPrivateSparseBlocksPerBucket,
          std::uint32_t FixedSlots = compactPrivateSparseSlotsPerBlock>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateCompactSparseApplySkip(
  __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspace, __gm__ std::uint8_t* selectedFlag,
  std::uint64_t hashSeed, std::uint32_t producerCount)
{
  constexpr std::uint32_t compactWorkspaceWords =
    PrivateCompactSparseApplyStorage<BlocksPerBucket, FixedSlots>::requiredWords;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
  __ubuf__ std::uint32_t compactWorkspace[compactWorkspaceWords];
#else
  __ubuf__ std::uint32_t* compactWorkspace = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
#endif
  AscendC::Simt::VF_CALL<AddPrivateCompactSparseApplySkipSimt<Policy, PreserveExisting, BucketBits,
                                                              RecordsPerSegment, FixedBlockShift, BlocksPerBucket,
                                                              FixedSlots>>(
    AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, workspace, selectedFlag, compactWorkspace, hashSeed,
    producerCount);
}

}  // namespace aclco::detail::bloom_filter
