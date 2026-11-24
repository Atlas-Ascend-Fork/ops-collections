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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <limits>

// CANN's public kernel entry must precede BloomFilterRef's SIMT dependencies.
#include "kernel_operator.h"

#include "bloom_filter_ref.h"
#include "macros.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace aclco {

constexpr std::uint32_t BLOOM_FILTER_THREAD_NUM = 1024;
constexpr std::uint32_t BLOOM_FILTER_ROUTE_THREAD_NUM = 512;
static_assert(BLOOM_FILTER_THREAD_NUM % 8 == 0, "BloomFilter H8V1 requires a thread count divisible by 8");
static_assert(BLOOM_FILTER_ROUTE_THREAD_NUM * 2 == BLOOM_FILTER_THREAD_NUM,
              "BloomFilter routed Add requires two keys per route thread");

namespace detail::bloom_filter {

constexpr std::uint64_t maximumRoutedCounterWords = 1ULL << 26;
constexpr std::uint64_t maximumRoutedSlotWords = 5ULL << 26;
constexpr std::uint32_t denseRoutedSlotsPerBlock = 95;
constexpr std::uint32_t mediumRoutedSlotsPerBlock = 20;
constexpr std::uint32_t sparseRoutedSlotsPerBlock = 5;
constexpr std::uint64_t maximumRoutedWorkspaceWords = maximumRoutedCounterWords + maximumRoutedSlotWords;
constexpr std::uint64_t minimumPrivateRoutedKeys = 1ULL << 20;
constexpr std::uint32_t maximumPrivateRouteProducers = 128;
constexpr std::uint32_t densePrivateRouteBucketBits = 13;
constexpr std::uint32_t densePrivateRouteBucketCount = 1U << densePrivateRouteBucketBits;
constexpr std::uint32_t densePrivateRouteRecordsPerSegment = 240;
constexpr std::uint32_t finePrivateRouteBucketBits = 15;
constexpr std::uint32_t finePrivateRouteBucketCount = 1U << finePrivateRouteBucketBits;
constexpr std::uint32_t finePrivateRouteRecordsPerSegment = 80;
constexpr std::uint32_t privateRoutePackedCounterWords = finePrivateRouteBucketCount / 4U;
constexpr std::uint32_t privateRouteOverflowFlag = 0x80U;
constexpr std::uint32_t privateRouteCountMask = 0x7fU;
constexpr std::uint32_t privateRouteOverflowPatternWords = 8;
constexpr std::uint32_t privateSparseSlotsPerBlock = 5;
constexpr std::uint32_t compactPrivateSparseSlotsPerBlock = 6;
constexpr std::uint32_t compactPrivateSparseBlocksPerBucket = 2048;
constexpr std::uint32_t privateApplyMaximumRecordWords = 1024U * (privateSparseSlotsPerBlock + 1U +
                                                                  privateRouteOverflowPatternWords);
constexpr std::uint32_t privateApplyLocalWords = 14336;
constexpr std::uint32_t privateDynamicUbAlignmentBytes = 32;
constexpr std::uint32_t maximumPrivateDynamicUbBytes = (256U - 8U - 32U) * 1024U;

COLLECTION_HOST_DEVICE constexpr std::uint32_t AlignPrivateDynamicUbBytes(std::uint32_t bytes) noexcept
{
    return (bytes + privateDynamicUbAlignmentBytes - 1U) & ~(privateDynamicUbAlignmentBytes - 1U);
}

template <std::uint32_t BucketBits, bool PackedCounters>
struct PrivateRouteStorage {
    static_assert(BucketBits != 0 && BucketBits < 32, "private route bucket bits are invalid");
    static constexpr std::uint32_t bucketCount = 1U << BucketBits;
    static_assert(!PackedCounters || bucketCount % 4U == 0, "packed private route counters require groups of four");
    static constexpr std::uint32_t requiredWords = PackedCounters ? bucketCount / 4U : bucketCount;
    static constexpr std::uint64_t requiredBytes64 = static_cast<std::uint64_t>(requiredWords) * sizeof(std::uint32_t);
    static_assert(requiredBytes64 <= maximumPrivateDynamicUbBytes,
                  "private route counters exceed available dynamic UB");
    static constexpr std::uint32_t requiredBytes = static_cast<std::uint32_t>(requiredBytes64);
    static constexpr std::uint32_t launchBytes = AlignPrivateDynamicUbBytes(requiredBytes);
};

template <std::uint32_t FixedSlotsPerBlock, std::uint32_t BlocksPerTile>
struct PrivateApplyStorage {
    static_assert(FixedSlotsPerBlock != 0 && BlocksPerTile != 0, "private Apply storage dimensions must be non-zero");
    static constexpr std::uint64_t recordWords64 = static_cast<std::uint64_t>(FixedSlotsPerBlock) + 1U +
                                                   privateRouteOverflowPatternWords;
    static_assert(recordWords64 <= std::numeric_limits<std::uint32_t>::max(),
                  "private Apply record size exceeds uint32");
    static constexpr std::uint32_t recordWords = static_cast<std::uint32_t>(recordWords64);
    static constexpr std::uint64_t requiredWords64 = static_cast<std::uint64_t>(BlocksPerTile) * recordWords;
    static constexpr std::uint64_t requiredBytes64 = requiredWords64 * sizeof(std::uint32_t);
    static_assert(requiredBytes64 <= maximumPrivateDynamicUbBytes, "private Apply records exceed available dynamic UB");
    static constexpr std::uint32_t requiredWords = static_cast<std::uint32_t>(requiredWords64);
    static constexpr std::uint32_t requiredBytes = static_cast<std::uint32_t>(requiredBytes64);
    static constexpr std::uint32_t launchBytes = AlignPrivateDynamicUbBytes(requiredBytes);
};

template <std::uint32_t BlocksPerBucket, std::uint32_t FixedSlots>
struct PrivateCompactSparseApplyStorage {
    static_assert(BlocksPerBucket != 0 && FixedSlots != 0, "compact private Apply storage dimensions must be non-zero");
    static constexpr std::uint64_t recordWords64 = static_cast<std::uint64_t>(FixedSlots) + 1U;
    static_assert(recordWords64 <= std::numeric_limits<std::uint32_t>::max(),
                  "compact private Apply record size exceeds uint32");
    static constexpr std::uint32_t recordWords = static_cast<std::uint32_t>(recordWords64);
    static constexpr std::uint64_t requiredWords64 = static_cast<std::uint64_t>(BlocksPerBucket) * recordWords;
    static constexpr std::uint64_t requiredBytes64 = requiredWords64 * sizeof(std::uint32_t);
    static_assert(requiredBytes64 <= maximumPrivateDynamicUbBytes,
                  "compact private Apply records exceed available dynamic UB");
    static constexpr std::uint32_t requiredWords = static_cast<std::uint32_t>(requiredWords64);
    static constexpr std::uint32_t requiredBytes = static_cast<std::uint32_t>(requiredBytes64);
    static constexpr std::uint32_t launchBytes = AlignPrivateDynamicUbBytes(requiredBytes);
};

static_assert(maximumRoutedWorkspaceWords <= std::numeric_limits<std::uint32_t>::max(),
              "BloomFilter routed workspace offsets must fit in uint32");
static_assert(densePrivateRouteBucketCount == privateRoutePackedCounterWords,
              "private routes must use the same number of UB counter words");
static_assert(finePrivateRouteRecordsPerSegment <= privateRouteCountMask,
              "packed private counters require a seven-bit capacity");
static_assert(privateApplyMaximumRecordWords == privateApplyLocalWords,
              "private sparse Apply must fit exactly in the reserved UB");
static_assert(compactPrivateSparseBlocksPerBucket * (compactPrivateSparseSlotsPerBlock + 1U) == privateApplyLocalWords,
              "compact private sparse Apply must fit exactly in the reserved UB");
static_assert(PrivateRouteStorage<densePrivateRouteBucketBits, false>::launchBytes == 32768U,
              "dense private Route dynamic UB size changed unexpectedly");
static_assert(PrivateRouteStorage<finePrivateRouteBucketBits, true>::launchBytes == 32768U,
              "fine private Route dynamic UB size changed unexpectedly");
static_assert(PrivateApplyStorage<denseRoutedSlotsPerBlock, 128>::launchBytes == 53248U,
              "dense private Apply dynamic UB size changed unexpectedly");
static_assert(PrivateApplyStorage<mediumRoutedSlotsPerBlock, 256>::launchBytes == 29696U,
              "medium private Apply dynamic UB size changed unexpectedly");
static_assert(PrivateApplyStorage<privateSparseSlotsPerBlock, 1024>::launchBytes == 57344U,
              "sparse private Apply dynamic UB size changed unexpectedly");
static_assert(PrivateCompactSparseApplyStorage<compactPrivateSparseBlocksPerBucket,
                                               compactPrivateSparseSlotsPerBlock>::launchBytes == 57344U,
              "compact private Apply dynamic UB size changed unexpectedly");

COLLECTION_SIMT_DEVICE std::uint64_t GlobalThreadIndex() noexcept
{
    return static_cast<std::uint64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
           AscendC::Simt::GetThreadIdx();
}

COLLECTION_SIMT_DEVICE std::uint64_t TotalThreadCount() noexcept
{
    return static_cast<std::uint64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
}

template <typename Index, class Key, class Ref>
COLLECTION_SIMT_DEVICE void AddIndependent(__gm__ Key* input, Ref const& ref, Index keyNum, Index start,
                                           Index stride) noexcept
{
    static_assert(std::numeric_limits<Index>::is_integer && !std::numeric_limits<Index>::is_signed,
                  "BloomFilter Add index must be unsigned");

    Index i = start;
    while (i < keyNum) {
        Key const key = input[i];
        // Keep adjacent lanes on independently hashed keys so their atomic word
        // updates are not forced into the same 32-byte BloomFilter block.
        ref.Add(key);
        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <std::uint32_t PairIndex, class Policy>
COLLECTION_SIMT_DEVICE void AddPackedPairs(__gm__ std::uint64_t* pairs, Policy const& policy, std::uint32_t block,
                                           std::uint32_t lowerHash) noexcept
{
    constexpr std::uint32_t pairCount = Policy::wordsPerBlock / 2;
    if constexpr (PairIndex < pairCount) {
        constexpr std::uint32_t lowerWord = PairIndex * 2;
        constexpr std::uint32_t upperWord = lowerWord + 1;
        std::uint64_t const lowerPattern = static_cast<std::uint64_t>(
            policy.template WordPattern<lowerWord>(lowerHash));
        std::uint64_t const upperPattern = static_cast<std::uint64_t>(
            policy.template WordPattern<upperWord>(lowerHash));
        std::uint64_t const packedPattern = lowerPattern | (upperPattern << 32);
        std::uint64_t const offset = static_cast<std::uint64_t>(block) * pairCount + PairIndex;
        (void)asc_atomic_or(pairs + offset, packedPattern);
        AddPackedPairs<PairIndex + 1>(pairs, policy, block, lowerHash);
    }
}

template <typename Index, class Key, class Policy>
COLLECTION_SIMT_DEVICE void AddIndependentPackedH8V1(__gm__ std::uint64_t* pairs, __gm__ Key* input,
                                                     Policy const& policy, std::uint64_t numBlocks, Index keyNum,
                                                     Index start, Index stride) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter packed Add requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter packed Add requires 32-bit words");
    static_assert(Policy::patternBits == 8, "BloomFilter packed Add requires eight pattern bits");
    static_assert(Policy::addHorizontalLayout == 8 && Policy::addVerticalLayout == 1,
                  "BloomFilter packed Add requires an 8x1 layout");
    static_assert(!Policy::conditionalAdd, "BloomFilter packed Add does not support conditional Add");
    static_assert(std::numeric_limits<Index>::is_integer && !std::numeric_limits<Index>::is_signed,
                  "BloomFilter packed Add index must be unsigned");

    Index i = start;
    while (i < keyNum) {
        Key const key = input[i];
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = policy.BlockIndex(upperHash, numBlocks);
        // Ascend 950PR&950DT products are little-endian. Packing adjacent U32 masks in low/high
        // halves preserves the public eight-U32-word H8V1 storage layout while
        // reducing the number of GM atomics from eight to four per key.
        AddPackedPairs<0>(pairs, policy, block, lowerHash);
        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <std::uint32_t PairIndex, class Policy>
COLLECTION_SIMT_DEVICE std::uint64_t PackedPairPattern(Policy const& policy, std::uint32_t lowerHash) noexcept
{
    static_assert(PairIndex < Policy::wordsPerBlock / 2, "BloomFilter packed pair index is out of range");
    constexpr std::uint32_t lowerWord = PairIndex * 2;
    constexpr std::uint32_t upperWord = lowerWord + 1;
    std::uint64_t const lowerPattern = static_cast<std::uint64_t>(policy.template WordPattern<lowerWord>(lowerHash));
    std::uint64_t const upperPattern = static_cast<std::uint64_t>(policy.template WordPattern<upperWord>(lowerHash));
    return lowerPattern | (upperPattern << 32);
}

template <class Policy>
COLLECTION_SIMT_DEVICE void AccumulateWordPatterns(Policy const& policy, std::uint32_t lowerHash, std::uint32_t& word0,
                                                   std::uint32_t& word1, std::uint32_t& word2, std::uint32_t& word3,
                                                   std::uint32_t& word4, std::uint32_t& word5, std::uint32_t& word6,
                                                   std::uint32_t& word7) noexcept
{
    word0 |= policy.template WordPattern<0>(lowerHash);
    word1 |= policy.template WordPattern<1>(lowerHash);
    word2 |= policy.template WordPattern<2>(lowerHash);
    word3 |= policy.template WordPattern<3>(lowerHash);
    word4 |= policy.template WordPattern<4>(lowerHash);
    word5 |= policy.template WordPattern<5>(lowerHash);
    word6 |= policy.template WordPattern<6>(lowerHash);
    word7 |= policy.template WordPattern<7>(lowerHash);
}

template <std::uint32_t FixedSlotsPerBlock>
COLLECTION_HOST_DEVICE constexpr std::uint32_t RoutedSlotsPerBlock(std::uint32_t slotsPerBlock) noexcept
{
    if constexpr (FixedSlotsPerBlock != 0) {
        (void)slotsPerBlock;
        return FixedSlotsPerBlock;
    }
    return slotsPerBlock;
}

template <std::uint32_t FixedSlotsPerBlock>
COLLECTION_SIMT_DEVICE std::uint32_t RoutedSlotOffset(std::uint32_t block, std::uint32_t ticket,
                                                      std::uint32_t blockCount, std::uint32_t slotsPerBlock) noexcept
{
    std::uint32_t const capacity = RoutedSlotsPerBlock<FixedSlotsPerBlock>(slotsPerBlock);
    return blockCount + block * capacity + ticket;
}

template <class Policy, std::uint32_t FixedSlotsPerBlock>
COLLECTION_SIMT_DEVICE void CommitRoutedPackedH8V1(__gm__ std::uint64_t* pairs, __gm__ std::uint32_t* workspace,
                                                   Policy const& policy, std::uint32_t blockCount,
                                                   std::uint32_t slotsPerBlock, std::uint32_t block,
                                                   std::uint32_t lowerHash, std::uint32_t ticket) noexcept
{
    std::uint32_t const capacity = RoutedSlotsPerBlock<FixedSlotsPerBlock>(slotsPerBlock);
    if (ticket < capacity) {
        // Counters occupy one compact prefix so random atomics have the smallest
        // possible L2 working set. Slots remain block-major in the suffix so
        // Apply consumes every retained hash for a block from one local region.
        std::uint32_t const slotOffset = RoutedSlotOffset<FixedSlotsPerBlock>(block, ticket, blockCount, slotsPerBlock);
        asc_stcg(workspace + slotOffset, lowerHash);
    } else {
        // Overflow is uncommon at the selected capacity. Updating it here is
        // exact and avoids an unbounded route buffer or a full repair pass.
        AddPackedPairs<0>(pairs, policy, block, lowerHash);
    }
}

template <class Policy, bool PowerOfTwoBlocks, std::uint32_t FixedBlockShift = 0>
COLLECTION_SIMT_DEVICE std::uint32_t RoutedBlockIndex(Policy const& policy, std::uint32_t upperHash,
                                                      std::uint64_t numBlocks, std::uint32_t powerOfTwoShift) noexcept
{
    if constexpr (FixedBlockShift != 0) {
        static_assert(PowerOfTwoBlocks, "a fixed BloomFilter block shift requires power-of-two blocks");
        static_assert(FixedBlockShift < 32, "a fixed BloomFilter block shift must be less than 32");
        (void)policy;
        (void)numBlocks;
        (void)powerOfTwoShift;
        return upperHash >> FixedBlockShift;
    } else if constexpr (PowerOfTwoBlocks) {
        (void)policy;
        (void)numBlocks;
        // numBlocks == 1 is represented by a shift of 32. Avoid the undefined
        // uint32 shift while preserving the exact multiply-high result.
        return powerOfTwoShift == 32 ? 0U : upperHash >> powerOfTwoShift;
    }
    (void)powerOfTwoShift;
    return policy.BlockIndex(upperHash, numBlocks);
}

template <class Key, class Policy, std::uint32_t FixedSlotsPerBlock, bool PowerOfTwoBlocks,
          std::uint32_t FixedBlockShift = 0>
COLLECTION_SIMT_DEVICE void RoutePackedH8V1(__gm__ std::uint64_t* pairs, __gm__ Key* input,
                                            __gm__ std::uint32_t* workspace, Policy const& policy,
                                            std::uint64_t numBlocks, std::uint32_t keyNum, std::uint32_t slotsPerBlock,
                                            std::uint32_t powerOfTwoShift, std::uint32_t start,
                                            std::uint32_t stride) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter routed Add requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter routed Add requires 32-bit words");
    static_assert(Policy::patternBits == 8, "BloomFilter routed Add requires eight pattern bits");
    static_assert(Policy::addHorizontalLayout == 8 && Policy::addVerticalLayout == 1,
                  "BloomFilter routed Add requires an 8x1 layout");
    static_assert(!Policy::conditionalAdd, "BloomFilter routed Add does not support conditional Add");

    std::uint32_t const blockCount = static_cast<std::uint32_t>(numBlocks);
    std::uint32_t i = start;
    constexpr std::uint32_t keysPerThread = 2;
    if (stride <= std::numeric_limits<std::uint32_t>::max() / keysPerThread) {
        std::uint32_t const batchStride = stride * keysPerThread;
        while (i < keyNum && keyNum - i > stride) {
            std::uint32_t const i1 = i + stride;

            Key const key0 = input[i];
            std::uint64_t const hash0 = policy.HashKey(key0);
            std::uint32_t const upperHash0 = static_cast<std::uint32_t>(hash0 >> 32);
            std::uint32_t const lowerHash0 = static_cast<std::uint32_t>(hash0);
            std::uint32_t const block0 = RoutedBlockIndex<Policy, PowerOfTwoBlocks, FixedBlockShift>(
                policy, upperHash0, numBlocks, powerOfTwoShift);
            std::uint32_t const ticket0 = asc_atomic_add(workspace + block0, 1U);

            // Hash and route an independent key before consuming ticket0. With a
            // 512-thread launch bound this exposes atomic latency with a larger
            // per-thread register budget than the former four-key/1024-thread
            // expansion.
            Key const key1 = input[i1];
            std::uint64_t const hash1 = policy.HashKey(key1);
            std::uint32_t const upperHash1 = static_cast<std::uint32_t>(hash1 >> 32);
            std::uint32_t const lowerHash1 = static_cast<std::uint32_t>(hash1);
            std::uint32_t const block1 = RoutedBlockIndex<Policy, PowerOfTwoBlocks, FixedBlockShift>(
                policy, upperHash1, numBlocks, powerOfTwoShift);
            std::uint32_t const ticket1 = asc_atomic_add(workspace + block1, 1U);

            CommitRoutedPackedH8V1<Policy, FixedSlotsPerBlock>(pairs, workspace, policy, blockCount, slotsPerBlock,
                                                               block0, lowerHash0, ticket0);
            CommitRoutedPackedH8V1<Policy, FixedSlotsPerBlock>(pairs, workspace, policy, blockCount, slotsPerBlock,
                                                               block1, lowerHash1, ticket1);

            if (keyNum - i <= batchStride) {
                return;
            }
            i += batchStride;
        }
    }

    while (i < keyNum) {
        Key const key = input[i];
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = RoutedBlockIndex<Policy, PowerOfTwoBlocks, FixedBlockShift>(
            policy, upperHash, numBlocks, powerOfTwoShift);
        std::uint32_t const ticket = asc_atomic_add(workspace + block, 1U);
        CommitRoutedPackedH8V1<Policy, FixedSlotsPerBlock>(pairs, workspace, policy, blockCount, slotsPerBlock, block,
                                                           lowerHash, ticket);
        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <class Policy>
COLLECTION_SIMT_DEVICE void RouteDenseI32SingleKey(__gm__ std::uint64_t* pairs, __gm__ typename Policy::KeyType* input,
                                                   __gm__ std::uint32_t* workspace, Policy const& policy,
                                                   std::uint32_t keyNum, std::uint32_t start,
                                                   std::uint32_t stride) noexcept
{
    using Key = typename Policy::KeyType;
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter routed Add requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter routed Add requires 32-bit words");
    static_assert(Policy::patternBits == 8, "BloomFilter routed Add requires eight pattern bits");
    static_assert(Policy::addHorizontalLayout == 8 && Policy::addVerticalLayout == 1,
                  "BloomFilter routed Add requires an 8x1 layout");
    static_assert(!Policy::conditionalAdd, "BloomFilter routed Add does not support conditional Add");
    static_assert(denseRoutedSlotsPerBlock + 1U == 96U, "dense BloomFilter routed records must contain 96 words");

    constexpr std::uint32_t blockCount = 1U << 20;
    std::uint32_t i = start;
    while (i < keyNum) {
        Key const key = input[i];
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);

        // The dedicated route is selected only for the 2^20-block dense layout.
        // One key per 1024-thread lane keeps the live register set small.
        std::uint32_t const block = upperHash >> 12;
        std::uint32_t const ticket = asc_atomic_add(workspace + block, 1U);
        CommitRoutedPackedH8V1<Policy, denseRoutedSlotsPerBlock>(pairs, workspace, policy, blockCount,
                                                                 denseRoutedSlotsPerBlock, block, lowerHash, ticket);

        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <std::uint32_t RecordsPerSegment>
COLLECTION_SIMT_DEVICE std::uint32_t PrivatePackedRouteTicket(__ubuf__ std::uint32_t* counters,
                                                              std::uint32_t bucket) noexcept
{
    static_assert(RecordsPerSegment <= privateRouteCountMask, "private route capacity does not fit in seven bits");
    constexpr std::uint32_t overflowTicket = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t const word = bucket >> 2;
    std::uint32_t const shift = (bucket & 3U) * 8U;
    std::uint32_t const byteMask = 0xffU << shift;
    std::uint32_t observed = counters[word];
    for (;;) {
        std::uint32_t const packed = (observed >> shift) & 0xffU;
        if ((packed & privateRouteOverflowFlag) != 0) {
            return overflowTicket;
        }

        std::uint32_t const count = packed & privateRouteCountMask;
        std::uint32_t const nextPacked = count < RecordsPerSegment ? count + 1U : packed | privateRouteOverflowFlag;
        std::uint32_t const desired = (observed & ~byteMask) | (nextPacked << shift);
        std::uint32_t const previous = asc_atomic_cas(counters + word, observed, desired);
        if (previous == observed) {
            return count < RecordsPerSegment ? count : overflowTicket;
        }
        // CAS operates on the complete U32. A lane updating one of the other
        // packed counters is therefore a retry, not a lost increment.
        observed = previous;
    }
}

template <bool PackedCounters, std::uint32_t RecordsPerSegment>
COLLECTION_SIMT_DEVICE std::uint32_t PrivateRouteTicket(__ubuf__ std::uint32_t* counters, std::uint32_t bucket) noexcept
{
    if constexpr (PackedCounters) {
        return PrivatePackedRouteTicket<RecordsPerSegment>(counters, bucket);
    }
    return asc_atomic_add(counters + bucket, 1U);
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_DEVICE void RoutePrivateH8V1(__gm__ std::uint64_t* pairs, __gm__ Key* input,
                                             __gm__ std::uint32_t* workspace, __ubuf__ std::uint32_t* localCounters,
                                             Policy const& policy, std::uint64_t numBlocks, std::uint32_t keyNum,
                                             std::uint32_t producerCount) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter private route requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter private route requires 32-bit words");
    static_assert(Policy::patternBits == 8, "BloomFilter private route requires eight pattern bits");
    static_assert(Policy::addHorizontalLayout == 8 && Policy::addVerticalLayout == 1,
                  "BloomFilter private route requires an 8x1 layout");
    static_assert(!Policy::conditionalAdd, "BloomFilter private route does not support conditional Add");
    static_assert(BucketBits != 0 && BucketBits < 32, "private route bucket bits are invalid");
    constexpr std::uint32_t bucketCount = 1U << BucketBits;
    constexpr std::uint32_t localCounterWords = PrivateRouteStorage<BucketBits, PackedCounters>::requiredWords;
    static_assert(localCounterWords != 0 && localCounterWords % 2U == 0,
                  "private route counters must keep records U64-aligned");
    static_assert(localCounterWords <= privateRoutePackedCounterWords, "private route counters exceed the reserved UB");

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
    std::uint32_t const start = producer * threadCount + thread;
    std::uint32_t const stride = blockCount * threadCount;
    std::uint32_t i = start;
    while (i < keyNum) {
        Key const key = input[i];
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const bucket = upperHash >> (32U - BucketBits);
        std::uint32_t const ticket = PrivateRouteTicket<PackedCounters, RecordsPerSegment>(localCounters, bucket);
        if (ticket < RecordsPerSegment) {
            // Form the potentially large GM offset only after the bounded ticket
            // check. This is also required for the route/apply OOB regression.
            std::uint64_t const segment = static_cast<std::uint64_t>(bucket) * producerCount + producer;
            std::uint64_t const recordOffset = segment * RecordsPerSegment + ticket;
            asc_stcg(records + recordOffset, hash);
        } else {
            std::uint32_t const block = policy.BlockIndex(upperHash, numBlocks);
            AddPackedPairs<0>(pairs, policy, block, static_cast<std::uint32_t>(hash));
        }

        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }

    asc_syncthreads();
    // Counts are overwritten on every Add. Records need no reset because Apply
    // consumes exactly the retained prefix described by these counts.
    for (std::uint32_t word = thread; word < localCounterWords; word += threadCount) {
        std::uint64_t const countOffset = static_cast<std::uint64_t>(word) * producerCount + producer;
        asc_stcg(workspace + countOffset, localCounters[word]);
    }
}

template <std::uint32_t BucketCount, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_DEVICE std::uint32_t PrivateSegmentCount(__gm__ std::uint32_t* counters, std::uint32_t bucket,
                                                         std::uint32_t producer, std::uint32_t producerCount) noexcept
{
    constexpr std::uint32_t overflowResult = 1U << 31;
    if constexpr (PackedCounters) {
        std::uint32_t const word = bucket >> 2;
        std::uint32_t const shift = (bucket & 3U) * 8U;
        std::uint64_t const countOffset = static_cast<std::uint64_t>(word) * producerCount + producer;
        std::uint32_t const packedWord = asc_ldcg(counters + countOffset);
        std::uint32_t const packed = (packedWord >> shift) & 0xffU;
        std::uint32_t const retained = packed & privateRouteCountMask;
        return retained | ((packed & privateRouteOverflowFlag) != 0 ? overflowResult : 0U);
    }

    std::uint64_t const countOffset = static_cast<std::uint64_t>(bucket) * producerCount + producer;
    std::uint32_t const routed = asc_ldcg(counters + countOffset);
    std::uint32_t const retained = routed < RecordsPerSegment ? routed : RecordsPerSegment;
    return retained | (routed > RecordsPerSegment ? overflowResult : 0U);
}

template <class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock, std::uint32_t FixedBlockShift, std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket, bool PreserveExisting>
COLLECTION_SIMT_DEVICE void ApplyPrivateH8V1(__gm__ ulonglong4* blocks, __gm__ std::uint32_t* workspace,
                                             __ubuf__ std::uint32_t* localWorkspace, Policy const& policy,
                                             std::uint32_t producerCount) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter private Apply requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter private Apply requires 32-bit words");
    static_assert(FixedSlotsPerBlock != 0, "BloomFilter private Apply requires fixed local slots");
    static_assert(FixedBlockShift != 0 && FixedBlockShift < 32,
                  "BloomFilter private Apply requires a fixed block shift");
    static_assert(BlocksPerTile != 0 && (BlocksPerTile & (BlocksPerTile - 1U)) == 0,
                  "BloomFilter private Apply tile must be a power of two");
    constexpr std::uint32_t bucketCount = 1U << BucketBits;
    constexpr std::uint32_t localCounterWords = PrivateRouteStorage<BucketBits, PackedCounters>::requiredWords;
    constexpr std::uint32_t blocksPerBucket = BlocksPerTile * TilesPerBucket;
    constexpr std::uint32_t localRecordWords = PrivateApplyStorage<FixedSlotsPerBlock, BlocksPerTile>::recordWords;
    constexpr std::uint32_t localWorkspaceWords = PrivateApplyStorage<FixedSlotsPerBlock, BlocksPerTile>::requiredWords;
    constexpr std::uint32_t totalMetadataOffset = 0;
    constexpr std::uint32_t overflowMetadataOffset = 1;
    constexpr std::uint32_t overflowResult = 1U << 31;
    constexpr std::uint32_t producerGroupWidth = PackedCounters ? 16U : 32U;
    static_assert(localCounterWords == privateRoutePackedCounterWords,
                  "private route counter prefix size changed unexpectedly");
    static_assert(static_cast<std::uint64_t>(bucketCount) * blocksPerBucket == (1ULL << (32U - FixedBlockShift)),
                  "private Apply block mapping must preserve every upper hash bit");
    static_assert(localWorkspaceWords <= privateApplyMaximumRecordWords,
                  "private Apply records exceed the reserved UB");
    static_assert(BLOOM_FILTER_THREAD_NUM % producerGroupWidth == 0,
                  "private Apply producer groups must divide the thread count");

    std::uint64_t const countWords = static_cast<std::uint64_t>(localCounterWords) * producerCount;
    __gm__ std::uint64_t* records = reinterpret_cast<__gm__ std::uint64_t*>(workspace + countWords);
    std::uint32_t const thread = AscendC::Simt::GetThreadIdx();
    std::uint32_t const threadCount = AscendC::Simt::GetThreadNum();
    std::uint32_t const core = AscendC::Simt::GetBlockIdx();
    std::uint32_t const coreCount = AscendC::Simt::GetBlockNum();
    std::uint32_t const lane = thread & 7U;
    std::uint32_t const group = thread >> 3;
    std::uint32_t const groupsPerCore = threadCount >> 3;
    std::uint32_t const producerLane = thread & (producerGroupWidth - 1U);
    std::uint32_t const producerGroup = thread / producerGroupWidth;
    std::uint32_t const producerGroupsPerCore = threadCount / producerGroupWidth;

    for (std::uint32_t bucket = core; bucket < bucketCount; bucket += coreCount) {
        if (thread == 0) {
            localWorkspace[totalMetadataOffset] = 0U;
            localWorkspace[overflowMetadataOffset] = 0U;
        }
        asc_syncthreads();

        for (std::uint32_t producer = producerGroup; producer < producerCount; producer += producerGroupsPerCore) {
            if (producerLane == 0) {
                std::uint32_t const count = PrivateSegmentCount<bucketCount, RecordsPerSegment, PackedCounters>(
                    workspace, bucket, producer, producerCount);
                std::uint32_t const retained = count & ~overflowResult;
                if (retained != 0) {
                    (void)asc_atomic_add(localWorkspace + totalMetadataOffset, retained);
                }
                if ((count & overflowResult) != 0) {
                    (void)asc_atomic_or(localWorkspace + overflowMetadataOffset, 1U);
                }
            }
        }
        asc_syncthreads();

        // Copy temporary metadata into per-thread registers before the complete
        // local record array (including words zero and one) is reused.
        std::uint32_t const totalRetained = localWorkspace[totalMetadataOffset];
        std::uint32_t const bucketOverflow = localWorkspace[overflowMetadataOffset];
        // No warp may clear words zero and one for the first tile until every
        // warp has captured the bucket metadata.
        asc_syncthreads();
        if (totalRetained == 0) {
            continue;
        }

        for (std::uint32_t tile = 0; tile < TilesPerBucket; ++tile) {
            for (std::uint32_t word = thread; word < localWorkspaceWords; word += threadCount) {
                localWorkspace[word] = 0U;
            }
            asc_syncthreads();

            for (std::uint32_t producer = producerGroup; producer < producerCount; producer += producerGroupsPerCore) {
                std::uint32_t count = 0;
                if (producerLane == 0) {
                    count = PrivateSegmentCount<bucketCount, RecordsPerSegment, PackedCounters>(
                        workspace, bucket, producer, producerCount);
                }
                count = asc_shfl(count, 0, producerGroupWidth);
                std::uint32_t const retained = count & ~overflowResult;
                for (std::uint32_t ticket = producerLane; ticket < retained; ticket += producerGroupWidth) {
                    std::uint64_t const segment = static_cast<std::uint64_t>(bucket) * producerCount + producer;
                    std::uint64_t const recordOffset = segment * RecordsPerSegment + ticket;
                    std::uint64_t const hash = asc_ldcg(records + recordOffset);
                    std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
                    std::uint32_t const fullLocalBlock = (upperHash >> FixedBlockShift) & (blocksPerBucket - 1U);
                    std::uint32_t const tileBegin = tile * BlocksPerTile;
                    if (fullLocalBlock < tileBegin || fullLocalBlock >= tileBegin + BlocksPerTile) {
                        continue;
                    }

                    std::uint32_t const localBlock = fullLocalBlock - tileBegin;
                    std::uint32_t const localOffset = localBlock * localRecordWords;
                    std::uint32_t const localTicket = asc_atomic_add(localWorkspace + localOffset, 1U);
                    std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
                    if (localTicket < FixedSlotsPerBlock) {
                        localWorkspace[localOffset + 1U + localTicket] = lowerHash;
                    } else {
                        std::uint32_t const overflowOffset = localOffset + 1U + FixedSlotsPerBlock;
                        (void)asc_atomic_or(localWorkspace + overflowOffset, policy.template WordPattern<0>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 1U,
                                            policy.template WordPattern<1>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 2U,
                                            policy.template WordPattern<2>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 3U,
                                            policy.template WordPattern<3>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 4U,
                                            policy.template WordPattern<4>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 5U,
                                            policy.template WordPattern<5>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 6U,
                                            policy.template WordPattern<6>(lowerHash));
                        (void)asc_atomic_or(localWorkspace + overflowOffset + 7U,
                                            policy.template WordPattern<7>(lowerHash));
                    }
                }
            }
            asc_syncthreads();

            for (std::uint32_t localBlock = group; localBlock < BlocksPerTile; localBlock += groupsPerCore) {
                std::uint32_t const localOffset = localBlock * localRecordWords;
                std::uint32_t const localRouted = localWorkspace[localOffset];
                std::uint32_t const localRetained = localRouted < FixedSlotsPerBlock ? localRouted : FixedSlotsPerBlock;
                if (localRetained == 0) {
                    continue;
                }

                std::uint32_t combined = 0;
                for (std::uint32_t slot = 0; slot < localRetained; ++slot) {
                    std::uint32_t const lowerHash = localWorkspace[localOffset + 1U + slot];
                    combined |= policy.WordPatternForLane(lane, lowerHash);
                }
                combined |= localWorkspace[localOffset + 1U + FixedSlotsPerBlock + lane];

                std::uint32_t const globalBlock = bucket * blocksPerBucket + tile * BlocksPerTile + localBlock;
                if constexpr (PreserveExisting) {
                    __gm__ std::uint32_t* words = reinterpret_cast<__gm__ std::uint32_t*>(blocks);
                    combined |= asc_ldcg(words + globalBlock * Policy::wordsPerBlock + lane);
                } else if (bucketOverflow != 0) {
                    // A route overflow can target any block in this bucket. Load only
                    // blocks that have retained records; overflow-only blocks are left
                    // untouched so their atomic updates cannot be overwritten.
                    __gm__ std::uint32_t* words = reinterpret_cast<__gm__ std::uint32_t*>(blocks);
                    combined |= asc_ldcg(words + globalBlock * Policy::wordsPerBlock + lane);
                }

                std::uint32_t const word0 = asc_shfl(combined, 0, 8);
                std::uint32_t const word1 = asc_shfl(combined, 1, 8);
                std::uint32_t const word2 = asc_shfl(combined, 2, 8);
                std::uint32_t const word3 = asc_shfl(combined, 3, 8);
                std::uint32_t const word4 = asc_shfl(combined, 4, 8);
                std::uint32_t const word5 = asc_shfl(combined, 5, 8);
                std::uint32_t const word6 = asc_shfl(combined, 6, 8);
                std::uint32_t const word7 = asc_shfl(combined, 7, 8);
                if (lane == 0) {
                    ulonglong4 output;
                    output.x = static_cast<std::uint64_t>(word0) | (static_cast<std::uint64_t>(word1) << 32);
                    output.y = static_cast<std::uint64_t>(word2) | (static_cast<std::uint64_t>(word3) << 32);
                    output.z = static_cast<std::uint64_t>(word4) | (static_cast<std::uint64_t>(word5) << 32);
                    output.w = static_cast<std::uint64_t>(word6) | (static_cast<std::uint64_t>(word7) << 32);
                    asc_stcg(blocks + globalBlock, output);
                }
            }
            // All warps must finish reading this tile before its UB records are
            // reused by the second 2048 MiB tile or the next bucket.
            asc_syncthreads();
        }
    }
}

template <class Policy, bool PreserveExisting, std::uint32_t BucketBits = finePrivateRouteBucketBits,
          std::uint32_t RecordsPerSegment = finePrivateRouteRecordsPerSegment, std::uint32_t FixedBlockShift = 6,
          std::uint32_t BlocksPerBucket = compactPrivateSparseBlocksPerBucket,
          std::uint32_t FixedSlots = compactPrivateSparseSlotsPerBlock>
COLLECTION_SIMT_DEVICE void ApplyPrivateCompactSparseH8V1(__gm__ ulonglong4* blocks, __gm__ std::uint32_t* workspace,
                                                          __ubuf__ std::uint32_t* localWorkspace, Policy const& policy,
                                                          std::uint32_t producerCount) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "compact private Apply requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "compact private Apply requires 32-bit words");
    constexpr std::uint32_t bucketCount = 1U << BucketBits;
    constexpr std::uint32_t localCounterWords = bucketCount / 4U;
    constexpr std::uint32_t
        localRecordWords = PrivateCompactSparseApplyStorage<BlocksPerBucket, FixedSlots>::recordWords;
    constexpr std::uint32_t
        localWorkspaceWords = PrivateCompactSparseApplyStorage<BlocksPerBucket, FixedSlots>::requiredWords;
    constexpr std::uint32_t overflowResult = 1U << 31;
    constexpr std::uint32_t producerGroupWidth = 16U;
    static_assert(BucketBits >= 3 && BucketBits < 32, "compact private Apply bucket bits are invalid");
    static_assert(localCounterWords % 2U == 0, "compact private Apply records must stay U64-aligned");
    static_assert(FixedBlockShift != 0 && FixedBlockShift < 32, "compact private Apply block shift is invalid");
    static_assert(BlocksPerBucket != 0 && (BlocksPerBucket & (BlocksPerBucket - 1U)) == 0,
                  "compact private Apply blocks must be a power of two");
    static_assert(static_cast<std::uint64_t>(bucketCount) * BlocksPerBucket == (1ULL << (32U - FixedBlockShift)),
                  "compact private Apply block mapping must preserve every upper hash bit");
    static_assert(localWorkspaceWords <= privateApplyLocalWords,
                  "compact private Apply records exceed the reserved UB");
    static_assert(BLOOM_FILTER_THREAD_NUM % producerGroupWidth == 0,
                  "compact private Apply producer groups must divide threads");

    std::uint64_t const countWords = static_cast<std::uint64_t>(localCounterWords) * producerCount;
    __gm__ std::uint64_t* records = reinterpret_cast<__gm__ std::uint64_t*>(workspace + countWords);
    __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(blocks);
    std::uint32_t const thread = AscendC::Simt::GetThreadIdx();
    std::uint32_t const threadCount = AscendC::Simt::GetThreadNum();
    std::uint32_t const core = AscendC::Simt::GetBlockIdx();
    std::uint32_t const coreCount = AscendC::Simt::GetBlockNum();
    std::uint32_t const producerLane = thread & (producerGroupWidth - 1U);
    std::uint32_t const producerGroup = thread / producerGroupWidth;
    std::uint32_t const producerGroupsPerCore = threadCount / producerGroupWidth;

    for (std::uint32_t bucket = core; bucket < bucketCount; bucket += coreCount) {
        // Slot words need no clear: atomic tickets make every retained slot a
        // fresh write, and output reads only that freshly written prefix.
        for (std::uint32_t localBlock = thread; localBlock < BlocksPerBucket; localBlock += threadCount) {
            localWorkspace[localBlock * localRecordWords] = 0U;
        }
        asc_syncthreads();

        bool wroteLocalOverflow = false;
        for (std::uint32_t producer = producerGroup; producer < producerCount; producer += producerGroupsPerCore) {
            std::uint32_t count = 0;
            if (producerLane == 0) {
                count = PrivateSegmentCount<bucketCount, RecordsPerSegment, true>(workspace, bucket, producer,
                                                                                  producerCount);
                if ((count & overflowResult) != 0) {
                    // Reuse block zero's high counter bit as the uniform indication
                    // that Route sent one or more hashes in this bucket directly to GM.
                    (void)asc_atomic_or(localWorkspace, overflowResult);
                }
            }
            count = asc_shfl(count, 0, producerGroupWidth);
            std::uint32_t const retained = count & ~overflowResult;
            for (std::uint32_t ticket = producerLane; ticket < retained; ticket += producerGroupWidth) {
                std::uint64_t const segment = static_cast<std::uint64_t>(bucket) * producerCount + producer;
                std::uint64_t const recordOffset = segment * RecordsPerSegment + ticket;
                std::uint64_t const hash = asc_ldcg(records + recordOffset);
                std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
                std::uint32_t const localBlock = (upperHash >> FixedBlockShift) & (BlocksPerBucket - 1U);
                std::uint32_t const localOffset = localBlock * localRecordWords;
                std::uint32_t const rawTicket = asc_atomic_add(localWorkspace + localOffset, 1U);
                std::uint32_t const localTicket = rawTicket & ~overflowResult;
                std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
                if (localTicket < FixedSlots) {
                    localWorkspace[localOffset + 1U + localTicket] = lowerHash;
                } else {
                    std::uint32_t const globalBlock = bucket * BlocksPerBucket + localBlock;
                    AddPackedPairs<0>(pairs, policy, globalBlock, lowerHash);
                    wroteLocalOverflow = true;
                }
            }
        }
        if (wroteLocalOverflow) {
            // Only writers need to publish their own prior GM atomics.
            asc_threadfence_block();
        }
        // The unique owner may reload and replace an overflowed block only after
        // every writer has published its atomics.
        asc_syncthreads();

        std::uint32_t const bucketOverflow = localWorkspace[0] & overflowResult;
        // One thread owns a complete block. This removes eight duplicate UB slot
        // reads and the lane shuffles used by the generic two-tile sparse Apply.
        for (std::uint32_t localBlock = thread; localBlock < BlocksPerBucket; localBlock += threadCount) {
            std::uint32_t const localOffset = localBlock * localRecordWords;
            std::uint32_t const localRouted = localWorkspace[localOffset] & ~overflowResult;
            std::uint32_t const localRetained = localRouted < FixedSlots ? localRouted : FixedSlots;
            if (localRetained == 0) {
                continue;
            }

            std::uint32_t word0 = 0;
            std::uint32_t word1 = 0;
            std::uint32_t word2 = 0;
            std::uint32_t word3 = 0;
            std::uint32_t word4 = 0;
            std::uint32_t word5 = 0;
            std::uint32_t word6 = 0;
            std::uint32_t word7 = 0;
            for (std::uint32_t slot = 0; slot < localRetained; ++slot) {
                std::uint32_t const lowerHash = localWorkspace[localOffset + 1U + slot];
                AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
            }

            std::uint32_t const globalBlock = bucket * BlocksPerBucket + localBlock;
            __gm__ std::uint32_t* words = reinterpret_cast<__gm__ std::uint32_t*>(blocks);
            std::uint64_t const wordOffset = static_cast<std::uint64_t>(globalBlock) * Policy::wordsPerBlock;
            if constexpr (PreserveExisting) {
                word0 |= asc_ldcg(words + wordOffset);
                word1 |= asc_ldcg(words + wordOffset + 1U);
                word2 |= asc_ldcg(words + wordOffset + 2U);
                word3 |= asc_ldcg(words + wordOffset + 3U);
                word4 |= asc_ldcg(words + wordOffset + 4U);
                word5 |= asc_ldcg(words + wordOffset + 5U);
                word6 |= asc_ldcg(words + wordOffset + 6U);
                word7 |= asc_ldcg(words + wordOffset + 7U);
            } else if (bucketOverflow != 0 || localRouted > FixedSlots) {
                word0 |= asc_ldcg(words + wordOffset);
                word1 |= asc_ldcg(words + wordOffset + 1U);
                word2 |= asc_ldcg(words + wordOffset + 2U);
                word3 |= asc_ldcg(words + wordOffset + 3U);
                word4 |= asc_ldcg(words + wordOffset + 4U);
                word5 |= asc_ldcg(words + wordOffset + 5U);
                word6 |= asc_ldcg(words + wordOffset + 6U);
                word7 |= asc_ldcg(words + wordOffset + 7U);
            }

            ulonglong4 output;
            output.x = static_cast<std::uint64_t>(word0) | (static_cast<std::uint64_t>(word1) << 32);
            output.y = static_cast<std::uint64_t>(word2) | (static_cast<std::uint64_t>(word3) << 32);
            output.z = static_cast<std::uint64_t>(word4) | (static_cast<std::uint64_t>(word5) << 32);
            output.w = static_cast<std::uint64_t>(word6) | (static_cast<std::uint64_t>(word7) << 32);
            asc_stcg(blocks + globalBlock, output);
        }
        // Prevent a fast warp from reusing UB while another still reads slots.
        asc_syncthreads();
    }
}

template <class Policy, std::uint32_t FixedSlotsPerBlock, bool PreserveExisting>
COLLECTION_SIMT_DEVICE void ApplyPackedH8V1(__gm__ ulonglong4* blocks, __gm__ std::uint32_t* workspace,
                                            Policy const& policy, std::uint64_t numBlocks, std::uint32_t slotsPerBlock,
                                            std::uint32_t start, std::uint32_t stride) noexcept
{
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter routed Add requires eight words per block");
    static_assert(sizeof(typename Policy::WordType) == sizeof(std::uint32_t),
                  "BloomFilter routed Add requires 32-bit words");
    static_assert(sizeof(ulonglong4) == Policy::wordsPerBlock * sizeof(typename Policy::WordType),
                  "BloomFilter routed Add requires one 32-byte vector per block");

    std::uint32_t const blockCount = static_cast<std::uint32_t>(numBlocks);
    std::uint32_t block = start;
    while (block < blockCount) {
        std::uint32_t const capacity = RoutedSlotsPerBlock<FixedSlotsPerBlock>(slotsPerBlock);
        std::uint32_t const routed = asc_ldcg(workspace + block);
        std::uint32_t const retained = routed < capacity ? routed : capacity;
        if (retained != 0) {
            std::uint32_t word0 = 0;
            std::uint32_t word1 = 0;
            std::uint32_t word2 = 0;
            std::uint32_t word3 = 0;
            std::uint32_t word4 = 0;
            std::uint32_t word5 = 0;
            std::uint32_t word6 = 0;
            std::uint32_t word7 = 0;
            std::uint32_t slotOffset = RoutedSlotOffset<FixedSlotsPerBlock>(block, 0U, blockCount, slotsPerBlock);
            if (capacity <= sparseRoutedSlotsPerBlock) {
                // Sparse layouts retain at most five keys per block. Sequential
                // conditionals remove loop control without keeping multiple hashes
                // live, which is important at a 1024-thread bound.
                std::uint32_t lowerHash = asc_ldcg(workspace + slotOffset);
                AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                if (retained > 1) {
                    ++slotOffset;
                    lowerHash = asc_ldcg(workspace + slotOffset);
                    AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                }
                if (retained > 2) {
                    ++slotOffset;
                    lowerHash = asc_ldcg(workspace + slotOffset);
                    AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                }
                if (retained > 3) {
                    ++slotOffset;
                    lowerHash = asc_ldcg(workspace + slotOffset);
                    AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                }
                if (retained > 4) {
                    ++slotOffset;
                    lowerHash = asc_ldcg(workspace + slotOffset);
                    AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                }
            } else {
                for (std::uint32_t slot = 0; slot < retained; ++slot) {
                    std::uint32_t const lowerHash = asc_ldcg(workspace + slotOffset);
                    AccumulateWordPatterns(policy, lowerHash, word0, word1, word2, word3, word4, word5, word6, word7);
                    if (slot + 1 < retained) {
                        ++slotOffset;
                    }
                }
            }

            // Route and apply are separate kernels on the same stream. When prior
            // contents must be preserved, the coherent load observes every
            // preceding overflow atomic. The sole writer for this block then
            // combines that value with every retained routed pattern.
            ulonglong4 combined;
            combined.x = static_cast<std::uint64_t>(word0) | (static_cast<std::uint64_t>(word1) << 32);
            combined.y = static_cast<std::uint64_t>(word2) | (static_cast<std::uint64_t>(word3) << 32);
            combined.z = static_cast<std::uint64_t>(word4) | (static_cast<std::uint64_t>(word5) << 32);
            combined.w = static_cast<std::uint64_t>(word6) | (static_cast<std::uint64_t>(word7) << 32);
            if constexpr (PreserveExisting) {
                ulonglong4 const stored = asc_ldcg(blocks + block);
                combined.x |= stored.x;
                combined.y |= stored.y;
                combined.z |= stored.z;
                combined.w |= stored.w;
            } else if (routed > capacity) {
                // Overflow is applied atomically by Route. Preserve those exact
                // updates even when the filter was known empty before this Add.
                ulonglong4 const stored = asc_ldcg(blocks + block);
                combined.x |= stored.x;
                combined.y |= stored.y;
                combined.z |= stored.z;
                combined.w |= stored.w;
            }
            asc_stcg(blocks + block, combined);
            // Leave the persistent workspace ready for the next same-stream Add.
            asc_stcg(workspace + block, 0U);
        }
        if (blockCount - block <= stride) {
            break;
        }
        block += stride;
    }
}

template <typename Index, class Key, class Policy, class Ref>
COLLECTION_SIMT_DEVICE void ContainsH8V1(__gm__ typename Policy::WordType* words, __gm__ Key* input,
                                         __gm__ std::uint8_t* output, Ref const& ref, Policy const& policy,
                                         std::uint64_t numBlocks, Index keyNum, Index start, Index stride) noexcept
{
    using Word = typename Policy::WordType;
    static_assert(Policy::wordsPerBlock == 8, "BloomFilter H8V1 Contains requires eight words per block");
    static_assert(Policy::patternBits == 8, "BloomFilter H8V1 Contains requires eight pattern bits");
    static_assert(Policy::containsHorizontalLayout == 8 && Policy::containsVerticalLayout == 1,
                  "BloomFilter H8V1 Contains requires an 8x1 layout");
    static_assert(!Policy::earlyExitContains, "BloomFilter H8V1 Contains does not support early exit");
    static_assert(std::numeric_limits<Index>::is_integer && !std::numeric_limits<Index>::is_signed &&
                      std::numeric_limits<Index>::digits >= 3,
                  "BloomFilter H8V1 Contains index must be an unsigned integer");

    std::uint32_t const lane = static_cast<std::uint32_t>(start & Index{7});
    Index groupBase = start - static_cast<Index>(lane);
    while (groupBase < keyNum) {
        Index const remaining = keyNum - groupBase;
        if (remaining < Index{8}) {
            if (static_cast<Index>(lane) < remaining) {
                Index const keyIndex = groupBase + static_cast<Index>(lane);
                Key const key = input[keyIndex];
                output[keyIndex] = static_cast<std::uint8_t>(ref.Contains(key) ? 1 : 0);
            }
            break;
        }

        Index const keyIndex = groupBase + static_cast<Index>(lane);
        Key const key = input[keyIndex];
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = policy.BlockIndex(upperHash, numBlocks);
        std::uint32_t resultMask = 0;
#pragma unroll 8
        for (std::uint32_t sourceLane = 0; sourceLane < 8; ++sourceLane) {
            std::uint32_t const sourceBlock = asc_shfl(block, sourceLane, 8);
            std::uint32_t const sourceLower = asc_shfl(lowerHash, sourceLane, 8);
            std::uint64_t const offset = static_cast<std::uint64_t>(sourceBlock) * Policy::wordsPerBlock + lane;
            Word const stored = words[offset];
            Word const pattern = policy.WordPatternForLane(lane, sourceLower);
            std::uint32_t match = static_cast<std::uint32_t>((stored & pattern) == pattern);
            match &= asc_shfl_xor(match, 1, 8);
            match &= asc_shfl_xor(match, 2, 8);
            match &= asc_shfl_xor(match, 4, 8);
            // The reduction result is identical in all eight lanes. Pack each
            // source key result once, then let every lane select its own bit.
            resultMask |= match << sourceLane;
        }
        output[keyIndex] = static_cast<std::uint8_t>((resultMask >> lane) & 1U);

        if (remaining <= stride) {
            break;
        }
        groupBase += stride;
    }
}

template <typename Word>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void ClearSimt(__gm__ std::uint8_t* filter,
                                                                               std::uint64_t numWords)
{
    __gm__ Word* words = reinterpret_cast<__gm__ Word*>(filter);
    std::uint64_t const start = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    for (std::uint64_t i = start; i < numWords; i += stride) {
        words[i] = Word{0};
    }
}

template <class Key, class Policy>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, std::uint64_t numBlocks, std::uint64_t keyNum,
    std::uint64_t hashSeed, std::uint32_t packedAtomicEnabled)
{
    using Word = typename Policy::WordType;
    __gm__ Word* words = reinterpret_cast<__gm__ Word*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    Policy policy(typename Policy::Hasher{hashSeed});
    BloomFilterRef<Key, std::uint64_t, Policy> ref(words, numBlocks, policy);

    std::uint64_t const start = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    if constexpr (Policy::wordsPerBlock == 8 && Policy::patternBits == 8 && Policy::addHorizontalLayout == 8 &&
                  Policy::addVerticalLayout == 1 && !Policy::conditionalAdd) {
        constexpr std::uint64_t uint32Max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
        if (packedAtomicEnabled != 0) {
            __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(filter);
            if (keyNum <= uint32Max && start <= uint32Max && stride <= uint32Max) {
                AddIndependentPackedH8V1<std::uint32_t>(
                    pairs, input, policy, numBlocks, static_cast<std::uint32_t>(keyNum),
                    static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(stride));
            } else {
                AddIndependentPackedH8V1<std::uint64_t>(pairs, input, policy, numBlocks, keyNum, start, stride);
            }
        } else if (keyNum <= uint32Max && start <= uint32Max && stride <= uint32Max) {
            AddIndependent<std::uint32_t>(input, ref, static_cast<std::uint32_t>(keyNum),
                                          static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(stride));
        } else {
            AddIndependent<std::uint64_t>(input, ref, keyNum, start, stride);
        }
    } else {
        for (std::uint64_t i = start; i < keyNum; i += stride) {
            Key const key = input[i];
            ref.Add(key);
        }
    }
}

template <class Key, class Policy>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddIfSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* stencil, std::uint64_t numBlocks,
    std::uint64_t keyNum, std::uint64_t hashSeed)
{
    using Word = typename Policy::WordType;
    __gm__ Word* words = reinterpret_cast<__gm__ Word*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    Policy policy(typename Policy::Hasher{hashSeed});
    BloomFilterRef<Key, std::uint64_t, Policy> ref(words, numBlocks, policy);

    std::uint64_t i = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    while (i < keyNum) {
        if (stencil[i] != 0) {
            Key const key = input[i];
            ref.Add(key);
        }
        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_ROUTE_THREAD_NUM) inline void AddPrivateRouteSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* workspaceStorage,
    __ubuf__ std::uint32_t* localCounters, std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed,
    std::uint32_t producerCount)
{
    __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    RoutePrivateH8V1<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>(
        pairs, input, workspace, localCounters, policy, numBlocks, static_cast<std::uint32_t>(keyNum), producerCount);
}

template <class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock, std::uint32_t FixedBlockShift, std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket, bool PreserveExisting>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddPrivateApplySimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspaceStorage, __ubuf__ std::uint32_t* localWorkspace,
    std::uint64_t hashSeed, std::uint32_t producerCount)
{
    __gm__ ulonglong4* blocks = reinterpret_cast<__gm__ ulonglong4*>(filter);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    ApplyPrivateH8V1<Policy, BucketBits, RecordsPerSegment, PackedCounters, FixedSlotsPerBlock, FixedBlockShift,
                     BlocksPerTile, TilesPerBucket, PreserveExisting>(blocks, workspace, localWorkspace, policy,
                                                                      producerCount);
}

template <class Policy, bool PreserveExisting, std::uint32_t BucketBits = finePrivateRouteBucketBits,
          std::uint32_t RecordsPerSegment = finePrivateRouteRecordsPerSegment, std::uint32_t FixedBlockShift = 6,
          std::uint32_t BlocksPerBucket = compactPrivateSparseBlocksPerBucket,
          std::uint32_t FixedSlots = compactPrivateSparseSlotsPerBlock>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddPrivateCompactSparseApplySimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspaceStorage, __ubuf__ std::uint32_t* localWorkspace,
    std::uint64_t hashSeed, std::uint32_t producerCount)
{
    __gm__ ulonglong4* blocks = reinterpret_cast<__gm__ ulonglong4*>(filter);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    ApplyPrivateCompactSparseH8V1<Policy, PreserveExisting, BucketBits, RecordsPerSegment, FixedBlockShift,
                                  BlocksPerBucket, FixedSlots>(blocks, workspace, localWorkspace, policy,
                                                               producerCount);
}

template <class Key, class Policy, std::uint32_t FixedSlotsPerBlock, bool PowerOfTwoBlocks,
          std::uint32_t FixedBlockShift = 0>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_ROUTE_THREAD_NUM) inline void AddRoutePackedSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* workspaceStorage,
    std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed, std::uint32_t slotsPerBlock,
    std::uint32_t powerOfTwoShift)
{
    __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    RoutePackedH8V1<Key, Policy, FixedSlotsPerBlock, PowerOfTwoBlocks, FixedBlockShift>(
        pairs, input, workspace, policy, numBlocks, static_cast<std::uint32_t>(keyNum), slotsPerBlock, powerOfTwoShift,
        static_cast<std::uint32_t>(GlobalThreadIndex()), static_cast<std::uint32_t>(TotalThreadCount()));
}

template <class Policy>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddDenseI32RouteSingleSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* workspaceStorage, std::uint64_t keyNum,
    std::uint64_t hashSeed)
{
    using Key = typename Policy::KeyType;
    __gm__ std::uint64_t* pairs = reinterpret_cast<__gm__ std::uint64_t*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    RouteDenseI32SingleKey(pairs, input, workspace, policy, static_cast<std::uint32_t>(keyNum),
                           static_cast<std::uint32_t>(GlobalThreadIndex()),
                           static_cast<std::uint32_t>(TotalThreadCount()));
}

template <class Policy, std::uint32_t FixedSlotsPerBlock, bool PreserveExisting>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void AddApplyPackedSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspaceStorage, std::uint64_t numBlocks, std::uint64_t hashSeed,
    std::uint32_t slotsPerBlock)
{
    __gm__ ulonglong4* blocks = reinterpret_cast<__gm__ ulonglong4*>(filter);
    __gm__ std::uint32_t* workspace = reinterpret_cast<__gm__ std::uint32_t*>(workspaceStorage);
    Policy policy(typename Policy::Hasher{hashSeed});

    ApplyPackedH8V1<Policy, FixedSlotsPerBlock, PreserveExisting>(blocks, workspace, policy, numBlocks, slotsPerBlock,
                                                                  static_cast<std::uint32_t>(GlobalThreadIndex()),
                                                                  static_cast<std::uint32_t>(TotalThreadCount()));
}

template <class Key, class Policy>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void ContainsSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* output, std::uint64_t numBlocks,
    std::uint64_t keyNum, std::uint64_t hashSeed)
{
    using Word = typename Policy::WordType;
    __gm__ Word* words = reinterpret_cast<__gm__ Word*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    Policy policy(typename Policy::Hasher{hashSeed});
    BloomFilterRef<Key, std::uint64_t, Policy> ref(words, numBlocks, policy);

    std::uint64_t const start = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    if constexpr (Policy::wordsPerBlock == 8 && Policy::patternBits == 8 && Policy::containsHorizontalLayout == 8 &&
                  Policy::containsVerticalLayout == 1 && !Policy::earlyExitContains) {
        constexpr std::uint64_t uint32Max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
        if (keyNum <= uint32Max && start <= uint32Max && stride <= uint32Max) {
            ContainsH8V1<std::uint32_t>(words, input, output, ref, policy, numBlocks,
                                        static_cast<std::uint32_t>(keyNum), static_cast<std::uint32_t>(start),
                                        static_cast<std::uint32_t>(stride));
        } else {
            ContainsH8V1<std::uint64_t>(words, input, output, ref, policy, numBlocks, keyNum, start, stride);
        }
    } else {
        for (std::uint64_t i = start; i < keyNum; i += stride) {
            Key const key = input[i];
            output[i] = static_cast<std::uint8_t>(ref.Contains(key) ? 1 : 0);
        }
    }
}

template <class Key, class Policy>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void ContainsIfSimt(
    __gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* stencil, __gm__ std::uint8_t* output,
    std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed)
{
    using Word = typename Policy::WordType;
    __gm__ Word* words = reinterpret_cast<__gm__ Word*>(filter);
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    Policy policy(typename Policy::Hasher{hashSeed});
    BloomFilterRef<Key, std::uint64_t, Policy> ref(words, numBlocks, policy);

    std::uint64_t i = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    while (i < keyNum) {
        if (stencil[i] != 0) {
            Key const key = input[i];
            output[i] = static_cast<std::uint8_t>(ref.Contains(key) ? 1 : 0);
        } else {
            output[i] = 0;
        }
        if (keyNum - i <= stride) {
            break;
        }
        i += stride;
    }
}

template <typename Word, bool IsMerge>
COLLECTION_SIMT_VF LAUNCH_BOUND(BLOOM_FILTER_THREAD_NUM) inline void CombineSimt(__gm__ std::uint8_t* destination,
                                                                                 __gm__ std::uint8_t* source,
                                                                                 std::uint64_t numWords)
{
    __gm__ Word* dst = reinterpret_cast<__gm__ Word*>(destination);
    __gm__ Word* src = reinterpret_cast<__gm__ Word*>(source);
    std::uint64_t const start = GlobalThreadIndex();
    std::uint64_t const stride = TotalThreadCount();
    for (std::uint64_t i = start; i < numWords; i += stride) {
        if constexpr (IsMerge) {
            dst[i] |= src[i];
        } else {
            dst[i] &= src[i];
        }
    }
}

} // namespace detail::bloom_filter

template <typename Word>
COLLECTION_AIV_GLOBAL void BloomFilterClear(__gm__ std::uint8_t* filter, std::uint64_t numWords)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::ClearSimt<Word>>(AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter,
                                                                  numWords);
}

template <class Key, class Policy>
COLLECTION_AIV_GLOBAL void BloomFilterAdd(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                          std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed,
                                          std::uint32_t packedAtomicEnabled)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddSimt<Key, Policy>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, keys, numBlocks, keyNum, hashSeed, packedAtomicEnabled);
}

template <class Key, class Policy>
COLLECTION_AIV_GLOBAL void BloomFilterAddIf(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                            __gm__ std::uint8_t* stencil, std::uint64_t numBlocks, std::uint64_t keyNum,
                                            std::uint64_t hashSeed)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddIfSimt<Key, Policy>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, keys, stencil, numBlocks, keyNum, hashSeed);
}

template <class Policy>
COLLECTION_AIV_GLOBAL void BloomFilterAddDenseI32Route(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                                       __gm__ std::uint8_t* workspace, std::uint64_t keyNum,
                                                       std::uint64_t hashSeed)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddDenseI32RouteSingleSimt<Policy>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, keys, workspace, keyNum, hashSeed);
}

template <class Key, class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateRoute(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                                      __gm__ std::uint8_t* workspace, std::uint64_t numBlocks,
                                                      std::uint64_t keyNum, std::uint64_t hashSeed,
                                                      std::uint32_t producerCount)
{
    constexpr std::uint32_t
        localCounterWords = detail::bloom_filter::PrivateRouteStorage<BucketBits, PackedCounters>::requiredWords;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    __ubuf__ std::uint32_t localCounters[localCounterWords];
#else
    // Keep this branch free of static __ubuf__ objects. With zero static UB,
    // get_imm(0) is the start of the dynamic UB reserved by the launcher and
    // avoids aggregating per-specialization static UB in older CANN linkers.
    __ubuf__ std::uint32_t* localCounters = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
#endif
    AscendC::Simt::VF_CALL<
        detail::bloom_filter::AddPrivateRouteSimt<Key, Policy, BucketBits, RecordsPerSegment, PackedCounters>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_ROUTE_THREAD_NUM}, filter, keys, workspace, localCounters, numBlocks, keyNum,
        hashSeed, producerCount);
}

template <class Policy, std::uint32_t BucketBits, std::uint32_t RecordsPerSegment, bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock, std::uint32_t FixedBlockShift, std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket, bool PreserveExisting>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateApply(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspace,
                                                      std::uint64_t hashSeed, std::uint32_t producerCount)
{
    constexpr std::uint32_t localWorkspaceWords = detail::bloom_filter::PrivateApplyStorage<
        FixedSlotsPerBlock, BlocksPerTile>::requiredWords;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    __ubuf__ std::uint32_t localWorkspace[localWorkspaceWords];
#else
    // This must remain a dynamic-only UB branch; see PrivateRouteStorage.
    __ubuf__ std::uint32_t* localWorkspace = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
#endif
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddPrivateApplySimt<
        Policy, BucketBits, RecordsPerSegment, PackedCounters, FixedSlotsPerBlock, FixedBlockShift, BlocksPerTile,
        TilesPerBucket, PreserveExisting>>(AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, workspace,
                                           localWorkspace, hashSeed, producerCount);
}

template <class Policy, bool PreserveExisting,
          std::uint32_t BucketBits = detail::bloom_filter::finePrivateRouteBucketBits,
          std::uint32_t RecordsPerSegment = detail::bloom_filter::finePrivateRouteRecordsPerSegment,
          std::uint32_t FixedBlockShift = 6,
          std::uint32_t BlocksPerBucket = detail::bloom_filter::compactPrivateSparseBlocksPerBucket,
          std::uint32_t FixedSlots = detail::bloom_filter::compactPrivateSparseSlotsPerBlock>
COLLECTION_AIV_GLOBAL void BloomFilterAddPrivateCompactSparseApply(__gm__ std::uint8_t* filter,
                                                                   __gm__ std::uint8_t* workspace,
                                                                   std::uint64_t hashSeed, std::uint32_t producerCount)
{
    constexpr std::uint32_t localWorkspaceWords = detail::bloom_filter::PrivateCompactSparseApplyStorage<
        BlocksPerBucket, FixedSlots>::requiredWords;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    __ubuf__ std::uint32_t localWorkspace[localWorkspaceWords];
#else
    // This must remain a dynamic-only UB branch; see PrivateRouteStorage.
    __ubuf__ std::uint32_t* localWorkspace = reinterpret_cast<__ubuf__ std::uint32_t*>(get_imm(0));
#endif
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddPrivateCompactSparseApplySimt<
        Policy, PreserveExisting, BucketBits, RecordsPerSegment, FixedBlockShift, BlocksPerBucket, FixedSlots>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, workspace, localWorkspace, hashSeed, producerCount);
}

template <class Key, class Policy, std::uint32_t FixedSlotsPerBlock, bool PowerOfTwoBlocks,
          std::uint32_t FixedBlockShift = 0>
COLLECTION_AIV_GLOBAL void BloomFilterAddRoute(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                               __gm__ std::uint8_t* workspace, std::uint64_t numBlocks,
                                               std::uint64_t keyNum, std::uint64_t hashSeed,
                                               std::uint32_t slotsPerBlock, std::uint32_t powerOfTwoShift)
{
    AscendC::Simt::VF_CALL<
        detail::bloom_filter::AddRoutePackedSimt<Key, Policy, FixedSlotsPerBlock, PowerOfTwoBlocks, FixedBlockShift>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_ROUTE_THREAD_NUM}, filter, keys, workspace, numBlocks, keyNum, hashSeed,
        slotsPerBlock, powerOfTwoShift);
}

template <class Policy, std::uint32_t FixedSlotsPerBlock, bool PreserveExisting>
COLLECTION_AIV_GLOBAL void BloomFilterAddApply(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* workspace,
                                               std::uint64_t numBlocks, std::uint64_t hashSeed,
                                               std::uint32_t slotsPerBlock)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::AddApplyPackedSimt<Policy, FixedSlotsPerBlock, PreserveExisting>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, workspace, numBlocks, hashSeed, slotsPerBlock);
}

template <class Key, class Policy>
COLLECTION_AIV_GLOBAL void BloomFilterContains(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                               __gm__ std::uint8_t* output, std::uint64_t numBlocks,
                                               std::uint64_t keyNum, std::uint64_t hashSeed)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::ContainsSimt<Key, Policy>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, keys, output, numBlocks, keyNum, hashSeed);
}

template <class Key, class Policy>
COLLECTION_AIV_GLOBAL void BloomFilterContainsIf(__gm__ std::uint8_t* filter, __gm__ std::uint8_t* keys,
                                                 __gm__ std::uint8_t* stencil, __gm__ std::uint8_t* output,
                                                 std::uint64_t numBlocks, std::uint64_t keyNum, std::uint64_t hashSeed)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::ContainsIfSimt<Key, Policy>>(
        AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM}, filter, keys, stencil, output, numBlocks, keyNum, hashSeed);
}

template <typename Word>
COLLECTION_AIV_GLOBAL void BloomFilterMerge(__gm__ std::uint8_t* destination, __gm__ std::uint8_t* source,
                                            std::uint64_t numWords)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::CombineSimt<Word, true>>(AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM},
                                                                          destination, source, numWords);
}

template <typename Word>
COLLECTION_AIV_GLOBAL void BloomFilterIntersect(__gm__ std::uint8_t* destination, __gm__ std::uint8_t* source,
                                                std::uint64_t numWords)
{
    AscendC::Simt::VF_CALL<detail::bloom_filter::CombineSimt<Word, false>>(AscendC::Simt::Dim3{BLOOM_FILTER_THREAD_NUM},
                                                                           destination, source, numWords);
}

} // namespace aclco
