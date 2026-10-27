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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "detail/bloom_filter/kernels.h"
#include "tiling/platform/platform_ascendc.h"

#include "detail/bloom_filter/private_route_device.inl"

namespace aclco {

#include "detail/bloom_filter/bloom_filter_launch.inl"

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::AddAsync(void* keys,
                                                           ExtentType keyNum,
                                                           aclrtStream stream)
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (data_ == nullptr) {
    throw std::logic_error("BloomFilter::AddAsync: filter has no storage");
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::AddAsync: keys must not be null when keyNum is non-zero");
  }
  bool const preserveExisting = !knownEmpty_;
  MarkDirty();
  std::uint32_t const cores = LaunchCoreNum(static_cast<std::uint64_t>(count));
  std::uint32_t const packedAtomicEnabled =
    static_cast<std::uint32_t>(
      reinterpret_cast<std::uintptr_t>(data_) %
        alignof(std::uint64_t) ==
      0);
  std::uint32_t const fullBlockAccessEnabled =
    static_cast<std::uint32_t>(
      reinterpret_cast<std::uintptr_t>(data_) %
        (PolicyType::wordsPerBlock * sizeof(WordType)) ==
      0);
  if constexpr (PolicyType::wordsPerBlock == 8 &&
                sizeof(WordType) == sizeof(std::uint32_t) &&
                PolicyType::patternBits == 8 &&
                PolicyType::addHorizontalLayout == 8 &&
                PolicyType::addVerticalLayout == 1 &&
                !PolicyType::conditionalAdd) {
    if (fullBlockAccessEnabled != 0 &&
        EnsureAddWorkspace(static_cast<std::uint64_t>(count), stream)) {
      auto* workspace = reinterpret_cast<std::uint32_t*>(addWorkspace_);
      std::uint32_t const blockCount =
        static_cast<std::uint32_t>(numBlocks_);
      std::uint32_t blockIndexShift = 0;
      if constexpr (
        detail::UsesMultiplyHighBlockIndex<PolicyType>::value) {
        if (blockCount != 0 &&
            (blockCount & (blockCount - 1)) == 0) {
          blockIndexShift = 32;
          for (std::uint32_t value = blockCount; value > 1; value >>= 1) {
            --blockIndexShift;
          }
        }
      }
      std::uint32_t const applyCores =
        LaunchCoreNum(static_cast<std::uint64_t>(numBlocks_));
      // Route hashes into a compact counter prefix and bounded block-major
      // slot records. The three common performance layouts use fixed template
      // capacities so address arithmetic and sparse Apply control flow are
      // compile-time constants.
      auto* filterStorage = reinterpret_cast<std::uint8_t*>(data_);
      auto* keyStorage = reinterpret_cast<std::uint8_t*>(keys);
      auto* workspaceStorage =
        reinterpret_cast<std::uint8_t*>(workspace);
      if (addProducerCores_ != 0) {
        std::uint64_t const hashSeed = policy_.GetHasher().Seed();
        if (blockCount == (1U << 20)) {
          detail::bloom_filter::LaunchPrivateAddRoute<
            detail::bloom_filter::densePrivateRouteBucketBits,
            detail::bloom_filter::densePrivateRouteRecordsPerSegment,
            false,
            KeyType,
            PolicyType>(
            filterStorage,
            keyStorage,
            workspaceStorage,
            static_cast<std::uint64_t>(numBlocks_),
            static_cast<std::uint64_t>(count),
            hashSeed,
            addProducerCores_,
            stream);
          detail::bloom_filter::LaunchPrivateAddApply<
            detail::bloom_filter::densePrivateRouteBucketBits,
            detail::bloom_filter::densePrivateRouteRecordsPerSegment,
            false,
            detail::bloom_filter::denseRoutedSlotsPerBlock,
            12,
            128,
            1,
            PolicyType>(
            filterStorage,
            workspaceStorage,
            hashSeed,
            addProducerCores_,
            preserveExisting,
            addProducerCores_,
            stream);
          return;
        }

        detail::bloom_filter::LaunchPrivateAddRoute<
          detail::bloom_filter::finePrivateRouteBucketBits,
          detail::bloom_filter::finePrivateRouteRecordsPerSegment,
          true,
          KeyType,
          PolicyType>(
          filterStorage,
          keyStorage,
          workspaceStorage,
          static_cast<std::uint64_t>(numBlocks_),
          static_cast<std::uint64_t>(count),
          hashSeed,
          addProducerCores_,
          stream);
        if (blockCount == (1U << 23)) {
          detail::bloom_filter::LaunchPrivateAddApply<
            detail::bloom_filter::finePrivateRouteBucketBits,
            detail::bloom_filter::finePrivateRouteRecordsPerSegment,
            true,
            detail::bloom_filter::mediumRoutedSlotsPerBlock,
            9,
            256,
            1,
            PolicyType>(
            filterStorage,
            workspaceStorage,
            hashSeed,
            addProducerCores_,
            preserveExisting,
            addProducerCores_,
            stream);
        } else {
          if constexpr (std::is_same_v<KeyType, std::int32_t> ||
                        std::is_same_v<KeyType, std::uint32_t>) {
            detail::bloom_filter::LaunchPrivateCompactSparseAddApply<
              PolicyType>(
              filterStorage,
              workspaceStorage,
              hashSeed,
              addProducerCores_,
              preserveExisting,
              addProducerCores_,
              stream);
          } else {
            detail::bloom_filter::LaunchPrivateAddApply<
              detail::bloom_filter::finePrivateRouteBucketBits,
              detail::bloom_filter::finePrivateRouteRecordsPerSegment,
              true,
              detail::bloom_filter::privateSparseSlotsPerBlock,
              6,
              1024,
              2,
              PolicyType>(
              filterStorage,
              workspaceStorage,
              hashSeed,
              addProducerCores_,
              preserveExisting,
              addProducerCores_,
              stream);
          }
        }
        return;
      }
      auto launchRoutedAdd = [&](auto fixedSlots, auto fixedBlockShift) {
        constexpr std::uint32_t slots = decltype(fixedSlots)::value;
        constexpr std::uint32_t shift = decltype(fixedBlockShift)::value;
        detail::bloom_filter::LaunchRoutedAddRoute<
          slots, shift, KeyType, PolicyType>(
          filterStorage,
          keyStorage,
          workspaceStorage,
          static_cast<std::uint64_t>(numBlocks_),
          static_cast<std::uint64_t>(count),
          policy_.GetHasher().Seed(),
          addSlotsPerBlock_,
          blockIndexShift,
          cores,
          stream);
        detail::bloom_filter::LaunchRoutedAddApply<
          slots, PolicyType>(
          filterStorage,
          workspaceStorage,
          static_cast<std::uint64_t>(numBlocks_),
          policy_.GetHasher().Seed(),
          addSlotsPerBlock_,
          preserveExisting,
          applyCores,
          stream);
      };
      auto launchKnownLayout = [&](auto fixedSlots,
                                   std::uint32_t expectedBlocks,
                                   auto fixedBlockShift) {
        if (blockCount == expectedBlocks) {
          launchRoutedAdd(fixedSlots, fixedBlockShift);
        } else {
          launchRoutedAdd(
            fixedSlots, std::integral_constant<std::uint32_t, 0>{});
        }
      };
      constexpr bool supportsKnownRoutedLayouts =
        detail::UsesMultiplyHighBlockIndex<PolicyType>::value &&
        (std::is_same_v<KeyType, std::int32_t> ||
         std::is_same_v<KeyType, std::uint32_t> ||
         std::is_same_v<KeyType, std::int64_t> ||
         std::is_same_v<KeyType, std::uint64_t>);

      if constexpr (supportsKnownRoutedLayouts) {
        if (addSlotsPerBlock_ ==
            detail::bloom_filter::denseRoutedSlotsPerBlock) {
          launchKnownLayout(
            std::integral_constant<
              std::uint32_t,
              detail::bloom_filter::denseRoutedSlotsPerBlock>{},
            1U << 20,
            std::integral_constant<std::uint32_t, 12>{});
          return;
        }

        if (addSlotsPerBlock_ ==
            detail::bloom_filter::mediumRoutedSlotsPerBlock) {
          launchKnownLayout(
            std::integral_constant<
              std::uint32_t,
              detail::bloom_filter::mediumRoutedSlotsPerBlock>{},
            1U << 23,
            std::integral_constant<std::uint32_t, 9>{});
          return;
        }

        if (addSlotsPerBlock_ ==
            detail::bloom_filter::sparseRoutedSlotsPerBlock) {
          launchKnownLayout(
            std::integral_constant<
              std::uint32_t,
              detail::bloom_filter::sparseRoutedSlotsPerBlock>{},
            1U << 26,
            std::integral_constant<std::uint32_t, 6>{});
          return;
        }
      }

      launchRoutedAdd(
        std::integral_constant<std::uint32_t, 0>{},
        std::integral_constant<std::uint32_t, 0>{});
      return;
    }
  }
  aclco::BloomFilterAdd<KeyType, PolicyType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(keys),
    static_cast<std::uint64_t>(numBlocks_),
    static_cast<std::uint64_t>(count),
    policy_.GetHasher().Seed(),
    packedAtomicEnabled);
}

template <class Key, class Extent, class Policy, class Allocator>
bool BloomFilter<Key, Extent, Policy, Allocator>::EnsureAddWorkspace(
  std::uint64_t keyNum,
  aclrtStream stream) noexcept
{
  // Small or very sparse batches are faster and substantially leaner on the
  // direct packed-atomic path. The aggregation path is reserved for batches
  // that can amortize its route/apply launches and per-block counters.
  constexpr std::uint64_t minimumAggregatedKeys = 16384;
  constexpr std::uint32_t maximumSlotsPerBlock =
    detail::bloom_filter::denseRoutedSlotsPerBlock;
  constexpr std::size_t deviceMemoryReserve = 256ULL * 1024ULL * 1024ULL;

  std::uint64_t const blocks = static_cast<std::uint64_t>(numBlocks_);
  // Apply visits every block once, so route only when the batch contains at
  // least one key per two blocks. Sparser batches stay on the direct atomic
  // path instead of paying a disproportionately large counter scan.
  std::uint64_t const densityThreshold = (blocks + 1) / 2;
  if (keyNum < std::max(minimumAggregatedKeys, densityThreshold) ||
      keyNum > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  if (addWorkspace_ != nullptr) {
    return true;
  }
  if (addWorkspaceUnavailable_ || blocks == 0 ||
      blocks > detail::bloom_filter::maximumRoutedCounterWords) {
    return false;
  }

  if constexpr (
    detail::UsesMultiplyHighBlockIndex<PolicyType>::value &&
    (std::is_same_v<KeyType, std::int32_t> ||
     std::is_same_v<KeyType, std::uint32_t> ||
     std::is_same_v<KeyType, std::int64_t> ||
     std::is_same_v<KeyType, std::uint64_t>) &&
    PolicyType::wordsPerBlock == 8 &&
    sizeof(WordType) == sizeof(std::uint32_t) &&
    PolicyType::patternBits == 8 &&
    PolicyType::addHorizontalLayout == 8 &&
    PolicyType::addVerticalLayout == 1 &&
    !PolicyType::conditionalAdd) {
    bool const privateLayout =
      blocks == (1ULL << 20) ||
      blocks == (1ULL << 23) ||
      blocks == (1ULL << 26);
    std::uint32_t const producerCores = LaunchCoreNum(keyNum);
    if (privateLayout &&
        keyNum >= detail::bloom_filter::minimumPrivateRoutedKeys &&
        producerCores <=
          detail::bloom_filter::maximumPrivateRouteProducers) {
      bool const denseLayout = blocks == (1ULL << 20);
      std::uint64_t const bucketCount =
        denseLayout
          ? detail::bloom_filter::densePrivateRouteBucketCount
          : detail::bloom_filter::finePrivateRouteBucketCount;
      std::uint64_t const recordsPerSegment =
        denseLayout
          ? detail::bloom_filter::densePrivateRouteRecordsPerSegment
          : detail::bloom_filter::finePrivateRouteRecordsPerSegment;
      std::uint64_t const countWords =
        static_cast<std::uint64_t>(
          detail::bloom_filter::privateRoutePackedCounterWords) *
        producerCores;
      std::uint64_t const recordWords =
        bucketCount * producerCores * recordsPerSegment * 2ULL;
      std::uint64_t const privateWorkspaceWords =
        countWords + recordWords;

      std::size_t freeBytes = 0;
      std::size_t totalBytes = 0;
      bool const memoryKnown =
        aclrtGetMemInfo(
          ACL_HBM_MEM, &freeBytes, &totalBytes) == ACL_SUCCESS;
      (void)totalBytes;
      std::uint64_t const privateWorkspaceBytes =
        privateWorkspaceWords * sizeof(std::uint32_t);
      bool const memoryAvailable =
        !memoryKnown ||
        (freeBytes > deviceMemoryReserve &&
         privateWorkspaceBytes <= freeBytes - deviceMemoryReserve);
      if (privateWorkspaceWords <=
            std::numeric_limits<std::size_t>::max() &&
          privateWorkspaceBytes / sizeof(std::uint32_t) ==
            privateWorkspaceWords &&
          memoryAvailable) {
        WordType* privateWorkspace = nullptr;
        try {
          privateWorkspace =
            allocator_.Allocate(
              static_cast<std::size_t>(privateWorkspaceWords));
        } catch (...) {
          privateWorkspace = nullptr;
        }
        if (privateWorkspace != nullptr &&
            reinterpret_cast<std::uintptr_t>(privateWorkspace) %
                alignof(std::uint64_t) ==
              0) {
          addWorkspace_ = privateWorkspace;
          addSlotsPerBlock_ = 0;
          addProducerCores_ = producerCores;
          return true;
        }
        if (privateWorkspace != nullptr) {
          // U64 records follow an even U32 prefix. A misaligned allocator
          // cannot use this route, but the established U32 workspace remains
          // a valid exact fallback.
          try {
            allocator_.Deallocate(privateWorkspace);
          } catch (...) {
            addWorkspaceUnavailable_ = true;
            return false;
          }
        }
      }
    }
  }

  std::uint64_t desiredSlots =
    std::max<std::uint64_t>(1, (keyNum * 2 + blocks - 1) / blocks);
  // At one-to-two keys per block, two extra slots cut the large-filter
  // overflow tail sharply while Apply still reads only populated slots.
  if (keyNum >= blocks && keyNum < blocks * 2) {
    desiredSlots += 2;
  }
  std::uint64_t const budgetSlots =
    detail::bloom_filter::maximumRoutedSlotWords / blocks;
  std::uint64_t slotsPerBlock =
    std::min<std::uint64_t>(
      maximumSlotsPerBlock, std::min(desiredSlots, budgetSlots));
  if (slotsPerBlock == 0) {
    addWorkspaceUnavailable_ = true;
    return false;
  }

  // Respect current device pressure. This query runs only before the lazy
  // allocation (the performance warm-up absorbs it) and leaves a reserve for
  // caller-owned buffers and runtime bookkeeping.
  std::size_t freeBytes = 0;
  std::size_t totalBytes = 0;
  if (aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes) == ACL_SUCCESS) {
    (void)totalBytes;
    std::uint64_t const availableWords =
      freeBytes > deviceMemoryReserve
        ? static_cast<std::uint64_t>(
            (freeBytes - deviceMemoryReserve) / sizeof(std::uint32_t))
        : 0;
    std::uint64_t const affordableSlots =
      availableWords / blocks > 1
        ? availableWords / blocks - 1
        : 0;
    slotsPerBlock = std::min(slotsPerBlock, affordableSlots);
  }
  if (slotsPerBlock == 0) {
    addWorkspaceUnavailable_ = true;
    return false;
  }

  std::uint64_t const workspaceWords64 =
    blocks * (slotsPerBlock + 1);
  if (workspaceWords64 >
        detail::bloom_filter::maximumRoutedWorkspaceWords ||
      workspaceWords64 > std::numeric_limits<std::size_t>::max()) {
    addWorkspaceUnavailable_ = true;
    return false;
  }

  WordType* workspace = nullptr;
  try {
    workspace =
      allocator_.Allocate(static_cast<std::size_t>(workspaceWords64));
  } catch (...) {
    addWorkspaceUnavailable_ = true;
    return false;
  }
  if (workspace == nullptr) {
    addWorkspaceUnavailable_ = true;
    return false;
  }

  addWorkspace_ = workspace;
  addSlotsPerBlock_ = static_cast<std::uint32_t>(slotsPerBlock);
  addProducerCores_ = 0;
  // The workspace starts with a compact counter prefix followed by
  // block-major slots. Initialize every word once; later Apply launches reset
  // only each populated counter.
  std::uint32_t const cores = LaunchCoreNum(workspaceWords64);
  aclco::BloomFilterClear<std::uint32_t><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(addWorkspace_), workspaceWords64);
  return true;
}

}  // namespace aclco

#include "detail/bloom_filter/bloom_filter_lifecycle.inl"
#include "detail/bloom_filter/bloom_filter_add.inl"
#include "detail/bloom_filter/bloom_filter_query.inl"
#include "detail/bloom_filter/bloom_filter_merge.inl"
#include "detail/bloom_filter/private_add_route.inl"
