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
#include "detail/bloom_filter/private_route_device.inl"

namespace aclco::detail::bloom_filter {

template <std::uint32_t BucketBits,
          std::uint32_t RecordsPerSegment,
          bool PackedCounters,
          class Key,
          class Policy>
inline void LaunchPrivateAddRouteIf(std::uint8_t* filter,
                                    std::uint8_t* keys,
                                    std::uint8_t* stencil,
                                    std::uint8_t* workspace,
                                    std::uint8_t* selectedFlag,
                                    std::uint64_t numBlocks,
                                    std::uint64_t keyNum,
                                    std::uint64_t hashSeed,
                                    std::uint32_t producerCores,
                                    aclrtStream stream)
{
  using Storage = PrivateRouteStorage<BucketBits, PackedCounters>;
  BloomFilterAddPrivateRouteIf<
    Key,
    Policy,
    BucketBits,
    RecordsPerSegment,
    PackedCounters><<<producerCores, Storage::launchBytes, stream>>>(
    filter, keys, stencil, workspace, selectedFlag, numBlocks, keyNum, hashSeed, producerCores);
}

template <class Key, class Policy>
inline void LaunchPrivateRouteForLayout(std::uint8_t* filter, std::uint8_t* keys, std::uint8_t* stencil,
                                        std::uint8_t* workspace, std::uint8_t* selectedFlag, std::uint64_t numBlocks,
                                        std::uint64_t count, std::uint64_t hashSeed, std::uint32_t producerCores,
                                        std::uint32_t blockCount, aclrtStream stream)
{
  if (blockCount == (1U << 20)) {
    if (stencil != nullptr) {
      LaunchPrivateAddRouteIf<densePrivateRouteBucketBits, densePrivateRouteRecordsPerSegment, false, Key, Policy>(
        filter, keys, stencil, workspace, selectedFlag, numBlocks, count, hashSeed, producerCores, stream);
    } else {
      LaunchPrivateAddRoute<densePrivateRouteBucketBits, densePrivateRouteRecordsPerSegment, false, Key, Policy>(
        filter, keys, workspace, numBlocks, count, hashSeed, producerCores, stream);
    }
    return;
  }
  if (stencil != nullptr) {
    LaunchPrivateAddRouteIf<finePrivateRouteBucketBits, finePrivateRouteRecordsPerSegment, true, Key, Policy>(
      filter, keys, stencil, workspace, selectedFlag, numBlocks, count, hashSeed, producerCores, stream);
  } else {
    LaunchPrivateAddRoute<finePrivateRouteBucketBits, finePrivateRouteRecordsPerSegment, true, Key, Policy>(
      filter, keys, workspace, numBlocks, count, hashSeed, producerCores, stream);
  }
}

template <class Key, class Policy>
inline void LaunchPrivateApplyForLayout(std::uint8_t* filter, std::uint8_t* workspace, std::uint8_t* selectedFlag,
                                        std::uint64_t hashSeed, std::uint32_t producerCores, bool preserveExisting,
                                        std::uint32_t applyCores, std::uint32_t blockCount, aclrtStream stream)
{
  if (blockCount == (1U << 20)) {
    LaunchPrivateAddApply<densePrivateRouteBucketBits, densePrivateRouteRecordsPerSegment, false,
                          denseRoutedSlotsPerBlock, 12, 128, 1, Policy>(
      filter, workspace, hashSeed, producerCores, preserveExisting, applyCores, stream, selectedFlag);
  } else if (blockCount == (1U << 23)) {
    LaunchPrivateAddApply<finePrivateRouteBucketBits, finePrivateRouteRecordsPerSegment, true,
                          mediumRoutedSlotsPerBlock, 9, 256, 1, Policy>(
      filter, workspace, hashSeed, producerCores, preserveExisting, applyCores, stream, selectedFlag);
  } else if constexpr (std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::uint32_t>) {
    LaunchPrivateCompactSparseAddApply<Policy>(filter, workspace, hashSeed, producerCores, preserveExisting, applyCores,
                                               stream, selectedFlag);
  } else {
    LaunchPrivateAddApply<finePrivateRouteBucketBits, finePrivateRouteRecordsPerSegment, true,
                          privateSparseSlotsPerBlock, 6, 1024, 2, Policy>(
      filter, workspace, hashSeed, producerCores, preserveExisting, applyCores, stream, selectedFlag);
  }
}

}  // namespace aclco::detail::bloom_filter

namespace aclco {

template <class Key, class Extent, class Policy, class Allocator>
bool BloomFilter<Key, Extent, Policy, Allocator>::EnsureSelectedFlag(aclrtStream stream)
{
  if (addSelectedFlag_ == nullptr) {
    try {
      addSelectedFlag_ = allocator_.Allocate(1);
    } catch (...) {
      addSelectedFlag_ = nullptr;
    }
  }
  if (addSelectedFlag_ == nullptr) {
    return false;
  }
  return aclrtMemsetAsync(addSelectedFlag_, sizeof(std::uint32_t), 0, sizeof(std::uint32_t), stream) == ACL_SUCCESS;
}

template <class Key, class Extent, class Policy, class Allocator>
bool BloomFilter<Key, Extent, Policy, Allocator>::TryPrivateAdd(void* keys, void* stencil, SizeType count,
                                                                bool preserveExisting, aclrtStream stream)
{
  if constexpr (PolicyType::wordsPerBlock == 8 &&
                sizeof(WordType) == sizeof(std::uint32_t) &&
                PolicyType::patternBits == 8 &&
                PolicyType::addHorizontalLayout == 8 &&
                PolicyType::addVerticalLayout == 1 &&
                !PolicyType::conditionalAdd) {
    std::uint32_t const blockCount = static_cast<std::uint32_t>(numBlocks_);
    bool const privateLayout =
      blockCount == (1U << 20) || blockCount == (1U << 23) || blockCount == (1U << 26);
    bool const aligned =
      reinterpret_cast<std::uintptr_t>(data_) %
        (PolicyType::wordsPerBlock * sizeof(WordType)) ==
      0;
    if (!privateLayout || !aligned || !EnsureAddWorkspace(static_cast<std::uint64_t>(count), stream) ||
        addProducerCores_ == 0) {
      return false;
    }
    if (!EnsureSelectedFlag(stream)) {
      return false;
    }
    auto* filter = reinterpret_cast<std::uint8_t*>(data_);
    auto* keyBytes = reinterpret_cast<std::uint8_t*>(keys);
    auto* stencilBytes = reinterpret_cast<std::uint8_t*>(stencil);
    auto* workspace = reinterpret_cast<std::uint8_t*>(addWorkspace_);
    auto* selectedFlag = reinterpret_cast<std::uint8_t*>(addSelectedFlag_);
    std::uint64_t const hashSeed = policy_.GetHasher().Seed();
    std::uint32_t const applyCores = LaunchCoreNum(static_cast<std::uint64_t>(numBlocks_));
    detail::bloom_filter::LaunchPrivateRouteForLayout<KeyType, PolicyType>(
      filter, keyBytes, stencilBytes, workspace, selectedFlag, static_cast<std::uint64_t>(numBlocks_),
      static_cast<std::uint64_t>(count), hashSeed, addProducerCores_, blockCount, stream);
    detail::bloom_filter::LaunchPrivateApplyForLayout<KeyType, PolicyType>(
      filter, workspace, selectedFlag, hashSeed, addProducerCores_, preserveExisting, applyCores, blockCount, stream);
    return true;
  } else {
    return false;
  }
}

}  // namespace aclco
