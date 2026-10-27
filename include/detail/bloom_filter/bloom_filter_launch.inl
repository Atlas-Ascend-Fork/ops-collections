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

namespace detail::bloom_filter {

template <std::uint32_t BucketBits,
          std::uint32_t RecordsPerSegment,
          bool PackedCounters,
          class Key,
          class Policy>
inline void LaunchPrivateAddRoute(std::uint8_t* filter,
                                  std::uint8_t* keys,
                                  std::uint8_t* workspace,
                                  std::uint64_t numBlocks,
                                  std::uint64_t keyNum,
                                  std::uint64_t hashSeed,
                                  std::uint32_t producerCores,
                                  aclrtStream stream)
{
  using Storage =
    PrivateRouteStorage<BucketBits, PackedCounters>;
  aclco::BloomFilterAddPrivateRoute<
    Key,
    Policy,
    BucketBits,
    RecordsPerSegment,
    PackedCounters><<<producerCores, Storage::launchBytes, stream>>>(
    filter,
    keys,
    workspace,
    numBlocks,
    keyNum,
    hashSeed,
    producerCores);
}

template <std::uint32_t BucketBits,
          std::uint32_t RecordsPerSegment,
          bool PackedCounters,
          std::uint32_t FixedSlotsPerBlock,
          std::uint32_t FixedBlockShift,
          std::uint32_t BlocksPerTile,
          std::uint32_t TilesPerBucket,
          class Policy>
inline void LaunchPrivateAddApply(std::uint8_t* filter,
                                  std::uint8_t* workspace,
                                  std::uint64_t hashSeed,
                                  std::uint32_t producerCores,
                                  bool preserveExisting,
                                  std::uint32_t applyCores,
                                  aclrtStream stream,
                                  std::uint8_t* selectedFlag = nullptr)
{
  using Storage =
    PrivateApplyStorage<FixedSlotsPerBlock, BlocksPerTile>;
  if (preserveExisting) {
    BloomFilterAddPrivateApplySkip<
      Policy,
      BucketBits,
      RecordsPerSegment,
      PackedCounters,
      FixedSlotsPerBlock,
      FixedBlockShift,
      BlocksPerTile,
      TilesPerBucket,
      true><<<applyCores, Storage::launchBytes, stream>>>(
      filter, workspace, selectedFlag, hashSeed, producerCores);
  } else {
    BloomFilterAddPrivateApplySkip<
      Policy,
      BucketBits,
      RecordsPerSegment,
      PackedCounters,
      FixedSlotsPerBlock,
      FixedBlockShift,
      BlocksPerTile,
      TilesPerBucket,
      false><<<applyCores, Storage::launchBytes, stream>>>(
      filter, workspace, selectedFlag, hashSeed, producerCores);
  }
}

template <class Policy>
inline void LaunchPrivateCompactSparseAddApply(
  std::uint8_t* filter,
  std::uint8_t* workspace,
  std::uint64_t hashSeed,
  std::uint32_t producerCores,
  bool preserveExisting,
  std::uint32_t applyCores,
  aclrtStream stream,
  std::uint8_t* selectedFlag = nullptr)
{
  using Storage = PrivateCompactSparseApplyStorage<
    compactPrivateSparseBlocksPerBucket,
    compactPrivateSparseSlotsPerBlock>;
  if (preserveExisting) {
    BloomFilterAddPrivateCompactSparseApplySkip<
      Policy,
      true><<<applyCores, Storage::launchBytes, stream>>>(
      filter, workspace, selectedFlag, hashSeed, producerCores);
  } else {
    BloomFilterAddPrivateCompactSparseApplySkip<
      Policy,
      false><<<applyCores, Storage::launchBytes, stream>>>(
      filter, workspace, selectedFlag, hashSeed, producerCores);
  }
}

struct RoutedAddArgs {
  std::uint8_t* filter;
  std::uint8_t* keys;
  std::uint8_t* workspace;
  std::uint64_t numBlocks;
  std::uint64_t keyNum;
  std::uint64_t hashSeed;
  std::uint32_t slotsPerBlock;
  std::uint32_t powerOfTwoShift;
  std::uint32_t cores;
  aclrtStream stream;
};

template <std::uint32_t FixedSlotsPerBlock,
          bool PowerOfTwoBlocks,
          std::uint32_t FixedBlockShift,
          class Key,
          class Policy>
inline void LaunchStandardRoutedAddRoute(RoutedAddArgs const& args)
{
  aclco::BloomFilterAddRoute<
    Key,
    Policy,
    FixedSlotsPerBlock,
    PowerOfTwoBlocks,
    FixedBlockShift><<<args.cores, 0, args.stream>>>(
    args.filter,
    args.keys,
    args.workspace,
    args.numBlocks,
    args.keyNum,
    args.hashSeed,
    args.slotsPerBlock,
    args.powerOfTwoShift);
}

template <std::uint32_t FixedSlotsPerBlock,
          std::uint32_t FixedBlockShift,
          class Key,
          class Policy>
inline void LaunchRoutedAddRoute(RoutedAddArgs const& args)
{
  if constexpr (
    (std::is_same_v<Key, std::int32_t> ||
     std::is_same_v<Key, std::uint32_t>) &&
    FixedSlotsPerBlock == denseRoutedSlotsPerBlock &&
    FixedBlockShift == 12) {
    static_assert(
      aclco::detail::UsesMultiplyHighBlockIndex<Policy>::value,
      "the dense I32 BloomFilter route requires multiply-high indexing");
    (void)args.numBlocks;
    (void)args.slotsPerBlock;
    (void)args.powerOfTwoShift;
    aclco::BloomFilterAddDenseI32Route<Policy><<<args.cores, 0, args.stream>>>(
      args.filter, args.keys, args.workspace, args.keyNum, args.hashSeed);
  } else if constexpr (FixedBlockShift != 0) {
    static_assert(
      aclco::detail::UsesMultiplyHighBlockIndex<Policy>::value,
      "a fixed BloomFilter block shift requires multiply-high indexing");
    LaunchStandardRoutedAddRoute<
      FixedSlotsPerBlock, true, FixedBlockShift, Key, Policy>(args);
  } else {
    if constexpr (aclco::detail::UsesMultiplyHighBlockIndex<Policy>::value) {
      if (args.powerOfTwoShift != 0) {
        LaunchStandardRoutedAddRoute<
          FixedSlotsPerBlock, true, 0, Key, Policy>(args);
        return;
      }
    }
    LaunchStandardRoutedAddRoute<
      FixedSlotsPerBlock, false, 0, Key, Policy>(args);
  }
}

template <std::uint32_t FixedSlotsPerBlock,
          std::uint32_t FixedBlockShift,
          class Key,
          class Policy>
inline void LaunchRoutedAddRoute(std::uint8_t* filter,
                                 std::uint8_t* keys,
                                 std::uint8_t* workspace,
                                 std::uint64_t numBlocks,
                                 std::uint64_t keyNum,
                                 std::uint64_t hashSeed,
                                 std::uint32_t slotsPerBlock,
                                 std::uint32_t powerOfTwoShift,
                                 std::uint32_t cores,
                                 aclrtStream stream)
{
  LaunchRoutedAddRoute<FixedSlotsPerBlock, FixedBlockShift, Key, Policy>(
    RoutedAddArgs{filter, keys, workspace, numBlocks, keyNum, hashSeed,
                  slotsPerBlock, powerOfTwoShift, cores, stream});
}

template <std::uint32_t FixedSlotsPerBlock, class Policy>
inline void LaunchRoutedAddApply(std::uint8_t* filter,
                                 std::uint8_t* workspace,
                                 std::uint64_t numBlocks,
                                 std::uint64_t hashSeed,
                                 std::uint32_t slotsPerBlock,
                                 bool preserveExisting,
                                 std::uint32_t cores,
                                 aclrtStream stream)
{
  if (preserveExisting) {
    aclco::BloomFilterAddApply<
      Policy, FixedSlotsPerBlock, true><<<cores, 0, stream>>>(
      filter,
      workspace,
      numBlocks,
      hashSeed,
      slotsPerBlock);
  } else {
    aclco::BloomFilterAddApply<
      Policy, FixedSlotsPerBlock, false><<<cores, 0, stream>>>(
      filter,
      workspace,
      numBlocks,
      hashSeed,
      slotsPerBlock);
  }
}

}  // namespace detail::bloom_filter
