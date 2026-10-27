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

namespace aclco {

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Add(void* keys,
                                                      ExtentType keyNum,
                                                      aclrtStream stream)
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (keys == nullptr) {
    throw std::invalid_argument("BloomFilter::Add: keys must not be null when keyNum is non-zero");
  }
  AddAsync(keys, keyNum, stream);
  Synchronize(stream, "BloomFilter::Add");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::AddIf(
  void* keys,
  void* stencil,
  ExtentType keyNum,
  aclrtStream stream)
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::AddIf: keys must not be null when keyNum is non-zero");
  }
  if (stencil == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::AddIf: stencil must not be null when keyNum is non-zero");
  }
  AddIfAsync(keys, stencil, keyNum, stream);
  Synchronize(stream, "BloomFilter::AddIf");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::AddIfAsync(
  void* keys,
  void* stencil,
  ExtentType keyNum,
  aclrtStream stream)
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (data_ == nullptr) {
    throw std::logic_error("BloomFilter::AddIfAsync: filter has no storage");
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::AddIfAsync: keys must not be null when keyNum is non-zero");
  }
  if (stencil == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::AddIfAsync: stencil must not be null when keyNum is non-zero");
  }
  bool const preserveExisting = !knownEmpty_;
  MarkDirty();
  if (TryPrivateAdd(keys, stencil, count, preserveExisting, stream)) {
    return;
  }
  std::uint32_t const cores =
    LaunchCoreNum(static_cast<std::uint64_t>(count));
  aclco::BloomFilterAddIf<KeyType, PolicyType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(keys),
    reinterpret_cast<std::uint8_t*>(stencil),
    static_cast<std::uint64_t>(numBlocks_),
    static_cast<std::uint64_t>(count),
    policy_.GetHasher().Seed());
}

}  // namespace aclco
