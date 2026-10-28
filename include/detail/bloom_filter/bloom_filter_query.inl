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
void BloomFilter<Key, Extent, Policy, Allocator>::Contains(
  void* keys,
  ExtentType keyNum,
  void* outputValues,
  aclrtStream stream) const
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::Contains: keys must not be null when keyNum is non-zero");
  }
  if (outputValues == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::Contains: outputValues must not be null when keyNum is non-zero");
  }
  ContainsAsync(keys, keyNum, outputValues, stream);
  Synchronize(stream, "BloomFilter::Contains");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Contains(
  void* keys,
  void* outputValues,
  ExtentType keyNum,
  aclrtStream stream) const
{
  Contains(keys, keyNum, outputValues, stream);
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ContainsAsync(
  void* keys,
  ExtentType keyNum,
  void* outputValues,
  aclrtStream stream) const
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (data_ == nullptr) {
    throw std::logic_error("BloomFilter::ContainsAsync: filter has no storage");
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::ContainsAsync: keys must not be null when keyNum is non-zero");
  }
  if (outputValues == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::ContainsAsync: outputValues must not be null when keyNum is non-zero");
  }
  if (knownEmpty_) {
    if (aclrtMemsetAsync(outputValues, static_cast<std::size_t>(count), 0,
                         static_cast<std::size_t>(count), stream) != ACL_SUCCESS) {
      throw std::runtime_error("BloomFilter::ContainsAsync: aclrtMemsetAsync failed");
    }
    return;
  }
  std::uint32_t const cores = LaunchCoreNum(static_cast<std::uint64_t>(count));
  aclco::BloomFilterContains<KeyType, PolicyType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(keys),
    reinterpret_cast<std::uint8_t*>(outputValues),
    static_cast<std::uint64_t>(numBlocks_),
    static_cast<std::uint64_t>(count),
    policy_.GetHasher().Seed());
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ContainsAsync(
  void* keys,
  void* outputValues,
  ExtentType keyNum,
  aclrtStream stream) const
{
  ContainsAsync(keys, keyNum, outputValues, stream);
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ContainsIf(
  void* keys,
  void* stencil,
  void* outputValues,
  ExtentType keyNum,
  aclrtStream stream) const
{
  if (static_cast<SizeType>(keyNum) == 0) {
    return;
  }
  ContainsIfAsync(keys, stencil, outputValues, keyNum, stream);
  Synchronize(stream, "BloomFilter::ContainsIf");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ContainsIfAsync(
  void* keys,
  void* stencil,
  void* outputValues,
  ExtentType keyNum,
  aclrtStream stream) const
{
  SizeType const count = static_cast<SizeType>(keyNum);
  if (count == 0) {
    return;
  }
  if (data_ == nullptr) {
    throw std::logic_error(
      "BloomFilter::ContainsIfAsync: filter has no storage");
  }
  if (keys == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::ContainsIfAsync: keys must not be null when keyNum is non-zero");
  }
  if (stencil == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::ContainsIfAsync: stencil must not be null when keyNum is non-zero");
  }
  if (outputValues == nullptr) {
    throw std::invalid_argument(
      "BloomFilter::ContainsIfAsync: outputValues must not be null when keyNum is non-zero");
  }
  if (knownEmpty_) {
    if (aclrtMemsetAsync(outputValues, static_cast<std::size_t>(count), 0,
                         static_cast<std::size_t>(count), stream) != ACL_SUCCESS) {
      throw std::runtime_error("BloomFilter::ContainsIfAsync: aclrtMemsetAsync failed");
    }
    return;
  }
  std::uint32_t const cores =
    LaunchCoreNum(static_cast<std::uint64_t>(count));
  aclco::BloomFilterContainsIf<KeyType, PolicyType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(keys),
    reinterpret_cast<std::uint8_t*>(stencil),
    reinterpret_cast<std::uint8_t*>(outputValues),
    static_cast<std::uint64_t>(numBlocks_),
    static_cast<std::uint64_t>(count),
    policy_.GetHasher().Seed());
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::Data() noexcept -> WordType*
{
  MarkDirty();
  mutableDataExposed_ = true;
  return data_;
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::Data() const noexcept -> WordType const*
{
  return data_;
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::BlockExtent() const noexcept -> ExtentType
{
  return ExtentType{numBlocks_};
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::NumWords() const noexcept -> SizeType
{
  return static_cast<SizeType>(numBlocks_ * wordsPerBlock);
}

template <class Key, class Extent, class Policy, class Allocator>
std::size_t BloomFilter<Key, Extent, Policy, Allocator>::SizeBytes() const noexcept
{
  return static_cast<std::size_t>(NumWords()) * sizeof(WordType);
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::GetPolicy() const noexcept
  -> PolicyType const&
{
  return policy_;
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ValidateCompatible(
  BloomFilter const& other,
  char const* operation) const
{
  if (data_ == nullptr || other.data_ == nullptr || numBlocks_ == 0 ||
      other.numBlocks_ == 0) {
    throw std::logic_error(std::string("BloomFilter::") + operation +
                           ": a moved-from filter cannot participate");
  }
  if (numBlocks_ != other.numBlocks_) {
    throw std::invalid_argument(std::string("BloomFilter::") + operation +
                                ": filters have different block extents");
  }
  if (policy_ != other.policy_) {
    throw std::invalid_argument(std::string("BloomFilter::") + operation +
                                ": filters have different policies");
  }
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Synchronize(aclrtStream stream,
                                                              char const* operation)
{
  std::uint32_t const result = aclrtSynchronizeStream(stream);
  if (result != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             ": aclrtSynchronizeStream failed, ret=" +
                             std::to_string(result));
  }
}

template <class Key, class Extent, class Policy, class Allocator>
std::uint32_t BloomFilter<Key, Extent, Policy, Allocator>::LaunchCoreNum(
  std::uint64_t workItems)
{
  std::uint32_t available = static_cast<std::uint32_t>(
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
  if (available == 0) {
    available = 1;
  }
  std::uint64_t const needed =
    (workItems + BLOOM_FILTER_THREAD_NUM - 1) / BLOOM_FILTER_THREAD_NUM;
  return static_cast<std::uint32_t>(
    std::max<std::uint64_t>(1, std::min<std::uint64_t>(available, needed)));
}

}  // namespace aclco
