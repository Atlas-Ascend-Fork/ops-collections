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
BloomFilter<Key, Extent, Policy, Allocator>::BloomFilter(
  ExtentType numBlocks,
  PolicyType const& policy,
  AllocatorType const& allocator,
  aclrtStream stream)
    : allocator_{allocator},
      numBlocks_{static_cast<SizeType>(numBlocks)},
      policy_{policy}
{
  if (numBlocks_ == 0) {
    throw std::invalid_argument("BloomFilter: numBlocks must be greater than zero");
  }
  if (static_cast<std::uint64_t>(numBlocks_) > PolicyType::maxFilterBlocks) {
    throw std::length_error("BloomFilter: numBlocks exceeds the supported block-index range");
  }
  if (static_cast<std::uint64_t>(numBlocks_) >
      std::numeric_limits<std::size_t>::max() / wordsPerBlock) {
    throw std::length_error("BloomFilter: word count overflows size_t");
  }
  if (static_cast<std::uint64_t>(numBlocks_) >
      static_cast<std::uint64_t>(std::numeric_limits<SizeType>::max()) / wordsPerBlock) {
    throw std::length_error("BloomFilter: word count overflows Extent::ValueType");
  }
  if (static_cast<std::uint64_t>(numBlocks_) >
      std::numeric_limits<std::size_t>::max() / wordsPerBlock / sizeof(WordType)) {
    throw std::length_error("BloomFilter: allocation size overflows size_t");
  }

  data_ = allocator_.Allocate(static_cast<std::size_t>(NumWords()));
  if (data_ == nullptr) {
    throw std::bad_alloc();
  }

  try {
    Clear(stream);
  } catch (...) {
    Release();
    throw;
  }
}

template <class Key, class Extent, class Policy, class Allocator>
BloomFilter<Key, Extent, Policy, Allocator>::BloomFilter(
  ExtentType numBlocks,
  PolicyType const& policy,
  aclrtStream stream)
    : BloomFilter(numBlocks, policy, AllocatorType{}, stream)
{
}

template <class Key, class Extent, class Policy, class Allocator>
BloomFilter<Key, Extent, Policy, Allocator>::~BloomFilter()
{
  Release();
}

template <class Key, class Extent, class Policy, class Allocator>
BloomFilter<Key, Extent, Policy, Allocator>::BloomFilter(BloomFilter&& other) noexcept(
  std::is_nothrow_move_constructible_v<AllocatorType> &&
  std::is_nothrow_move_constructible_v<PolicyType>)
    : allocator_{std::move(other.allocator_)},
      data_{std::exchange(other.data_, nullptr)},
      addWorkspace_{std::exchange(other.addWorkspace_, nullptr)},
      addSelectedFlag_{std::exchange(other.addSelectedFlag_, nullptr)},
      numBlocks_{std::exchange(other.numBlocks_, 0)},
      policy_{std::move(other.policy_)},
      addSlotsPerBlock_{std::exchange(other.addSlotsPerBlock_, 0)},
      addProducerCores_{std::exchange(other.addProducerCores_, 0)},
      addWorkspaceUnavailable_{
        std::exchange(other.addWorkspaceUnavailable_, false)},
      knownEmpty_{std::exchange(other.knownEmpty_, false)},
      mutableDataExposed_{std::exchange(other.mutableDataExposed_, false)}
{
}

template <class Key, class Extent, class Policy, class Allocator>
auto BloomFilter<Key, Extent, Policy, Allocator>::operator=(BloomFilter&& other) noexcept(
  std::is_nothrow_move_assignable_v<AllocatorType> &&
  std::is_nothrow_move_assignable_v<PolicyType>)
  -> BloomFilter&
{
  if (this != &other) {
    Release();
    allocator_ = std::move(other.allocator_);
    data_ = std::exchange(other.data_, nullptr);
    addWorkspace_ = std::exchange(other.addWorkspace_, nullptr);
    addSelectedFlag_ = std::exchange(other.addSelectedFlag_, nullptr);
    numBlocks_ = std::exchange(other.numBlocks_, 0);
    policy_ = std::move(other.policy_);
    addSlotsPerBlock_ = std::exchange(other.addSlotsPerBlock_, 0);
    addProducerCores_ = std::exchange(other.addProducerCores_, 0);
    addWorkspaceUnavailable_ =
      std::exchange(other.addWorkspaceUnavailable_, false);
    knownEmpty_ = std::exchange(other.knownEmpty_, false);
    mutableDataExposed_ = std::exchange(other.mutableDataExposed_, false);
  }
  return *this;
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Clear(aclrtStream stream)
{
  ClearAsync(stream);
  Synchronize(stream, "BloomFilter::Clear");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::ClearAsync(aclrtStream stream)
{
  if (data_ == nullptr || numBlocks_ == 0) {
    return;
  }
  std::uint64_t const words = static_cast<std::uint64_t>(NumWords());
  std::uint32_t const cores = LaunchCoreNum(words);
  aclco::BloomFilterClear<WordType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_), words);
  ResetKnownEmpty();
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Release() noexcept
{
  if (addWorkspace_ != nullptr) {
    allocator_.Deallocate(addWorkspace_);
    addWorkspace_ = nullptr;
  }
  if (addSelectedFlag_ != nullptr) {
    allocator_.Deallocate(addSelectedFlag_);
    addSelectedFlag_ = nullptr;
  }
  if (data_ != nullptr) {
    allocator_.Deallocate(data_);
    data_ = nullptr;
  }
  numBlocks_ = 0;
  addSlotsPerBlock_ = 0;
  addProducerCores_ = 0;
  addWorkspaceUnavailable_ = false;
  MarkDirty();
  mutableDataExposed_ = false;
}

}  // namespace aclco
