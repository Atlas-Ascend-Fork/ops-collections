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
void BloomFilter<Key, Extent, Policy, Allocator>::Merge(BloomFilter const& other,
                                                        aclrtStream stream)
{
  MergeAsync(other, stream);
  Synchronize(stream, "BloomFilter::Merge");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::MergeAsync(BloomFilter const& other,
                                                             aclrtStream stream)
{
  ValidateCompatible(other, "Merge");
  if (this == &other) {
    return;
  }
  MarkDirty();
  std::uint64_t const words = static_cast<std::uint64_t>(NumWords());
  std::uint32_t const cores = LaunchCoreNum(words);
  aclco::BloomFilterMerge<WordType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(other.data_),
    words);
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::Intersect(BloomFilter const& other,
                                                            aclrtStream stream)
{
  IntersectAsync(other, stream);
  Synchronize(stream, "BloomFilter::Intersect");
}

template <class Key, class Extent, class Policy, class Allocator>
void BloomFilter<Key, Extent, Policy, Allocator>::IntersectAsync(BloomFilter const& other,
                                                                 aclrtStream stream)
{
  ValidateCompatible(other, "Intersect");
  if (this == &other) {
    return;
  }
  MarkDirty();
  std::uint64_t const words = static_cast<std::uint64_t>(NumWords());
  std::uint32_t const cores = LaunchCoreNum(words);
  aclco::BloomFilterIntersect<WordType><<<cores, 0, stream>>>(
    reinterpret_cast<std::uint8_t*>(data_),
    reinterpret_cast<std::uint8_t*>(other.data_),
    words);
}

}  // namespace aclco
