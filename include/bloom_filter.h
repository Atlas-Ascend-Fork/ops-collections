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

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "bloom_filter_policy.h"
#include "bloom_filter_ref.h"
#include "extent.h"
#include "utility/allocator.h"

namespace aclco {

/**
 * @brief Fixed-size, NPU-accelerated sectorized Bloom filter.
 *
 * The container owns a device bit array. It guarantees no false negatives for
 * keys added under the documented stream-ordering rules; queries for keys that
 * were not added may return false positives.
 */
template <class Key, class Extent = aclco::Extent<std::size_t>, class Policy = aclco::BloomFilterPolicy<Key>,
          class Allocator = aclco::DefaultAllocator<typename Policy::WordType>>
class BloomFilter {
public:
    using KeyType = Key;
    using ExtentType = Extent;
    using SizeType = typename ExtentType::ValueType;
    using PolicyType = Policy;
    using WordType = typename PolicyType::WordType;
    using AllocatorType = Allocator;
    using RefType = BloomFilterRef<KeyType, SizeType, PolicyType>;

    static constexpr std::uint32_t wordsPerBlock = PolicyType::wordsPerBlock;

    BloomFilter(BloomFilter const&) = delete;
    BloomFilter& operator=(BloomFilter const&) = delete;

    BloomFilter(BloomFilter&& other) noexcept(std::is_nothrow_move_constructible_v<AllocatorType> &&
                                              std::is_nothrow_move_constructible_v<PolicyType>);
    BloomFilter& operator=(BloomFilter&& other) noexcept(std::is_nothrow_move_assignable_v<AllocatorType> &&
                                                         std::is_nothrow_move_assignable_v<PolicyType>);

    explicit BloomFilter(ExtentType numBlocks, PolicyType const& policy = PolicyType{},
                         AllocatorType const& allocator = AllocatorType{}, aclrtStream stream = nullptr);

    explicit BloomFilter(ExtentType numBlocks, PolicyType const& policy, aclrtStream stream);

    ~BloomFilter();

    void Clear(aclrtStream stream = nullptr);
    void ClearAsync(aclrtStream stream = nullptr);

    void Add(void* keys, ExtentType keyNum, aclrtStream stream = nullptr);
    void AddAsync(void* keys, ExtentType keyNum, aclrtStream stream = nullptr);
    void AddIf(void* keys, void* stencil, ExtentType keyNum, aclrtStream stream = nullptr);
    void AddIfAsync(void* keys, void* stencil, ExtentType keyNum, aclrtStream stream = nullptr);

    void Contains(void* keys, ExtentType keyNum, void* outputValues, aclrtStream stream = nullptr) const;
    void Contains(void* keys, void* outputValues, ExtentType keyNum, aclrtStream stream = nullptr) const;
    void ContainsAsync(void* keys, ExtentType keyNum, void* outputValues, aclrtStream stream = nullptr) const;
    void ContainsAsync(void* keys, void* outputValues, ExtentType keyNum, aclrtStream stream = nullptr) const;
    void ContainsIf(void* keys, void* stencil, void* outputValues, ExtentType keyNum,
                    aclrtStream stream = nullptr) const;
    void ContainsIfAsync(void* keys, void* stencil, void* outputValues, ExtentType keyNum,
                         aclrtStream stream = nullptr) const;

    void Merge(BloomFilter const& other, aclrtStream stream = nullptr);
    void MergeAsync(BloomFilter const& other, aclrtStream stream = nullptr);

    void Intersect(BloomFilter const& other, aclrtStream stream = nullptr);
    void IntersectAsync(BloomFilter const& other, aclrtStream stream = nullptr);

    [[nodiscard]] WordType* Data() noexcept;
    [[nodiscard]] WordType const* Data() const noexcept;
    [[nodiscard]] ExtentType BlockExtent() const noexcept;
    [[nodiscard]] SizeType NumWords() const noexcept;
    [[nodiscard]] std::size_t SizeBytes() const noexcept;
    [[nodiscard]] PolicyType const& GetPolicy() const noexcept;

private:
    static_assert(std::is_same_v<KeyType, std::int32_t> || std::is_same_v<KeyType, std::uint32_t> ||
                      std::is_same_v<KeyType, std::int64_t> || std::is_same_v<KeyType, std::uint64_t> ||
                      std::is_same_v<KeyType, float>,
                  "BloomFilter supports int32_t, uint32_t, int64_t, uint64_t and float keys");
    static_assert(std::is_same_v<typename PolicyType::KeyType, KeyType>, "BloomFilter policy key type must match Key");
    static_assert(std::is_same_v<typename PolicyType::Hasher, aclco::xxhash_64<KeyType>>,
                  "BloomFilter kernels currently support xxhash_64<Key> policies only");
    static_assert(std::is_integral_v<SizeType> && std::is_unsigned_v<SizeType>,
                  "BloomFilter extent value type must be an unsigned integer");

    void ValidateCompatible(BloomFilter const& other, char const* operation) const;
    static void Synchronize(aclrtStream stream, char const* operation);
    static std::uint32_t LaunchCoreNum(std::uint64_t workItems);
    bool EnsureAddWorkspace(std::uint64_t keyNum, aclrtStream stream) noexcept;
    // knownEmpty_ is valid only until the next write or mutable Data() exposure.
    // Every mutating path calls MarkDirty(); Clear() is the only path that can
    // restore the known-empty state through ResetKnownEmpty().
    void MarkDirty() noexcept { knownEmpty_ = false; }
    void ResetKnownEmpty() noexcept { knownEmpty_ = !mutableDataExposed_; }
    bool EnsureSelectedFlag(aclrtStream stream);
    bool TryPrivateAdd(void* keys, void* stencil, SizeType count, bool preserveExisting, aclrtStream stream);
    void Release() noexcept;

    AllocatorType allocator_;
    WordType* data_{nullptr};
    WordType* addWorkspace_{nullptr};
    WordType* addSelectedFlag_{nullptr};
    SizeType numBlocks_{0};
    PolicyType policy_;
    std::uint32_t addSlotsPerBlock_{0};
    std::uint32_t addProducerCores_{0};
    bool addWorkspaceUnavailable_{false};
    // Used only to omit an unnecessary block load in routed Add Apply.
    bool knownEmpty_{false};
    // Sticky after mutable Data() exposure because external writes are opaque.
    bool mutableDataExposed_{false};
};

} // namespace aclco

#include "detail/bloom_filter/bloom_filter.inl"
