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

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "detail/roaring_bitmap/kernels.h"
#include "tiling/platform/platform_ascendc.h"

namespace aclco {

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>::RoaringBitmap(void const* bitmap, Allocator const& allocator, aclrtStream stream)
    : allocator_{allocator}
{
    Initialize(bitmap, detail::kRoaringUnboundedBytes, stream);
}

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>::RoaringBitmap(void const* bitmap, aclrtStream stream)
    : RoaringBitmap(bitmap, Allocator{}, stream)
{}

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>::RoaringBitmap(void const* bitmap, size_t bitmapBytes, Allocator const& allocator,
                                                     aclrtStream stream)
    : allocator_{allocator}
{
    Initialize(bitmap, bitmapBytes, stream);
}

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>::RoaringBitmap(RoaringBitmap&& other) noexcept(
    std::is_nothrow_move_constructible_v<Allocator>)
    : allocator_{std::move(other.allocator_)}
{
    allocation_ = other.allocation_;
    allocationBytes_ = other.allocationBytes_;
    metadataOffset_ = other.metadataOffset_;
    bucketsOffset_ = other.bucketsOffset_;
    serializedBytes_ = other.serializedBytes_;
    numKeys_ = other.numKeys_;
    loadAlignment_ = other.loadAlignment_;

    other.allocation_ = nullptr;
    other.allocationBytes_ = 0;
    other.metadataOffset_ = 0;
    other.bucketsOffset_ = 0;
    other.serializedBytes_ = 0;
    other.numKeys_ = 0;
    other.loadAlignment_ = 0;
}

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>& RoaringBitmap<Key, Extent, Allocator>::operator=(RoaringBitmap&& other) noexcept(
    std::is_nothrow_move_assignable_v<Allocator>)
{
    if (this != &other) {
        Reset();
        MoveFrom(std::move(other));
    }
    return *this;
}

template <typename Key, typename Extent, typename Allocator>
RoaringBitmap<Key, Extent, Allocator>::~RoaringBitmap()
{
    Reset();
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::Contains(void const* keys, void* outputValues, Extent keyNum,
                                                     aclrtStream stream) const
{
    ContainsAsync(keys, outputValues, keyNum, stream);
    CheckAcl(aclrtSynchronizeStream(stream), "RoaringBitmap::Contains aclrtSynchronizeStream");
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::ContainsAsync(void const* keys, void* outputValues, Extent keyNum,
                                                          aclrtStream stream) const
{
    SizeType countValue = static_cast<SizeType>(keyNum);
    if constexpr (std::is_signed_v<SizeType>) {
        if (countValue < 0) {
            throw std::invalid_argument("RoaringBitmap::ContainsAsync keyNum must not be negative");
        }
    }
    uint64_t count = static_cast<uint64_t>(countValue);
    if (count == 0) {
        return;
    }
    if (keys == nullptr || outputValues == nullptr) {
        throw std::invalid_argument("RoaringBitmap::ContainsAsync keys and outputValues must not be null");
    }
    if (allocation_ == nullptr) {
        throw std::invalid_argument("RoaringBitmap::ContainsAsync bitmap storage is not initialized");
    }

    uint64_t requiredBlocks = count / detail::kRoaringThreadsPerBlock +
                              static_cast<uint64_t>(count % detail::kRoaringThreadsPerBlock != 0);
    uint32_t availableBlocks = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint32_t blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(requiredBlocks, std::max<uint32_t>(availableBlocks, 1U)));

    if constexpr (std::is_same_v<Key, uint32_t>) {
        auto const* metadata = reinterpret_cast<detail::RoaringBitmapMetadata32 const*>(allocation_ + metadataOffset_);
        // Dispatch outside the SIMT loop so each query uses one compile-time load strategy.
        if (loadAlignment_ == (detail::kRoaringAligned16 | detail::kRoaringAligned32)) {
            RoaringBitmapContains32AlignedKernel<<<blockDim, 0, stream>>>(
                allocation_, metadata, static_cast<uint32_t const*>(keys), static_cast<bool*>(outputValues), count);
        } else if (loadAlignment_ == detail::kRoaringAligned16) {
            RoaringBitmapContains32Aligned16Kernel<<<blockDim, 0, stream>>>(
                allocation_, metadata, static_cast<uint32_t const*>(keys), static_cast<bool*>(outputValues), count);
        } else {
            RoaringBitmapContains32Kernel<<<blockDim, 0, stream>>>(
                allocation_, metadata, static_cast<uint32_t const*>(keys), static_cast<bool*>(outputValues), count);
        }
    } else {
        auto const* metadata = reinterpret_cast<detail::RoaringBitmapMetadata64 const*>(allocation_ + metadataOffset_);
        auto const* buckets = reinterpret_cast<detail::RoaringBitmapBucket const*>(allocation_ + bucketsOffset_);
        RoaringBitmapContains64Kernel<<<blockDim, 0, stream>>>(allocation_, metadata, buckets,
                                                               static_cast<uint64_t const*>(keys),
                                                               static_cast<bool*>(outputValues), count);
    }
}

template <typename Key, typename Extent, typename Allocator>
uint64_t RoaringBitmap<Key, Extent, Allocator>::Size() const noexcept
{
    return numKeys_;
}

template <typename Key, typename Extent, typename Allocator>
bool RoaringBitmap<Key, Extent, Allocator>::Empty() const noexcept
{
    return numKeys_ == 0;
}

template <typename Key, typename Extent, typename Allocator>
uint8_t const* RoaringBitmap<Key, Extent, Allocator>::Data() const noexcept
{
    return allocation_;
}

template <typename Key, typename Extent, typename Allocator>
uint64_t RoaringBitmap<Key, Extent, Allocator>::SizeBytes() const noexcept
{
    return serializedBytes_;
}

template <typename Key, typename Extent, typename Allocator>
Allocator RoaringBitmap<Key, Extent, Allocator>::GetAllocator() const
{
    return allocator_;
}

template <typename Key, typename Extent, typename Allocator>
typename RoaringBitmap<Key, Extent, Allocator>::RefType RoaringBitmap<Key, Extent, Allocator>::Ref() const noexcept
{
    if constexpr (std::is_same_v<Key, uint32_t>) {
        auto const* metadata = reinterpret_cast<detail::RoaringBitmapMetadata32 const*>(allocation_ + metadataOffset_);
        return RefType{allocation_, metadata, numKeys_, serializedBytes_};
    } else {
        auto const* metadata = reinterpret_cast<detail::RoaringBitmapMetadata64 const*>(allocation_ + metadataOffset_);
        auto const* buckets = reinterpret_cast<detail::RoaringBitmapBucket const*>(allocation_ + bucketsOffset_);
        return RefType{allocation_, metadata, buckets, numKeys_, serializedBytes_};
    }
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::Initialize(void const* bitmap, size_t bitmapBytes, aclrtStream stream)
{
    auto const* bytes = static_cast<uint8_t const*>(bitmap);
    auto parsed = detail::ParseRoaringBitmap<Key>(bytes, bitmapBytes);

    if constexpr (std::is_same_v<Key, uint32_t>) {
        serializedBytes_ = parsed.metadata.sizeBytes;
        numKeys_ = parsed.metadata.numKeys;
        metadataOffset_ = AlignUp(static_cast<size_t>(serializedBytes_), alignof(detail::RoaringBitmapMetadata32));
        if (metadataOffset_ > std::numeric_limits<size_t>::max() - sizeof(detail::RoaringBitmapMetadata32)) {
            throw std::invalid_argument("RoaringBitmap: device allocation size overflow");
        }
        bucketsOffset_ = metadataOffset_ + sizeof(detail::RoaringBitmapMetadata32);
        allocationBytes_ = bucketsOffset_;
    } else {
        serializedBytes_ = parsed.metadata.sizeBytes;
        numKeys_ = parsed.metadata.numKeys;
        metadataOffset_ = AlignUp(static_cast<size_t>(serializedBytes_), alignof(detail::RoaringBitmapMetadata64));
        if (metadataOffset_ > std::numeric_limits<size_t>::max() - sizeof(detail::RoaringBitmapMetadata64)) {
            throw std::invalid_argument("RoaringBitmap: device allocation size overflow");
        }
        size_t metadataEnd = metadataOffset_ + sizeof(detail::RoaringBitmapMetadata64);
        bucketsOffset_ = AlignUp(metadataEnd, alignof(detail::RoaringBitmapBucket));
        if (parsed.buckets.size() >
            (std::numeric_limits<size_t>::max() - bucketsOffset_) / sizeof(detail::RoaringBitmapBucket)) {
            throw std::invalid_argument("RoaringBitmap: device allocation size overflow");
        }
        allocationBytes_ = bucketsOffset_ + parsed.buckets.size() * sizeof(detail::RoaringBitmapBucket);
    }

    constexpr size_t alignment = std::is_same_v<Key, uint32_t> ? alignof(detail::RoaringBitmapMetadata32) :
        std::max(alignof(detail::RoaringBitmapMetadata64), alignof(detail::RoaringBitmapBucket));
    if (allocationBytes_ > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::invalid_argument("RoaringBitmap: device allocation size overflow");
    }
    allocationBytes_ += alignment - 1;
    allocation_ = allocator_.Allocate(allocationBytes_);
    if (allocation_ == nullptr) {
        throw std::bad_alloc{};
    }

    // Keep the serialized bytes at the allocator's address and align the typed metadata independently.
    const auto base = reinterpret_cast<uintptr_t>(allocation_);
    const size_t padding = (alignment - base % alignment) % alignment;
    metadataOffset_ += padding;
    bucketsOffset_ += padding;

    if constexpr (std::is_same_v<Key, uint32_t>) {
        parsed.metadata.aligned16 = static_cast<uint8_t>((base + parsed.metadata.keyCards) % alignof(uint16_t) == 0);
        parsed.metadata.offsetsAligned = static_cast<uint8_t>(
            parsed.metadata.offsetsInSerializedData != 0 &&
            (base + parsed.metadata.containerOffsets) % alignof(uint32_t) == 0);
        loadAlignment_ = parsed.metadata.aligned16 != 0 ? detail::kRoaringAligned16 : 0U;
        if (parsed.metadata.offsetsAligned != 0) {
            loadAlignment_ |= detail::kRoaringAligned32;
        }
    }

    try {
        CheckAcl(aclrtMemcpyAsync(allocation_, allocationBytes_, bytes, static_cast<size_t>(serializedBytes_),
                                  ACL_MEMCPY_HOST_TO_DEVICE, stream),
                 "RoaringBitmap constructor bitmap H2D");
        if constexpr (std::is_same_v<Key, uint32_t>) {
            CheckAcl(aclrtMemcpyAsync(allocation_ + metadataOffset_, allocationBytes_ - metadataOffset_,
                                      &parsed.metadata, sizeof(parsed.metadata), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                     "RoaringBitmap constructor metadata H2D");
        } else {
            CheckAcl(aclrtMemcpyAsync(allocation_ + metadataOffset_, allocationBytes_ - metadataOffset_,
                                      &parsed.metadata, sizeof(parsed.metadata), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                     "RoaringBitmap constructor metadata H2D");
            if (!parsed.buckets.empty()) {
                CheckAcl(
                    aclrtMemcpyAsync(allocation_ + bucketsOffset_, allocationBytes_ - bucketsOffset_,
                                     parsed.buckets.data(), parsed.buckets.size() * sizeof(detail::RoaringBitmapBucket),
                                     ACL_MEMCPY_HOST_TO_DEVICE, stream),
                    "RoaringBitmap constructor buckets H2D");
            }
        }
        CheckAcl(aclrtSynchronizeStream(stream), "RoaringBitmap constructor aclrtSynchronizeStream");
    } catch (...) {
        // A successful memcpy submission can still be pending when a later submission fails.
        // Keep the destination alive until all work already queued on this stream has completed.
        (void)aclrtSynchronizeStream(stream);
        Reset();
        throw;
    }
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::Reset() noexcept
{
    if (allocation_ != nullptr) {
        allocator_.Deallocate(allocation_);
    }
    allocation_ = nullptr;
    allocationBytes_ = 0;
    metadataOffset_ = 0;
    bucketsOffset_ = 0;
    serializedBytes_ = 0;
    numKeys_ = 0;
    loadAlignment_ = 0;
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::MoveFrom(RoaringBitmap&& other) noexcept(
    std::is_nothrow_move_assignable_v<Allocator>)
{
    allocator_ = std::move(other.allocator_);
    allocation_ = other.allocation_;
    allocationBytes_ = other.allocationBytes_;
    metadataOffset_ = other.metadataOffset_;
    bucketsOffset_ = other.bucketsOffset_;
    serializedBytes_ = other.serializedBytes_;
    numKeys_ = other.numKeys_;
    loadAlignment_ = other.loadAlignment_;

    other.allocation_ = nullptr;
    other.allocationBytes_ = 0;
    other.metadataOffset_ = 0;
    other.bucketsOffset_ = 0;
    other.serializedBytes_ = 0;
    other.numKeys_ = 0;
    other.loadAlignment_ = 0;
}

template <typename Key, typename Extent, typename Allocator>
size_t RoaringBitmap<Key, Extent, Allocator>::AlignUp(size_t value, size_t alignment)
{
    size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    size_t increment = alignment - remainder;
    if (increment > std::numeric_limits<size_t>::max() - value) {
        throw std::invalid_argument("RoaringBitmap: device allocation size overflow");
    }
    return value + increment;
}

template <typename Key, typename Extent, typename Allocator>
void RoaringBitmap<Key, Extent, Allocator>::CheckAcl(aclError status, char const* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, ret=" + std::to_string(status));
    }
}

} // namespace aclco
