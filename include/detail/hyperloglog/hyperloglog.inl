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
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include "detail/hyperloglog/kernels.h"
#include "detail/hyperloglog/finalizer.h"
#include "tiling/platform/platform_ascendc.h"

namespace aclco::detail::hyperloglog {
inline void Check(aclError status, char const* operation)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string("HyperLogLog: ") + operation + " failed, ret=" + std::to_string(status));
}
inline void Sync(aclrtStream stream) { Check(aclrtSynchronizeStream(stream), "synchronize stream"); }
template <class Key>
inline void ValidateInput(void* keys, std::size_t n, int32_t device)
{
    if (!keys || reinterpret_cast<std::uintptr_t>(keys) % alignof(Key) != 0 ||
        n > std::numeric_limits<std::size_t>::max() / sizeof(Key))
        throw std::invalid_argument("HyperLogLog: invalid input pointer or size");
    aclrtPtrAttributes attr{};
    Check(aclrtPointerGetAttributes(keys, &attr), "query input attributes");
    if (attr.location.type != ACL_MEM_LOCATION_TYPE_DEVICE || attr.location.id != device)
        throw std::invalid_argument("HyperLogLog: input must belong to the current device");
    void* base = nullptr;
    std::size_t bytes = 0;
    Check(aclrtMemGetAddressRange(keys, &base, &bytes), "query input allocation");
    auto const addr = reinterpret_cast<std::uintptr_t>(keys);
    auto const start = reinterpret_cast<std::uintptr_t>(base);
    if (addr < start || addr - start > bytes || n * sizeof(Key) > bytes - (addr - start))
        throw std::invalid_argument("HyperLogLog: input exceeds device allocation");
}
template <class Key, unsigned Precision>
inline void LaunchPrivate(Key const* keys, std::uint32_t* workspace, std::uint32_t count, std::uint32_t cores,
                          aclrtStream stream, std::uint32_t const* initialSketch = nullptr)
{
    constexpr unsigned pack = Precision <= 15 ? 1 : (Precision <= 17 ? 4 : 5);
    constexpr unsigned bytes = ((((1U << Precision) + pack - 1U) / pack) * 4U + 31U) & ~31U;
    constexpr unsigned inputBytes = 2 * TiledInputBytes<Precision>();
    if constexpr (Precision <= 17)
        TiledAddKernel<Key, pack, Precision>
            <<<cores, bytes + inputBytes, stream>>>(keys, workspace, count, initialSketch);
    else
        PrivateAddKernel<Key, pack, Precision><<<cores, bytes, stream>>>(keys, workspace, count);
}
inline void LaunchReduce(std::uint32_t* words, std::uint32_t const* sketches, unsigned wordCount, unsigned sketchCount,
                         unsigned cores, aclrtStream stream)
{
    unsigned tileWords = ((wordCount + cores - 1) / cores + 7U) & ~7U;
    tileWords = std::min(tileWords, std::min(768U, (216U * 1024U / (sketchCount * 32U)) * 8U));
    ReduceSketchesKernel<>
        <<<cores, tileWords * sketchCount * 4U, stream>>>(words, sketches, wordCount, sketchCount, tileWords);
}
template <class Key, unsigned Precision>
inline void LaunchLargeAdd(Key const* keys, std::uint32_t* words, std::uint32_t* workspace, unsigned count,
                           unsigned addCores, unsigned reduceCores, aclrtStream stream)
{
    if constexpr (Precision == 16) {
        constexpr unsigned bytes = (1U << Precision) + 2 * TiledInputBytes<Precision>();
        FusedP16AddKernel<Key><<<addCores, bytes, stream>>>(keys, words, workspace, count);
        return;
    }
    unsigned processed = 0;
    if constexpr (Precision >= 15 && Precision <= 17) {
        // Max is associative and idempotent. Share early results so later private
        // sketches can reject most redundant atomic updates.
        // Keep full-size stages balanced across cores.
        unsigned stageAlignment = addCores * (TiledInputBytes<Precision>() / sizeof(Key));
        constexpr unsigned boundaries[]{1U << 20, 8U << 20};
        for (unsigned boundary : boundaries) {
            unsigned alignedBoundary = std::max(stageAlignment, (boundary / stageAlignment) * stageAlignment);
            unsigned end = std::min(count / 2, alignedBoundary);
            if (end == processed)
                continue;
            LaunchPrivate<Key, Precision>(keys + processed, workspace, end - processed, addCores, stream,
                                          processed == 0 ? nullptr : words);
            LaunchReduce(words, workspace, (1U << Precision) / 4, addCores, reduceCores, stream);
            processed = end;
        }
    }
    if constexpr (Precision >= 15 && Precision <= 17) {
        LaunchPrivate<Key, Precision>(keys + processed, workspace, count - processed, addCores, stream,
                                      processed == 0 ? nullptr : words);
    } else {
        LaunchPrivate<Key, Precision>(keys + processed, workspace, count - processed, addCores, stream);
    }
    LaunchReduce(words, workspace, (1U << Precision) / 4, addCores, reduceCores, stream);
}

template <class Key, unsigned Precision>
inline void LaunchSmallAdd(Key const* keys, std::uint32_t* words, std::uint32_t* workspace, unsigned count,
                           unsigned addCores, unsigned reduceCores, aclrtStream stream)
{
    constexpr unsigned elementsPerCore = Precision <= 17 ? TiledInputBytes<Precision>() / sizeof(Key) : 4096U;
    auto const activeCores = std::min(addCores, (count + elementsPerCore - 1) / elementsPerCore);
    LaunchPrivate<Key, Precision>(keys, workspace, count, activeCores, stream);
    LaunchReduce(words, workspace, (1U << Precision) / 4, activeCores, reduceCores, stream);
}
} // namespace aclco::detail::hyperloglog
namespace aclco {
template <class Key>
HyperLogLog<Key> HyperLogLog<Key>::CreateWithSketchSizeKB(std::uint32_t kb, aclrtStream stream)
{
    if (kb < 8 || kb > 256 || (kb & (kb - 1)) != 0)
        throw std::invalid_argument("HyperLogLog: invalid sketch size");
    std::uint32_t p = 13;
    while ((1U << p) < kb * 1024U)
        ++p;
    return HyperLogLog(p, stream);
}
template <class Key>
HyperLogLog<Key> HyperLogLog<Key>::CreateWithPrecision(std::uint32_t p, aclrtStream stream)
{
    if (p < 13 || p > 18)
        throw std::invalid_argument("HyperLogLog: precision must be in [13, 18]");
    return HyperLogLog(p, stream);
}
template <class Key>
HyperLogLog<Key> HyperLogLog<Key>::CreateWithStandardDeviation(double deviation, aclrtStream stream)
{
    if (!std::isfinite(deviation) || deviation <= 0)
        throw std::invalid_argument("HyperLogLog: invalid deviation");
    for (std::uint32_t p = 13; p <= 18; ++p) {
        if (1.04 / std::sqrt(static_cast<double>(1U << p)) <= deviation)
            return HyperLogLog(p, stream);
    }
    throw std::invalid_argument("HyperLogLog: deviation requires more than 256 KiB");
}
template <class Key>
HyperLogLog<Key>::HyperLogLog(std::uint32_t p, aclrtStream stream) : precision_(p)
{
    namespace h = detail::hyperloglog;
    h::Check(aclrtGetDevice(&device_), "get device");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform)
        throw std::runtime_error("HyperLogLog: unavailable device platform");
    cores_ = std::max(1U, static_cast<std::uint32_t>(platform->GetCoreNumAiv()));
    addCores_ = cores_;
    cores_ = std::min(cores_, (1U << p) / (4U * h::threads));
    hostHistogram_.resize(cores_ * h::bins);
    auto const sketchBytes = std::size_t{1} << p;
    auto const workspaceOffset = (sketchBytes + hostHistogram_.size() * 4 + 31U) & ~std::size_t{31U};
    h::Check(aclrtMalloc(reinterpret_cast<void**>(&words_), workspaceOffset + sketchBytes * addCores_,
                         ACL_MEM_MALLOC_HUGE_FIRST),
             "allocate sketch and workspace");
    histogram_ = words_ + sketchBytes / 4;
    addWorkspace_ = words_ + workspaceOffset / 4;
    try {
        Clear(stream);
    } catch (...) {
        Release();
        throw;
    }
}
template <class Key>
HyperLogLog<Key>::HyperLogLog(HyperLogLog&& other) noexcept
{
    *this = std::move(other);
}
template <class Key>
HyperLogLog<Key>& HyperLogLog<Key>::operator=(HyperLogLog&& other) noexcept
{
    if (this != &other) {
        Release();
        words_ = std::exchange(other.words_, nullptr);
        histogram_ = std::exchange(other.histogram_, nullptr);
        addWorkspace_ = std::exchange(other.addWorkspace_, nullptr);
        addCores_ = std::exchange(other.addCores_, 0);
        precision_ = std::exchange(other.precision_, 0);
        cores_ = std::exchange(other.cores_, 0);
        device_ = std::exchange(other.device_, -1);
        knownEmpty_ = std::exchange(other.knownEmpty_, false);
        hostHistogram_ = std::move(other.hostHistogram_);
    }
    return *this;
}
template <class Key>
HyperLogLog<Key>::~HyperLogLog()
{
    Release();
}
template <class Key>
void HyperLogLog<Key>::Release() noexcept
{
    addWorkspace_ = nullptr;
    if (words_)
        (void)aclrtFree(words_);
    words_ = nullptr;
    histogram_ = nullptr;
}
template <class Key>
void HyperLogLog<Key>::Validate() const
{
    if (!words_)
        throw std::logic_error("HyperLogLog: moved-from or uninitialized sketch");
    std::int32_t device;
    detail::hyperloglog::Check(aclrtGetDevice(&device), "get device");
    if (device != device_)
        throw std::invalid_argument("HyperLogLog: current device differs from owner device");
}
template <class Key>
void HyperLogLog<Key>::Clear(aclrtStream stream)
{
    Validate();
    knownEmpty_ = false;
    detail::hyperloglog::ClearKernel<><<<cores_, 0, stream>>>(words_, (1U << precision_) / 4);
    detail::hyperloglog::Sync(stream);
    knownEmpty_ = true;
}
template <class Key>
void HyperLogLog<Key>::Add(void* keys, ExtentType count, aclrtStream stream)
{
    Validate();
    auto const n = static_cast<std::size_t>(count);
    if (n == 0) {
        detail::hyperloglog::Sync(stream);
        return;
    }
    namespace h = detail::hyperloglog;
    h::ValidateInput<Key>(keys, n, device_);
    // Private sketches amortize their initialization on a fresh sketch, but
    // regress redundant Adds whose global registers already reject most keys.
    // Preserve the direct path after the first nonempty Add and for tiny inputs.
    auto const smallThreshold = precision_ <= 14 ? (1U << 14) : (1U << 18);
    bool const useSmall = knownEmpty_ && n >= smallThreshold && n < (1U << 20);
    knownEmpty_ = false;
    if (useSmall) {
        switch (precision_) {
            case 13:
                h::LaunchSmallAdd<Key, 13>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
            case 14:
                h::LaunchSmallAdd<Key, 14>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
            case 15:
                h::LaunchSmallAdd<Key, 15>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
            case 16:
                h::LaunchSmallAdd<Key, 16>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
            case 17:
                h::LaunchSmallAdd<Key, 17>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
            case 18:
                h::LaunchSmallAdd<Key, 18>(static_cast<Key*>(keys), words_, addWorkspace_, n, addCores_, cores_,
                                           stream);
                break;
        }
        h::Sync(stream);
        return;
    }
    // Device fast paths use 32-bit offsets. Keep their existing safe bound,
    // but split larger inputs instead of sending the entire input to AddKernel.
    // Full chunks keep DMA input addresses aligned; the final chunk may be small.
    // Each reduction merges with words_, so prior Adds and chunks are retained,
    // including p18 whose private sketches always start from zero.
    for (std::size_t offset = 0; offset < n;) {
        auto const remaining = n - offset;
        auto const batch = remaining <= 0x7fffffffU ? remaining : std::size_t{1U << 30};
        auto* batchKeys = static_cast<Key*>(keys) + offset;
        if (batch >= (1U << 20)) {
            switch (precision_) {
                case 13:
                    h::LaunchLargeAdd<Key, 13>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
                case 14:
                    h::LaunchLargeAdd<Key, 14>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
                case 15:
                    h::LaunchLargeAdd<Key, 15>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
                case 16:
                    h::LaunchLargeAdd<Key, 16>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
                case 17:
                    h::LaunchLargeAdd<Key, 17>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
                case 18:
                    h::LaunchLargeAdd<Key, 18>(batchKeys, words_, addWorkspace_,
                                               static_cast<std::uint32_t>(batch), addCores_, cores_, stream);
                    break;
            }
        } else {
            auto const launchCores = std::min<std::size_t>(addCores_, (batch + h::threads - 1) / h::threads);
            h::AddKernel<Key><<<launchCores, 0, stream>>>(words_, batchKeys, batch, precision_);
        }
        offset += batch;
    }
    h::Sync(stream);
}
template <class Key>
void HyperLogLog<Key>::Merge(HyperLogLog const& other, aclrtStream stream)
{
    Validate();
    other.Validate();
    if (precision_ != other.precision_ || device_ != other.device_)
        throw std::invalid_argument("HyperLogLog: incompatible sketches");
    if (this != &other) {
        knownEmpty_ = knownEmpty_ && other.knownEmpty_;
        detail::hyperloglog::MergeKernel<><<<cores_, 0, stream>>>(words_, other.words_, (1U << precision_) / 4);
    }
    detail::hyperloglog::Sync(stream);
}
template <class Key>
std::uint64_t HyperLogLog<Key>::Estimate(aclrtStream stream) const
{
    Validate();
    namespace h = detail::hyperloglog;
    h::HistogramKernel<><<<cores_, 0, stream>>>(words_, histogram_, (1U << precision_) / 4);
    h::Sync(stream);
    h::Check(aclrtMemcpy(hostHistogram_.data(), hostHistogram_.size() * 4, histogram_, hostHistogram_.size() * 4,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "copy histogram");
    std::uint32_t histogram[h::bins]{};
    for (std::uint32_t c = 0; c < cores_; ++c)
        for (std::uint32_t rank = 0; rank < h::bins; ++rank)
            histogram[rank] += hostHistogram_[c * h::bins + rank];
    double z = 0;
    for (std::uint32_t rank = 0; rank < h::bins; ++rank)
        z += std::ldexp(static_cast<double>(histogram[rank]), -static_cast<int>(rank));
    return h::Finalizer(precision_)(z, histogram[0]);
}
} // namespace aclco
