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

#include <acl/acl.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "extent.h"
#include "tiling/platform/platform_ascendc.h"
#include "detail/static_multiset/kernels.h"

namespace aclco {
/**
 * @brief Fixed-capacity multiset with exact integer multiplicities.
 * @tparam Key int32_t or int64_t.
 * @details All bulk calls are synchronous on the supplied stream.
 * Device buffers are supplied by the caller; nullptr selects the default ACL stream.
 * Calls on the same object must be externally serialized, including across streams.
 * RetrieveAll preserves accepted insertion order. Overflow retains the earliest
 * valid selected keys; every duplicate consumes one element of logical capacity.
 */
template <typename Key>
class StaticMultiset {
    static_assert(std::is_same_v<Key, int32_t> || std::is_same_v<Key, int64_t>,
                  "StaticMultiset supports int32_t and int64_t keys");

public:
    using KeyType = Key;
    using SizeType = uint64_t;
    using Extent = aclco::Extent<std::size_t>;
    using RefType = StaticMultisetRef<Key>;
    using Hasher = std::conditional_t<sizeof(Key) == 4, murmurhash3_fmix32<Key>, murmurhash3_fmix64<Key>>;

    /**
     * @brief Create an empty container with a fixed logical capacity.
     * @param capacity Maximum number of stored elements, including duplicates.
     * @param emptyKey Reserved sentinel that cannot be inserted.
     * @throws std::invalid_argument If capacity is zero or storage sizes overflow.
     * @throws std::runtime_error If ACL allocation or stream synchronization fails.
     * @note Hash keys and counters are allocated only when their representation is required.
     */
    StaticMultiset(Extent capacity, Key emptyKey, aclrtStream stream = nullptr) : capacity_(capacity), empty_(emptyKey)
    {
        if (!capacity_ || capacity_ > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t)) {
            throw std::invalid_argument("StaticMultiset: invalid capacity");
        }
        tableCapacity_ = 1;
        while (tableCapacity_ < capacity_) {
            tableCapacity_ *= 2;
        }
        constexpr std::size_t slotBytes = sizeof(Key) + sizeof(uint64_t);
        constexpr std::size_t workBytes = detail::static_multiset::WORK_SIZE * sizeof(uint64_t);
        constexpr std::size_t maxBytes = std::numeric_limits<std::size_t>::max();
        if (capacity_ > (maxBytes - workBytes) / sizeof(Key) ||
            tableCapacity_ > (maxBytes - workBytes - capacity_ * sizeof(Key)) / slotBytes) {
            throw std::invalid_argument("StaticMultiset: hash storage size overflow");
        }
        ordered_ = Allocate(capacity_ * sizeof(Key));
        work_ = Allocate(detail::static_multiset::WORK_SIZE * sizeof(uint64_t));
        Clear(stream);
    }
    StaticMultiset(const StaticMultiset&) = delete;
    StaticMultiset& operator=(const StaticMultiset&) = delete;
    /**
     * @brief Transfer all storage and state from another container.
     * @param other Source container; only destruction or move assignment is valid after the move.
     */
    StaticMultiset(StaticMultiset&& other) noexcept
        : capacity_(other.capacity_),
          tableCapacity_(other.tableCapacity_),
          size_(other.size_),
          empty_(other.empty_),
          direct_(other.direct_),
          directBase_(other.directBase_),
          uniformCount_(other.uniformCount_),
          uniformLength_(other.uniformLength_),
          counts_(std::move(other.counts_)),
          keys_(std::move(other.keys_)),
          ordered_(std::move(other.ordered_)),
          work_(std::move(other.work_))
    {
        other.capacity_ = other.tableCapacity_ = other.size_ = 0;
    }
    /**
     * @brief Release owned storage and transfer another container.
     * @param other Source container; self-assignment has no effect.
     * @return This container.
     */
    StaticMultiset& operator=(StaticMultiset&& other) noexcept
    {
        if (this != &other) {
            capacity_ = other.capacity_;
            tableCapacity_ = other.tableCapacity_;
            size_ = other.size_;
            empty_ = other.empty_;
            direct_ = other.direct_;
            directBase_ = other.directBase_;
            uniformCount_ = other.uniformCount_;
            uniformLength_ = other.uniformLength_;
            counts_ = std::move(other.counts_);
            keys_ = std::move(other.keys_);
            ordered_ = std::move(other.ordered_);
            work_ = std::move(other.work_);
            other.capacity_ = other.tableCapacity_ = other.size_ = 0;
        }
        return *this;
    }
    /**
     * @brief Destroy the container and release all owned device storage.
     */
    ~StaticMultiset() = default;

    /**
     * @brief Return the fixed logical capacity, including duplicate elements.
     * @return Requested capacity, or zero for a moved-from container.
     */
    SizeType Capacity() const noexcept { return capacity_; }
    /**
     * @brief Return the number of stored elements, including duplicates.
     * @return Current element count.
     * @throws std::logic_error If the container has been moved from.
     */
    SizeType Size(aclrtStream stream = nullptr) const
    {
        CheckState();
        Sync(stream);
        return size_;
    }
    /**
     * @brief Remove all elements and release optional hash and counter arrays.
     * @note Capacity and the element array are retained. The next general insertion allocates its required storage.
     */
    void Clear(aclrtStream stream = nullptr)
    {
        CheckState();
        Sync(stream);
        keys_.reset();
        counts_.reset();
        size_ = 0;
        direct_ = false;
        directBase_ = 0;
        uniformCount_ = uniformLength_ = 0;
    }

    /**
     * @brief Insert keys, preserving the earliest valid keys when capacity is exhausted.
     * @return Number of failed insertions, including sentinel keys and capacity overflow.
     * @note A null keys pointer with nonzero n returns n without changing the container.
     */
    SizeType Insert(void* keys, Extent n, aclrtStream stream = nullptr)
    {
        if (static_cast<SizeType>(n) <= detail::static_multiset::SMALL_INSERT_LIMIT) {
            return InsertSmallImpl<Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, n, {}, stream);
        }
        return InsertImpl<Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, n, {}, stream);
    }

    template <typename Stencil, typename Predicate>
    SizeType InsertIf(void* keys, Stencil* stencil, Extent n, aclrtStream stream = nullptr)
    {
        return InsertIf(keys, n, stencil, Predicate{}, stream);
    }
    /**
     * @brief Insert keys selected by a device predicate.
     * @param stencil Device array of n predicate inputs.
     * @param pred Device-callable predicate object; its state is preserved.
     * @return Failed selected insertions; unselected inputs are not failures.
     */
    template <typename Stencil, typename Predicate>
    SizeType InsertIf(void* keys, Extent n, Stencil* stencil, Predicate pred, aclrtStream stream = nullptr)
    {
        if (static_cast<SizeType>(n) <= detail::static_multiset::SMALL_INSERT_LIMIT) {
            return InsertSmallImpl<Stencil, Predicate, true>(keys, stencil, n, pred, stream);
        }
        return InsertImpl<Stencil, Predicate, true>(keys, stencil, n, pred, stream);
    }

    /**
     * @brief Write membership as one-byte 0/1 for each input key.
     */
    void Contains(void* keys, void* output, Extent n, aclrtStream stream = nullptr)
    {
        Query<0, Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, output, n, {}, stream);
    }
    void Contains(void* keys, Extent n, void* output, aclrtStream stream = nullptr)
    {
        Contains(keys, output, n, stream);
    }
    /**
     * @brief Write the matching Key or emptyKey for each input key.
     */
    void Find(void* keys, void* output, Extent n, aclrtStream stream = nullptr)
    {
        Query<1, Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, output, n, {}, stream);
    }
    void Find(void* keys, Extent n, void* output, aclrtStream stream = nullptr) { Find(keys, output, n, stream); }
    template <typename Stencil, typename Predicate>
    void ContainsIf(void* keys, Stencil* stencil, void* output, Extent n, aclrtStream stream = nullptr)
    {
        ContainsIf(keys, n, stencil, Predicate{}, output, stream);
    }
    /**
     * @brief Write membership as one-byte 0/1; unselected inputs produce zero.
     * @param stencil Device array of n predicate inputs.
     * @param pred Device-callable predicate object; its state is preserved.
     */
    template <typename Stencil, typename Predicate>
    void ContainsIf(void* keys, Extent n, Stencil* stencil, Predicate pred, void* output, aclrtStream stream = nullptr)
    {
        Query<0, Stencil, Predicate, true>(keys, stencil, output, n, pred, stream);
    }
    template <typename Stencil, typename Predicate>
    void FindIf(void* keys, Stencil* stencil, void* output, Extent n, aclrtStream stream = nullptr)
    {
        FindIf(keys, n, stencil, Predicate{}, output, stream);
    }
    /**
     * @brief Write each selected matching key, or emptyKey for unselected or missing inputs.
     * @param stencil Device array of n predicate inputs.
     * @param pred Device-callable predicate object; its state is preserved.
     */
    template <typename Stencil, typename Predicate>
    void FindIf(void* keys, Extent n, Stencil* stencil, Predicate pred, void* output, aclrtStream stream = nullptr)
    {
        Query<1, Stencil, Predicate, true>(keys, stencil, output, n, pred, stream);
    }

    /**
     * @brief Write the exact uint64_t multiplicity of each query key.
     */
    void CountEach(void* keys, void* output, Extent n, aclrtStream stream = nullptr)
    {
        Query<2, Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, output, n, {}, stream);
    }
    void CountEach(void* keys, Extent n, void* output, aclrtStream stream = nullptr)
    {
        CountEach(keys, output, n, stream);
    }
    /**
     * @brief Write uint64_t multiplicity, replacing zero with one.
     */
    void CountEachOuter(void* keys, void* output, Extent n, aclrtStream stream = nullptr)
    {
        Query<3, Key, detail::static_multiset::AcceptAll, false>(keys, nullptr, output, n, {}, stream);
    }
    void CountEachOuter(void* keys, Extent n, void* output, aclrtStream stream = nullptr)
    {
        CountEachOuter(keys, output, n, stream);
    }

    /**
     * @brief Count with an explicit compatible equality and probe hash.
     * @param equal Device equality object consistent with exact key equality.
     * @param hash Device hash object consistent with Hasher and the table construction policy.
     * @note Output is uint64_t; missing keys produce zero.
     * @pre Equality and hash must match the construction policy; incompatible probing is unsupported.
     */
    template <typename Equal, typename Hash>
    void CountEach(void* keys, Extent n, Equal equal, Hash hash, void* output, aclrtStream stream = nullptr)
    {
        ProbeCount<false>(keys, n, equal, hash, output, stream);
    }
    /**
     * @brief Count with an explicit compatible equality and probe hash.
     * @param equal Device equality object consistent with exact key equality.
     * @param hash Device hash object consistent with Hasher and the table construction policy.
     * @note Output is uint64_t; missing keys produce one.
     * @pre Equality and hash must match the construction policy; incompatible probing is unsupported.
     */
    template <typename Equal, typename Hash>
    void CountEachOuter(void* keys, Extent n, Equal equal, Hash hash, void* output, aclrtStream stream = nullptr)
    {
        ProbeCount<true>(keys, n, equal, hash, output, stream);
    }

    /**
     * @brief Sum multiplicities over all query positions, including repeated queries.
     * @return Exact total match count.
     * @throws std::overflow_error If n times Size cannot be represented by SizeType.
     */
    SizeType Count(void* keys, Extent n, aclrtStream stream = nullptr) { return CountImpl<false>(keys, n, stream); }
    /**
     * @brief Write all matching pairs in query order.
     * @param probeOutput Device Key array receiving each repeated query key.
     * @param matchOutput Device Key array receiving its matching stored key.
     * @param outputCapacity Available Key elements in each output array.
     * @return Number of pairs written.
     * @throws std::length_error If outputCapacity is too small; neither output is written.
     * @note Output pointers may be null when there are no matches.
     */
    SizeType Retrieve(void* keys, Extent n, void* probeOutput, void* matchOutput, Extent outputCapacity,
                      aclrtStream stream = nullptr)
    {
        SizeType count = CountImpl<true>(keys, n, stream);
        if (count > SizeType(outputCapacity)) {
            throw std::length_error("StaticMultiset: output capacity too small");
        }
        if (!count) {
            return 0;
        }
        RequirePointer(probeOutput);
        RequirePointer(matchOutput);
        detail::static_multiset::RetrieveKernel<Key, true><<<detail::static_multiset::BLOCKS, 0, stream>>>(
            Bytes(keys), n, empty_, KeysData(), CountsData(), tableCapacity_, work_.get(), Bytes(probeOutput),
            Bytes(matchOutput), direct_, directBase_, uniformCount_, uniformLength_);
        Sync(stream);
        return count;
    }
    /**
     * @brief Write every stored key in accepted insertion order.
     * @param output Device Key array; may be null when Size is zero.
     * @param outputCapacity Available Key elements in output.
     * @return Number of stored elements written.
     * @throws std::length_error If outputCapacity is smaller than Size; output is not written.
     */
    SizeType RetrieveAll(void* output, Extent outputCapacity, aclrtStream stream = nullptr)
    {
        CheckState();
        if (size_ > SizeType(outputCapacity)) {
            throw std::length_error("StaticMultiset: output capacity too small");
        }
        if (!size_) {
            return 0;
        }
        RequirePointer(output);
        Check(aclrtMemcpyAsync(output, size_ * sizeof(Key), ordered_.get(), size_ * sizeof(Key),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "retrieve all");
        Sync(stream);
        return size_;
    }

private:
    struct FreeDevice {
        void operator()(uint8_t* ptr) const noexcept
        {
            if (ptr) {
                aclrtFree(ptr);
            }
        }
    };
    using Buffer = std::unique_ptr<uint8_t, FreeDevice>;
    static void Check(aclError error, const char* operation)
    {
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(std::string("StaticMultiset ") + operation + ": ACL error " +
                                     std::to_string(error));
        }
    }
    static Buffer Allocate(std::size_t bytes)
    {
        void* ptr = nullptr;
        Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
        return Buffer(static_cast<uint8_t*>(ptr));
    }
    static uint8_t* Bytes(void* ptr) { return static_cast<uint8_t*>(ptr); }
    static void RequirePointer(const void* ptr)
    {
        if (!ptr) {
            throw std::invalid_argument("StaticMultiset: null pointer for nonempty operation");
        }
    }
    static void Sync(aclrtStream stream) { Check(aclrtSynchronizeStream(stream), "synchronize stream"); }
    static uint32_t QueryBlocks() { return platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv(); }
    uint8_t* CountsData() const { return counts_.get(); }
    uint8_t* KeysData() const { return keys_.get(); }
    void CheckState() const
    {
        if (!ordered_) {
            throw std::logic_error("StaticMultiset: moved-from object");
        }
    }
    // Allocate only the arrays required by the new representation. Allocate both
    // missing arrays before publishing either one, so allocation failure leaves
    // an existing interval or direct representation usable.
    void InitializeStorage(bool hash, aclrtStream stream)
    {
        Buffer counts = counts_ ? Buffer{} : Allocate(tableCapacity_ * sizeof(uint64_t));
        Buffer keys = hash && !keys_ ? Allocate(tableCapacity_ * sizeof(Key)) : Buffer{};
        if (counts) {
            counts_ = std::move(counts);
        }
        if (keys) {
            keys_ = std::move(keys);
        }
        detail::static_multiset::ClearKernel<Key><<<detail::static_multiset::BLOCKS, 0, stream>>>(
            hash ? KeysData() : nullptr, CountsData(), tableCapacity_, empty_);
    }
    SizeType ReadTotal(aclrtStream stream)
    {
        Sync(stream);
        uint64_t total = 0;
        Check(aclrtMemcpy(&total, sizeof(total), work_.get() + detail::static_multiset::PARTITIONS * sizeof(uint64_t),
                          sizeof(total), ACL_MEMCPY_DEVICE_TO_HOST),
              "read total");
        return total;
    }
    template <bool Conditional>
    bool InsertInterval(void* keys, SizeType n, SizeType accepted, const uint64_t* totals, const uint64_t* range,
                        aclrtStream stream)
    {
        // The complete input check proves every accepted prefix is a contiguous
        // integer interval. Store its exact count function instead of per-key counts.
        bool interval = totals[0] == n && range[2] == 0 && range[1] - range[0] == n - 1;
        bool firstInterval = !size_ && interval;
        bool repeatedInterval = uniformCount_ && interval && range[0] == directBase_ && accepted == uniformLength_;
        if (firstInterval || repeatedInterval) {
            if constexpr (Conditional) {
                Check(aclrtMemcpyAsync(ordered_.get() + size_ * sizeof(Key), accepted * sizeof(Key), keys,
                                       accepted * sizeof(Key), ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "insert interval");
                Sync(stream);
            }
            if (firstInterval) {
                direct_ = true;
                directBase_ = range[0];
                uniformLength_ = accepted;
                uniformCount_ = 1;
            } else {
                ++uniformCount_;
            }
            size_ += accepted;
            return true;
        }
        return false;
    }
    void PrepareInsertion(const uint64_t* totals, const uint64_t* range, aclrtStream stream)
    {
        if (!size_ && totals[0]) {
            bool direct = range[1] - range[0] < tableCapacity_;
            InitializeStorage(!direct, stream);
            direct_ = direct;
            directBase_ = range[0];
        } else if (direct_ && totals[0] && (range[0] < directBase_ || range[1] - directBase_ >= tableCapacity_)) {
            InitializeStorage(true, stream);
            detail::static_multiset::RebuildKernel<Key><<<detail::static_multiset::BLOCKS, 0, stream>>>(
                ordered_.get(), size_, empty_, KeysData(), CountsData(), tableCapacity_);
            direct_ = false;
            uniformCount_ = uniformLength_ = 0;
        } else if (uniformCount_) {
            InitializeStorage(false, stream);
            detail::static_multiset::MaterializeIntervalKernel<Key>
                <<<detail::static_multiset::BLOCKS, 0, stream>>>(CountsData(), uniformLength_, uniformCount_);
            uniformCount_ = uniformLength_ = 0;
        }
    }
    template <typename Stencil, bool Conditional>
    bool ValidateInsertionInput(void* keys, Stencil* stencil, SizeType n) const
    {
        CheckState();
        if (!n || !keys) {
            return false;
        }
        if constexpr (Conditional) {
            RequirePointer(stencil);
        }
        return true;
    }
    template <typename Stencil, typename Predicate, bool Conditional>
    SizeType FinishInsertion(void* keys, Stencil* stencil, SizeType n, const Predicate& pred, const uint64_t* totals,
                             const uint64_t* range, aclrtStream stream)
    {
        SizeType accepted = std::min(totals[0], capacity_ - size_);
        if (!accepted) {
            return totals[0] + totals[1];
        }
        if (InsertInterval<Conditional>(keys, n, accepted, totals, range, stream)) {
            return totals[0] - accepted + totals[1];
        }
        PrepareInsertion(totals, range, stream);
        detail::static_multiset::InsertKernel<Key, Stencil, Predicate, Conditional>
            <<<detail::static_multiset::BLOCKS, 0, stream>>>(Bytes(keys), Bytes(stencil), n, empty_, KeysData(),
                                                             CountsData(), ordered_.get(), capacity_, tableCapacity_,
                                                             size_, work_.get(), pred, direct_, directBase_);
        Sync(stream);
        size_ += accepted;
        return totals[0] - accepted + totals[1];
    }
    // Keep small-batch preprocessing out of the large-batch insertion frame.
    template <typename Stencil, typename Predicate, bool Conditional>
    __attribute__((noinline)) SizeType InsertSmallImpl(void* keys, Stencil* stencil, SizeType n, Predicate pred,
                                                       aclrtStream stream)
    {
        if (!ValidateInsertionInput<Stencil, Conditional>(keys, stencil, n)) {
            return n;
        }
        detail::static_multiset::SmallSelectKernel<Key, Stencil, Predicate, Conditional><<<1, 0, stream>>>(
            Bytes(keys), Bytes(stencil), n, empty_, work_.get(), pred, ordered_.get(), size_, capacity_ - size_);
        Sync(stream);
        uint64_t metadata[5] = {};
        Check(
            aclrtMemcpy(metadata, sizeof(metadata), work_.get() + detail::static_multiset::RANGE_MIN * sizeof(uint64_t),
                        sizeof(metadata), ACL_MEMCPY_DEVICE_TO_HOST),
            "read small insert statistics");
        auto const* totals = metadata;
        auto const* range = metadata + 2;
        return FinishInsertion<Stencil, Predicate, Conditional>(keys, stencil, n, pred, totals, range, stream);
    }
    template <typename Stencil, typename Predicate, bool Conditional>
    SizeType InsertImpl(void* keys, Stencil* stencil, SizeType n, Predicate pred, aclrtStream stream)
    {
        if (!ValidateInsertionInput<Stencil, Conditional>(keys, stencil, n)) {
            return n;
        }
        detail::static_multiset::SelectKernel<Key, Stencil, Predicate, Conditional>
            <<<detail::static_multiset::BLOCKS, 0, stream>>>(Bytes(keys), Bytes(stencil), n, empty_, work_.get(), pred,
                                                             ordered_.get(), size_, capacity_ - size_);
        detail::static_multiset::ScanBlocksKernel<true><<<detail::static_multiset::BLOCKS, 0, stream>>>(work_.get());
        detail::static_multiset::ScanKernel<true><<<1, 0, stream>>>(work_.get());
        Sync(stream);
        uint64_t totals[2] = {}, range[3] = {};
        Check(aclrtMemcpy(totals, sizeof(totals), work_.get() + detail::static_multiset::PARTITIONS * sizeof(uint64_t),
                          sizeof(totals), ACL_MEMCPY_DEVICE_TO_HOST),
              "read insert counts");
        Check(aclrtMemcpy(range, sizeof(range),
                          work_.get() + (detail::static_multiset::BLOCK_PREFIX + 5 * detail::static_multiset::BLOCKS) *
                                            sizeof(uint64_t),
                          sizeof(range), ACL_MEMCPY_DEVICE_TO_HOST),
              "read key range");
        return FinishInsertion<Stencil, Predicate, Conditional>(keys, stencil, n, pred, totals, range, stream);
    }
    template <int Mode, typename Stencil, typename Predicate, bool Conditional>
    void Query(void* keys, Stencil* stencil, void* output, SizeType n, Predicate pred, aclrtStream stream)
    {
        CheckState();
        if (!n) {
            return;
        }
        RequirePointer(keys);
        RequirePointer(output);
        if constexpr (Conditional) {
            RequirePointer(stencil);
        }
        detail::static_multiset::QueryKernel<Key, Stencil, Predicate, Conditional, Mode><<<QueryBlocks(), 0, stream>>>(
            Bytes(keys), Bytes(stencil), n, empty_, KeysData(), CountsData(), tableCapacity_, Bytes(output), pred,
            direct_, directBase_, uniformCount_, uniformLength_);
        Sync(stream);
    }
    template <bool Outer, typename Equal, typename Hash>
    void ProbeCount(void* keys, SizeType n, Equal equal, Hash hash, void* output, aclrtStream stream)
    {
        CheckState();
        if (!n) {
            return;
        }
        RequirePointer(keys);
        RequirePointer(output);
        detail::static_multiset::ProbeCountKernel<Key, Equal, Hash, Outer><<<QueryBlocks(), 0, stream>>>(
            Bytes(keys), n, empty_, KeysData(), CountsData(), tableCapacity_, Bytes(output), equal, hash, direct_,
            directBase_, uniformCount_, uniformLength_);
        Sync(stream);
    }
    template <bool Ordered>
    SizeType CountImpl(void* keys, SizeType n, aclrtStream stream)
    {
        CheckState();
        if (!n) {
            return 0;
        }
        RequirePointer(keys);
        if (size_ && n > std::numeric_limits<SizeType>::max() / size_) {
            throw std::overflow_error("StaticMultiset: potential count overflow");
        }
        if constexpr (Ordered) {
            detail::static_multiset::RetrieveKernel<Key, false><<<detail::static_multiset::BLOCKS, 0, stream>>>(
                Bytes(keys), n, empty_, KeysData(), CountsData(), tableCapacity_, work_.get(), nullptr, nullptr,
                direct_, directBase_, uniformCount_, uniformLength_);
            detail::static_multiset::ScanBlocksKernel<false>
                <<<detail::static_multiset::BLOCKS, 0, stream>>>(work_.get());
            detail::static_multiset::ScanKernel<false><<<1, 0, stream>>>(work_.get());
        } else {
            uint8_t* total = work_.get() + detail::static_multiset::PARTITIONS * sizeof(uint64_t);
            auto blocks = QueryBlocks();
            constexpr uint32_t WARPS_PER_BLOCK =
                detail::static_multiset::QUERY_THREADS / 32;
            if (blocks * WARPS_PER_BLOCK > detail::static_multiset::PARTITIONS) {
                throw std::runtime_error("Count reduction workspace too small");
            }
            auto partial = work_.get();
            detail::static_multiset::CountKernel<Key>
                <<<blocks, 0, stream>>>(Bytes(keys), n, empty_, KeysData(), CountsData(),
                                       tableCapacity_, partial, direct_, directBase_,
                                       uniformCount_, uniformLength_);
            detail::static_multiset::CountFinalizeKernel<<<1, 0, stream>>>(
                partial, total, blocks * WARPS_PER_BLOCK);
        }
        return ReadTotal(stream);
    }
    SizeType capacity_ = 0, tableCapacity_ = 0, size_ = 0;
    Key empty_;
    bool direct_ = false;
    uint64_t directBase_ = 0;
    uint64_t uniformCount_ = 0, uniformLength_ = 0;
    Buffer counts_, keys_, ordered_, work_;
};
} // namespace aclco
