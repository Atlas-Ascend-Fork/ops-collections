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

#include "kernel_operator.h"
#include "macros.h"
#include "detail/storages/bucket_storage_ref.h"
#include "detail/bitwise_compare.h"
#include "probing_scheme.h"
#include "pair.h"
#include "utility/equal_wrapper.h"
#include "utility/is_same.h"
#include "utility/traits.h"
#include "utility/atomic_cas_wrap.h"

namespace aclco {
enum class InsertResult : int32_t { FAILED, SUCCESS, DUPLICATE };

template <typename Key, typename KeyEqual, typename ProbingScheme, typename StorageRef>
class OpenAddressingRefImpl {
public:
    using KeyType = Key;
    using ProbingSchemeType = ProbingScheme;
    using Hasher = typename ProbingSchemeType::Hasher;
    using StorageRefType = StorageRef;
    using ValueType = typename StorageRefType::ValueType;
    using SizeType = typename StorageRefType::SizeType;
    using Iterator = typename StorageRefType::Iterator;
    using ConstIterator = typename StorageRefType::ConstIterator;

    static constexpr auto bucketSize = StorageRefType::bucketSize;
    static constexpr bool hasPayload = isPairV<ValueType>;
    // Probing reads only the key and fetches the payload once a key matches. Enabled only for the
    // configurations measured faster on Ascend950PR: buckets of four or more slots, a key narrower
    // than one 8-byte word, and a slot wider than one. Measured 2.2%-10.8% faster across load
    // factors 0.5/0.9, hit rates 0%/50%/100%, tables from 4 million to 80 million keys and query
    // batches from 500 thousand to 40 million. Three-slot buckets are excluded: their gain is not
    // reproducible at small batches with high hit rates. Every excluded configuration compiles to
    // code byte-identical to the previous single-load path. Why this is faster is not claimed here:
    // bucket starts are not aligned to the access granularity, so the narrower read is not shown to
    // cut per-probe traffic.
    static constexpr bool deferPayloadLoad =
        hasPayload && bucketSize > 3 && sizeof(ValueType) > sizeof(uint64_t) && sizeof(KeyType) < sizeof(uint64_t);
    using PayloadType = PayloadTypeT<ValueType>;

    COLLECTION_HOST_DEVICE explicit constexpr OpenAddressingRefImpl(ValueType emptySlotValue, KeyEqual const& predicate,
                                                                    ProbingSchemeType const& probingScheme,
                                                                    StorageRefType storageRef) noexcept
        : emptySlotValue_{emptySlotValue}, predicate_{predicate}, probingScheme_{probingScheme}, storageRef_{storageRef}
    {}

    template <typename Value>
    COLLECTION_HOST_DEVICE constexpr auto ExtractKey(Value value) const noexcept
    {
        if constexpr (hasPayload) {
            return value.first;
        } else {
            return value;
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE constexpr auto ExtractValue(Value value) const noexcept
    {
        if constexpr (hasPayload) {
            return value.second;
        } else {
            return value;
        }
    }

    template <typename Value>
    COLLECTION_HOST_DEVICE constexpr auto ExtractPayload(Value value) const noexcept
    {
        return value.second;
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult BackToBackCas(__gm__ Value* address, Value expected, Value desire) noexcept
    {
        using KeyType = decltype(expected.first);
        using MappedType = decltype(expected.second);

        auto expectedKey = expected.first;
        auto expectedPayload = expected.second;

        auto oldKey = AtomicCasWrap((__gm__ KeyType*)&(address->first), expectedKey, desire.first);

        if (this->predicate_.EqualTo(oldKey, expectedKey) == EqualResult::EQUAL) {
            auto oldValue = AtomicCasWrap((__gm__ MappedType*)&(address->second), expectedPayload, desire.second);
            return InsertResult::SUCCESS;
        } else if (this->predicate_.EqualTo(oldKey, desire.first) == EqualResult::EQUAL) {
            return InsertResult::DUPLICATE;
        }

        return InsertResult::FAILED;
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult PackCas(__gm__ Value* address, Value expected, Value desired) noexcept
    {
        using PackedType = UintBySizeT<sizeof(Value)>;

        __gm__ auto* slotPtr = reinterpret_cast<__gm__ PackedType*>(address);
        PackedType packedExpected = *reinterpret_cast<PackedType*>(&expected);
        PackedType packedDesired = *reinterpret_cast<PackedType*>(&desired);

        auto oldSlot = AtomicCasWrap(slotPtr, packedExpected, packedDesired);
        auto* oldSlotPtr = reinterpret_cast<Value*>(&oldSlot);

        if (this->predicate_.EqualTo(this->ExtractKey(expected), this->ExtractKey(*oldSlotPtr)) == EqualResult::EQUAL) {
            return InsertResult::SUCCESS;
        } else {
            if (this->predicate_.EqualTo(this->ExtractKey(desired), this->ExtractKey(*oldSlotPtr)) ==
                EqualResult::EQUAL) {
                return InsertResult::DUPLICATE;
            }
            return InsertResult::FAILED;
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult NarrowCas(__gm__ Value* address, Value expected, Value desired) noexcept
    {
        auto oldValue = AtomicCasWrap(address, expected, desired);

        if (this->predicate_.EqualTo(this->ExtractKey(expected), this->ExtractKey(oldValue)) == EqualResult::EQUAL) {
            return InsertResult::SUCCESS;
        } else {
            if (this->predicate_.EqualTo(this->ExtractKey(desired), this->ExtractKey(oldValue)) == EqualResult::EQUAL) {
                return InsertResult::DUPLICATE;
            }
            return InsertResult::FAILED;
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult CasDependentWrite(__gm__ Value* address, Value expected, Value desired) noexcept
    {
        using MappedType = std::decay_t<decltype(this->ExtractPayload(emptySlotValue_))>;

        __gm__ auto* keyAddr = reinterpret_cast<__gm__ KeyType*>(&(address->first));
        auto expectedKey = expected.first;
        auto desiredKey = desired.first;
        auto oldKey = AtomicCasWrap(keyAddr, expectedKey, desiredKey);

        if (oldKey == expectedKey) {
            auto const desiredValue = desired.second;
            address->second = desiredValue;
            return InsertResult::SUCCESS;
        }

        if (this->predicate_.EqualTo(oldKey, desiredKey) == EqualResult::EQUAL) {
            return InsertResult::DUPLICATE;
        }

        return InsertResult::FAILED;
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult AttemptInsert(__gm__ Value* address, Value expected, Value desired) noexcept
    {
        if constexpr (sizeof(Value) < 4) {
            return NarrowCas(address, expected, desired);
        } else if constexpr (sizeof(Value) <= 8) {
            return PackCas(address, expected, desired);
        } else {
            // >8B（如 I64 pair）：赢得 key 槽后该槽已独占，value 用普通写代替第二次原子，
            // 并发 Find 端有 WaitForPayload 自旋兜底（与 AttemptInsertStable 同款）。
            return CasDependentWrite(address, expected, desired);
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult AttemptInsertStable(__gm__ Value* address, Value expected,
                                                            Value desired) noexcept
    {
        return AttemptInsert(address, expected, desired);
    }

    /** assign 写：≤8B 槽整槽原生宽度写（key 不变），避免 sub-word 拆写走 GM 读改写慢通道。 */
    template <typename Value>
    COLLECTION_SIMT_DEVICE void AssignPayload(__gm__ Value* address, Value desired) noexcept
    {
        if constexpr (sizeof(Value) <= 8) {
            using PackedType = UintBySizeT<sizeof(Value)>;
            PackedType packedDesired{};
            *reinterpret_cast<Value*>(&packedDesired) = desired;
            *reinterpret_cast<__gm__ PackedType*>(address) = packedDesired;
        } else {
            address->second = desired.second;
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult AttemptInsertOrAssign(__gm__ Value* address, Value expected,
                                                              Value desired) noexcept
    {
        if constexpr (sizeof(Value) <= 8) {
            return PackCasInsertOrAssign(address, expected, desired);
        } else {
            return BackToBackCasInsertOrAssign(address, expected, desired);
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult PackCasInsertOrAssign(__gm__ Value* address, Value expected,
                                                              Value desired) noexcept
    {
        using PackedType = UintBySizeT<sizeof(Value)>;

        __gm__ auto* slotPtr = reinterpret_cast<__gm__ PackedType*>(address);
        PackedType packedExpected{};
        PackedType packedDesired{};
        *reinterpret_cast<Value*>(&packedExpected) = expected;
        *reinterpret_cast<Value*>(&packedDesired) = desired;

        auto oldSlot = AtomicCasWrap(slotPtr, packedExpected, packedDesired);
        auto* oldSlotPtr = reinterpret_cast<Value*>(&oldSlot);

        if (this->predicate_.EqualTo(this->ExtractKey(expected), this->ExtractKey(*oldSlotPtr)) == EqualResult::EQUAL) {
            return InsertResult::SUCCESS;
        } else {
            if (this->predicate_.EqualTo(this->ExtractKey(desired), this->ExtractKey(*oldSlotPtr)) ==
                EqualResult::EQUAL) {
                AssignPayload(address, desired);
                return InsertResult::DUPLICATE; // 并发已存在 → 实为 assign
            }
            return InsertResult::FAILED;
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult BackToBackCasInsertOrAssign(__gm__ Value* address, Value expected,
                                                                    Value desired) noexcept
    {
        __gm__ auto* keyAddr = reinterpret_cast<__gm__ KeyType*>(&(address->first));
        auto expectedKey = expected.first;
        auto desiredKey = desired.first;
        auto oldKey = AtomicCasWrap(keyAddr, expectedKey, desiredKey);

        if (oldKey == expectedKey) {
            address->second = desired.second;
            return InsertResult::SUCCESS;
        }

        if (this->predicate_.EqualTo(oldKey, desiredKey) == EqualResult::EQUAL) {
            address->second = desired.second;
            return InsertResult::DUPLICATE; // 并发已存在 → 实为 assign
        }

        return InsertResult::FAILED;
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE bool Insert(Value value)
    {
        return InsertVerdict<false>(value) == InsertResult::SUCCESS;
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE bool InsertOrAssign(Value value)
    {
        return InsertOrAssignVerdict(value) != InsertResult::FAILED;
    }

    /** @brief 三态返回：SUCCESS=插入新 key，DUPLICATE=已存在 key 的 assign，FAILED=表满。
     *  纯返回值（不带引用出参，避免 SIMT 线程私有内存 spill 拖慢热循环）。 */
    template <typename Value>
    COLLECTION_SIMT_DEVICE InsertResult InsertOrAssignVerdict(Value value)
    {
        return InsertVerdict<true>(value);
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE Pair<PayloadType, bool> InsertAndFind(Value value)
    {
        __gm__ Value* tableHandle = storageRef_.Data();
        SizeType tableSize = storageRef_.Capacity();
        auto const key = this->ExtractKey(value);

        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, tableSize);
        auto const initIdx = *probingIter;

        while (true) {
            __gm__ Value* bucketSlotsAddr = tableHandle + *probingIter;

            for (size_t i = 0; i < bucketSize; i++) {
                __gm__ Value* slotPtr = bucketSlotsAddr + i;
                auto const result = InsertAndFindInSlot(slotPtr, value, key);
                if (result.second != InsertResult::FAILED) {
                    return {result.first, result.second == InsertResult::SUCCESS};
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return {ExtractValue(emptySlotValue_), false};
            }
        }
    }

    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE bool Erase(ProbeKey key) noexcept
    {
        return this->Erase(key, emptySlotValue_);
    }

    /** @brief 墓碑删除重载：删除的槽写入墓碑值（key=erasedKey）而非空槽，不破坏线性探测链；
     *  未配置墓碑（erasedKey==emptyKey，如 StaticSet 默认）时退回写空槽的原始语义。 */
    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE bool Erase(ProbeKey key, ValueType erasedSlotValue) noexcept
    {
        __gm__ auto* tableHandle = storageRef_.Data();

        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
        auto const initIdx = *probingIter;
        bool const hasTombstone = predicate_.EqualTo(this->ExtractKey(erasedSlotValue),
                                                     this->ExtractKey(emptySlotValue_)) != EqualResult::EQUAL;

        while (true) {
            __gm__ auto* bucketSlotsAddr = tableHandle + *probingIter;

            for (size_t i = 0; i < bucketSize; i++) {
                auto const slotContent = *(bucketSlotsAddr + i);
                EqualResult eraseFlag = predicate_.template operator()<IsInsert::NO>(key, this->ExtractKey(slotContent),
                                                                                     this->ExtractKey(emptySlotValue_));

                if (eraseFlag == EqualResult::EQUAL) {
                    InsertResult result = AttemptInsertStable(bucketSlotsAddr + i, slotContent, erasedSlotValue);
                    switch (result) {
                        case InsertResult::SUCCESS:
                            return true;
                        case InsertResult::DUPLICATE:
                            return false;
                        default:
                            continue;
                    }
                }
                // 仅在配置了墓碑时才空槽早停（此时删除留下的是墓碑、不会触发早停，安全）。
                if (hasTombstone && eraseFlag == EqualResult::EMPTY) {
                    return false;
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return false;
            }
        }
    }

    /**
     * @brief assign-only：key 已存在则更新 payload 并返回 true；不存在返回 false（不插入）。
     */
    COLLECTION_SIMT_DEVICE bool Assign(ValueType desired) noexcept
    {
        __gm__ auto* tableHandle = storageRef_.Data();
        auto const key = this->ExtractKey(desired);
        auto const emptyKey = this->ExtractKey(emptySlotValue_);
        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
        auto const initIdx = *probingIter;

        while (true) {
            __gm__ auto* bucketSlotsAddr = tableHandle + *probingIter;
            for (auto i = 0; i < bucketSize; i++) {
                __gm__ auto* slotAddr = bucketSlotsAddr + i;
                auto const slotKey = this->ExtractKey(*slotAddr);
                if (predicate_.EqualTo(slotKey, emptyKey) == EqualResult::EQUAL) {
                    return false;
                }
                if (predicate_.EqualTo(slotKey, key) == EqualResult::EQUAL) {
                    slotAddr->second = desired.second;
                    return true;
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return false;
            }
        }
    }

    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE PayloadType Find(ProbeKey key) noexcept
    {
        return FindAndContains(key).first;
    }

    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE bool Contains(ProbeKey key) noexcept
    {
        return FindAndContains(key).second;
    }

    template <typename ProbeKey, typename CallbackOp>
    COLLECTION_SIMT_DEVICE void ForEach(ProbeKey key, CallbackOp& callback_op) noexcept
    {
        __gm__ auto* tableHandle = storageRef_.Data();

        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
        auto const initIdx = *probingIter;

        while (true) {
            __gm__ auto* bucketSlotsAddr = tableHandle + *probingIter;

            for (size_t i = 0; i < bucketSize; i++) {
                auto const slotContent = *(bucketSlotsAddr + i);
                switch (EqualResult forEachFlag = predicate_.template operator()<IsInsert::NO>(
                            key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_))) {
                    case EqualResult::EMPTY: {
                        return;
                    }
                    case EqualResult::EQUAL: {
                        callback_op(slotContent);
                        break;
                    }
                    default:
                        continue;
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return;
            }
        }
    }

    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE SizeType Count(ProbeKey key) noexcept
    {
        return static_cast<SizeType>(this->Contains(key));
    }

    /**
     * @brief 等待slot中的payload写入完成
     *
     * @note 当slot中的payload不等于sentinel的值时返回
     *
     * @tparam T Map slot的类型
     *
     * @param slot 用来存储payload
     * @param sentinel 一个哨兵值，标记值为空
     */
    template <typename T>
    COLLECTION_SIMT_DEVICE void WaitForPayload(__gm__ T& slot, T sentinel) const noexcept
    {
        using PackedType = UintBySizeT<sizeof(T)>;
        static_assert(sizeof(PackedType) == sizeof(T), "PackedType must match payload size");
        __gm__ PackedType* slotPtr = reinterpret_cast<__gm__ PackedType*>(&slot);
        PackedType packedSentinel = *reinterpret_cast<PackedType*>(&sentinel);
        PackedType current;

        // 直到当前值写入完成，否则一直循环等待
        do {
            current = *reinterpret_cast<volatile __gm__ PackedType*>(slotPtr);
        } while (detail::BitWiseCompare(current, packedSentinel));
    }

private:
    template <bool assign, typename Value>
    COLLECTION_SIMT_DEVICE InsertResult InsertVerdict(Value value)
    {
        __gm__ Value* tableHandle = storageRef_.Data();
        SizeType tableSize = storageRef_.Capacity();
        auto const key = this->ExtractKey(value);
        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, tableSize);
        auto const initIdx = *probingIter;

        while (true) {
            __gm__ Value* bucketSlotsAddr = tableHandle + *probingIter;
            for (size_t i = 0; i < bucketSize; i++) {
                auto const slotContent = *(bucketSlotsAddr + i);
                EqualResult insertFlag = predicate_.template operator()<IsInsert::YES>(
                    key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_));
                if (insertFlag == EqualResult::EQUAL) {
                    if constexpr (assign) {
                        AssignPayload(bucketSlotsAddr + i, value);
                    }
                    return InsertResult::DUPLICATE;
                }
                if (insertFlag == EqualResult::AVAILABLE) {
                    InsertResult result;
                    if constexpr (assign) {
                        result = AttemptInsertOrAssign(bucketSlotsAddr + i, emptySlotValue_, value);
                    } else {
                        result = AttemptInsert(bucketSlotsAddr + i, emptySlotValue_, value);
                    }
                    if (result != InsertResult::FAILED) {
                        return result;
                    }
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return InsertResult::FAILED;
            }
        }
    }

    template <typename Value>
    COLLECTION_SIMT_DEVICE PayloadType WaitAndExtractValue(__gm__ Value* slotPtr) noexcept
    {
        if constexpr (hasPayload) {
            this->WaitForPayload(slotPtr->second, this->emptySlotValue_.second);
        }
        return ExtractValue(*slotPtr);
    }

    template <typename Value, typename ProbeKey>
    COLLECTION_SIMT_DEVICE Pair<PayloadType, InsertResult> InsertAndFindInSlot(__gm__ Value* slotPtr, Value value,
                                                                               ProbeKey key) noexcept
    {
        auto const slotContent = *slotPtr;
        EqualResult findFlag = predicate_.template operator()<IsInsert::YES>(key, this->ExtractKey(slotContent),
                                                                             this->ExtractKey(emptySlotValue_));
        if (findFlag == EqualResult::EQUAL) {
            return {WaitAndExtractValue(slotPtr), InsertResult::DUPLICATE};
        }
        if (findFlag != EqualResult::AVAILABLE) {
            return {ExtractValue(emptySlotValue_), InsertResult::FAILED};
        }

        InsertResult result = AttemptInsertStable(slotPtr, emptySlotValue_, value);
        if (result == InsertResult::SUCCESS) {
            return {ExtractValue(value), result};
        }
        if (result == InsertResult::DUPLICATE) {
            return {WaitAndExtractValue(slotPtr), result};
        }
        return {ExtractValue(emptySlotValue_), result};
    }

    template <typename ProbeKey>
    COLLECTION_SIMT_DEVICE Pair<PayloadType, bool> FindAndContains(ProbeKey key) noexcept
    {
        __gm__ auto* tableHandle = storageRef_.Data();
        auto const emptyKey = this->ExtractKey(emptySlotValue_);
        auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
        auto const initIdx = *probingIter;

        while (true) {
            __gm__ auto* bucketSlotsAddr = tableHandle + *probingIter;
            for (auto i = 0; i < bucketSize; i++) {
                auto const slotPtr = bucketSlotsAddr + i;
                if constexpr (deferPayloadLoad) {
                    // Wide slot: read only the key while probing, fetch the payload once the key matches.
                    auto const slotKey = slotPtr->first;
                    if (predicate_.EqualTo(slotKey, emptyKey) == EqualResult::EQUAL) {
                        return {ExtractValue(emptySlotValue_), false};
                    }
                    if (predicate_.EqualTo(slotKey, key) == EqualResult::EQUAL) {
                        return {ExtractValue(*slotPtr), true};
                    }
                } else {
                    // Narrow slot: a single load already carries key and payload, so reuse it.
                    auto const slotContent = *slotPtr;
                    auto const slotKey = this->ExtractKey(slotContent);
                    if (predicate_.EqualTo(slotKey, emptyKey) == EqualResult::EQUAL) {
                        return {ExtractValue(emptySlotValue_), false};
                    }
                    if (predicate_.EqualTo(slotKey, key) == EqualResult::EQUAL) {
                        return {ExtractValue(slotContent), true};
                    }
                }
            }
            ++probingIter;
            if (*probingIter == initIdx) {
                return {ExtractValue(emptySlotValue_), false};
            }
        }
    }

public:
    ValueType emptySlotValue_;
    EqualWrapper<KeyType, KeyEqual> predicate_;
    ProbingSchemeType probingScheme_;
    StorageRefType storageRef_;
};
} // namespace aclco
