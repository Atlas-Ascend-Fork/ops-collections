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

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "detail/dynamic_map/kernels.h"       // op_kernel: MergeFind / MergeContains 合并核
#include "detail/storages/arg_storage.h"      // third_party: 计数器
#include "static_map.h"                       // third_party: ops-collections StaticMap
#include "tiling/platform/platform_ascendc.h" // third_party: aivCoreNum

namespace aclco {
template <class Key, class T, class Extent = aclco::Extent<size_t>, class KeyEqual = aclco::EqualTo<Key>,
          class ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>,
          class Storage = aclco::Storage<defaultMapBucketSize>>
class DynamicMap {
public:
    using MapType = aclco::StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>;
    using SizeType = typename MapType::SizeType;
    using KeyType = Key;
    using ValueType = T;

    /** @brief 插入模式：kExactDedup 精确去重（默认），kAppendUnique 唯一键追加。 */
    enum class InsertMode { kExactDedup, kAppendUnique };

    static constexpr SizeType DEFAULT_MIN_INSERT = 10000; // 默认最小插入规模
    static constexpr float DEFAULT_MAX_LOAD = 0.60f;      // 默认最大负载因子
    static constexpr SizeType GROWTH_FACTOR = 2;
    static constexpr SizeType INSERT_SUBBATCH = 800000; // 活跃子表内异步插入的子批大小

    DynamicMap(const DynamicMap&) = delete;
    DynamicMap& operator=(const DynamicMap&) = delete;
    DynamicMap(DynamicMap&&) = default;
    DynamicMap& operator=(DynamicMap&&) = default;
    ~DynamicMap() = default;

    DynamicMap(Extent initialCapacity, Key emptyKey, T emptyValue, Key erasedKey, KeyEqual const& pred = {},
               ProbingScheme const& ps = {}, Storage storage = {}, aclrtStream stream = nullptr)
        : size_(0),
          minInsertSize_(DEFAULT_MIN_INSERT),
          maxLoadFactor_(DEFAULT_MAX_LOAD),
          emptyKey_(emptyKey),
          emptyValue_(emptyValue),
          erasedKey_(erasedKey),
          pred_(pred),
          probingScheme_(ps),
          storage_(storage)
    {
        if (emptyKey == erasedKey) {
            throw std::invalid_argument("DynamicMap: emptyKey and erasedKey must differ");
        }
        submaps_.push_back(std::make_unique<MapType>(initialCapacity, emptyKey, emptyValue, pred, ps, storage, stream));
        submaps_.back()->SetErasedKey(erasedKey_, stream); // 启用墓碑删除
        submapSize_.push_back(0);
        totalCapacity_ = submaps_.back()->Capacity();
        nextCapacity_ = static_cast<SizeType>(submaps_.back()->Capacity()) * GROWTH_FACTOR;
        // 常驻：emptyValue 的 device 副本（合并核需要）
        void* p = nullptr;
        CheckRet(aclrtMalloc(&p, sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST), "DynamicMap::ctor::aclrtMalloc");
        CheckRet(aclrtMemcpy(p, sizeof(T), &emptyValue_, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE),
                 "DynamicMap::ctor::aclrtMemcpy");
        dEmptyValue_.reset(p);
        aivCoreNum_ = static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
        if (aivCoreNum_ == 0)
            aivCoreNum_ = 1;
        // 预建用后台流 + 事件（创建失败不致命，退化为同步增长）
        aclrtStream prepS = nullptr;
        aclrtEvent prepE = nullptr;
        if (aclrtCreateStream(&prepS) == ACL_SUCCESS && aclrtCreateEvent(&prepE) == ACL_SUCCESS) {
            prepStreamHolder_.reset(prepS);
            prepEventHolder_.reset(prepE);
            prepStream_ = prepS;
            prepEvent_ = prepE;
        } else if (prepS != nullptr) {
            aclrtDestroyStream(prepS);
        }
    }

    /** @brief 当前元素个数。 */
    SizeType Size() const noexcept { return size_; }
    SizeType Capacity() const noexcept { return totalCapacity_; } // 全部子表容量之和
    size_t NumSubmaps() const noexcept { return submaps_.size(); }

    /** @brief 预留容量：保证总容量可容纳 n 个元素。 */
    void Reserve(SizeType n, aclrtStream stream)
    {
        SizeType remaining = n;
        size_t idx = 0;
        while (remaining > 0) {
            SizeType cap;
            if (idx < submaps_.size()) {
                cap = submaps_[idx]->Capacity();
            } else {
                cap = nextCapacity_;
                submaps_.push_back(std::make_unique<MapType>(Extent{cap}, emptyKey_, emptyValue_, pred_, probingScheme_,
                                                             storage_, stream));
                submaps_.back()->SetErasedKey(erasedKey_, stream); // 启用墓碑删除
                submapSize_.push_back(0);
                totalCapacity_ += submaps_.back()->Capacity();
                nextCapacity_ = cap * GROWTH_FACTOR;
            }
            SizeType usable = Usable(cap);
            if (usable >= remaining)
                break;
            remaining -= usable;
            ++idx;
        }
    }

    /** @brief 同步插入一批键值对，返回新插入成功的个数。 */
    SizeType Insert(void* values, Extent valueNum, aclrtStream stream)
    {
        return Insert(values, valueNum, stream, InsertMode::kExactDedup);
    }

    /** @brief 同步插入（指定插入模式），返回新插入成功的个数。 */
    SizeType Insert(void* values, Extent valueNum, aclrtStream stream, InsertMode mode)
    {
        SizeType n = static_cast<SizeType>(valueNum);
        if (n == 0 || values == nullptr)
            return 0;
        EnsureActiveRoom(n, stream); // 一次性增长到可容纳全部 n
        size_t a = ActiveIdx();

        bool appendUnique = (mode == InsertMode::kAppendUnique);
        bool fast = appendUnique || !HasNonEmptyBefore(a);
        if (fast) {
            if (appendUnique) {
                // 唯一键追加：子批异步插入到单一活跃子表（唯一键失败数为 0，乐观计数即精确），不逐批同步。
                for (SizeType off = 0; off < n; off += INSERT_SUBBATCH) {
                    SizeType cur = (n - off < INSERT_SUBBATCH) ? (n - off) : INSERT_SUBBATCH;
                    submaps_[a]->InsertAsync((uint8_t*)values + static_cast<size_t>(off) * sizeof(Pair<Key, T>),
                                             Extent{cur}, stream);
                }
                submapSize_[a] += n;
                size_ += n;
                return n;
            }
            // 精确快路径：单一活跃子表同步插入，返回 n−失败（含批内重复键、表满）。
            SizeType failed = submaps_[a]->Insert(values, Extent{n}, stream);
            SizeType ins = (n >= failed) ? (n - failed) : 0;
            submapSize_[a] += ins;
            size_ += ins;
            return ins;
        }
        // 慢路径：存在更早非空子表（如对已增长容器重插旧键），跨子表去重保持精确语义。
        SizeType total = 0, offset = 0;
        while (offset < n) {
            SizeType room = ActiveRoom();
            if (room == 0) {
                GrowOneSubmap(n - offset, stream);
                continue;
            }
            if (room < (n - offset) * 2)
                StartPrepareNext(n - offset); // 快满：后台流预建下一子表
            SizeType chunk = (n - offset < room) ? (n - offset) : room;
            total += InsertChunk((uint8_t*)values + static_cast<size_t>(offset) * sizeof(Pair<Key, T>), chunk, stream);
            offset += chunk;
        }
        return total;
    }

    /** @brief 异步插入（乐观计数，单子表快路径）：写入活跃子表的 StaticMap.InsertAsync，免每批全流同步。 */
    SizeType InsertAsync(void* values, Extent valueNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(valueNum);
        if (n == 0 || values == nullptr)
            return 0;
        EnsureActiveRoom(n, stream);
        if (ActiveRoom() < n * 2)
            StartPrepareNext(n); // 快满：后台流预建下一子表
        size_t a = ActiveIdx();
        submaps_[a]->InsertAsync(values, Extent{n}, stream);
        submapSize_[a] += n;
        size_ += n;
        return n;
    }

    /** @brief 同步插入或更新键值对，返回新插入的个数（assign 不计入）。 */
    SizeType InsertOrAssign(void* values, Extent valueNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(valueNum);
        if (n == 0 || values == nullptr)
            return 0;
        EnsureActiveRoom(n, stream); // 一次性增长到可容纳全部 n
        SizeType total = 0, offset = 0;
        while (offset < n) {
            SizeType room = ActiveRoom();
            if (room == 0) {
                GrowOneSubmap(n - offset, stream);
                continue;
            }
            if (room < (n - offset) * 2)
                StartPrepareNext(n - offset);
            SizeType chunk = (n - offset < room) ? (n - offset) : room;
            total += InsertOrAssignChunk((uint8_t*)values + static_cast<size_t>(offset) * sizeof(Pair<Key, T>), chunk,
                                         stream);
            offset += chunk;
        }
        return total;
    }

    /** @brief 查找：对各非空子表查询，device 合并核取唯一命中（key 至多在一个子表）。
     *  子表查询与合并核在同一 stream 上连续提交，循环外统一同步一次，返回时结果已就绪。 */
    void Find(void* keys, void* outputValues, Extent keyNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(keyNum);
        if (n == 0 || keys == nullptr || outputValues == nullptr)
            return;
        // 基线 = 第一个非空子表；全空时退化为 submap[0]（填 emptyValue）。
        size_t base = FirstNonEmpty();
        if (base == submaps_.size()) {
            submaps_[0]->Find(keys, outputValues, n, stream);
            return;
        }
        // 子表查询用 Async：同一 stream 上的顺序已保证与合并核的依赖（dScratch_ 复用、
        // outputValues 累加合并均按 stream 顺序执行），逐表同步只会让 Host 在循环里阻塞 S 次。
        submaps_[base]->FindAsync(keys, outputValues, n, stream);
        if (HasMoreNonEmptyAfter(base)) {
            EnsureScratch(static_cast<size_t>(n) * sizeof(T));
            for (size_t s = base + 1; s < submaps_.size(); ++s) {
                if (submapSize_[s] == 0)
                    continue;
                submaps_[s]->FindAsync(keys, dScratch_.get(), n, stream);
                aclco::MergeFind<T><<<aivCoreNum_, 0, stream>>>((uint8_t*)outputValues, (uint8_t*)dScratch_.get(),
                                                                (uint8_t*)dEmptyValue_.get(), static_cast<uint32_t>(n));
            }
        }
        // 统一完成边界：一次等待覆盖全部子表查询与合并核（S 次逐表等待 → 1 次）。
        CheckRet(aclrtSynchronizeStream(stream), "DynamicMap::Find::aclrtSynchronizeStream");
    }

    /** @brief 是否包含：输出 bool（uint8）。device 合并核做按位 OR。
     *  子表查询与合并核在同一 stream 上连续提交，循环外统一同步一次，返回时结果已就绪。 */
    void Contains(void* keys, void* outputValues, Extent keyNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(keyNum);
        if (n == 0 || keys == nullptr || outputValues == nullptr)
            return;
        size_t base = FirstNonEmpty();
        if (base == submaps_.size()) {
            submaps_[0]->Contains(keys, outputValues, n, stream);
            return;
        }
        // 与 Find 相同：子表 ContainsAsync + 合并核同流连续提交，循环外统一同步一次。
        submaps_[base]->ContainsAsync(keys, outputValues, n, stream);
        if (HasMoreNonEmptyAfter(base)) {
            EnsureScratch(static_cast<size_t>(n)); // uint8/元素
            for (size_t s = base + 1; s < submaps_.size(); ++s) {
                if (submapSize_[s] == 0)
                    continue;
                submaps_[s]->ContainsAsync(keys, dScratch_.get(), n, stream);
                aclco::MergeContains<><<<aivCoreNum_, 0, stream>>>((uint8_t*)outputValues, (uint8_t*)dScratch_.get(),
                                                                   static_cast<uint32_t>(n));
            }
        }
        CheckRet(aclrtSynchronizeStream(stream), "DynamicMap::Contains::aclrtSynchronizeStream");
    }

    /** @brief 删除：key 至多在一个子表，对所有非空子表执行 Erase，累加成功数。 */
    SizeType Erase(void* keys, Extent keyNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(keyNum);
        if (n == 0 || keys == nullptr)
            return 0;
        SizeType total = 0;
        for (size_t s = 0; s < submaps_.size(); ++s) {
            if (submapSize_[s] == 0)
                continue;                                          // 跳过空子表
            SizeType failed = submaps_[s]->Erase(keys, n, stream); // 返回失败(未删)数
            SizeType del = n - failed;                             // 该表实际删除数
            submapSize_[s] -= (submapSize_[s] >= del) ? del : submapSize_[s];
            total += del;
        }
        size_ -= (size_ >= total) ? total : size_;
        return total;
    }

    /** @brief 异步删除（性能路径）：对各非空子表走 StaticMap.EraseAsync。不返回精确删除数、不即时更新 size_。 */
    void EraseAsync(void* keys, Extent keyNum, aclrtStream stream)
    {
        SizeType n = static_cast<SizeType>(keyNum);
        if (n == 0 || keys == nullptr)
            return;
        for (size_t s = 0; s < submaps_.size(); ++s) {
            if (submapSize_[s] == 0)
                continue; // 跳过空子表
            submaps_[s]->EraseAsync(keys, n, stream);
        }
    }

    void Clear(aclrtStream stream)
    {
        if (nextSubmap_ && prepStream_ != nullptr) // 预建子表可能仍在后台流构建，先同步再释放
        {
            CheckRet(aclrtSynchronizeStream(prepStream_), "DynamicMap::Clear::aclrtSynchronizeStream");
        }
        nextSubmap_.reset();
        nextSubmapCap_ = 0;
        for (auto& sm : submaps_)
            sm->Clear(stream);
        for (auto& s : submapSize_)
            s = 0;
        size_ = 0;
    }

private:
    struct AclFreeDeleter {
        void operator()(void* p) const noexcept
        {
            if (p)
                aclrtFree(p);
        }
    };

    SizeType Usable(SizeType cap) const
    {
        SizeType u = static_cast<SizeType>(std::floor(maxLoadFactor_ * static_cast<double>(cap)));
        return (u > minInsertSize_) ? (u - minInsertSize_) : 0;
    }
    size_t ActiveIdx() const { return submaps_.size() - 1; }

    size_t FirstNonEmpty() const
    {
        for (size_t s = 0; s < submaps_.size(); ++s)
            if (submapSize_[s] != 0)
                return s;
        return submaps_.size();
    }
    bool HasNonEmptyBefore(size_t a) const
    {
        for (size_t s = 0; s < a; ++s)
            if (submapSize_[s] != 0)
                return true;
        return false;
    }

    // 去重 scratch 布局：keys[n] | mask[n] | tmp[n]，64B 对齐。
    uint8_t* Keys() const { return (uint8_t*)dDedup_.get(); }
    uint8_t* Mask() const { return Keys() + keysBytes_; }
    uint8_t* Tmp() const { return Mask() + maskBytes_; }

    void EnsureDedupScratch(SizeType n)
    {
        size_t keysBytes = ((static_cast<size_t>(n) * sizeof(Key) + 63) / 64) * 64;
        size_t maskBytes = ((static_cast<size_t>(n) + 63) / 64) * 64;
        size_t bytes = keysBytes + maskBytes * 2;
        if (dedupBytes_ >= bytes && dDedup_) {
            keysBytes_ = keysBytes;
            maskBytes_ = maskBytes;
            return;
        }
        void* p = nullptr;
        CheckRet(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "DynamicMap::EnsureDedupScratch::aclrtMalloc");
        dDedup_.reset(p);
        dedupBytes_ = bytes;
        keysBytes_ = keysBytes;
        maskBytes_ = maskBytes;
    }

    /** @brief mask[i] = key i 是否存在于任一非空旧子表（不含活跃子表）。调用前提：至少一个旧子表非空。 */
    void BuildContainsMask(void* pairs, SizeType n, aclrtStream stream)
    {
        EnsureDedupScratch(n);
        aclco::ExtractKeys<Key, Pair<Key, T>>
            <<<aivCoreNum_, 0, stream>>>(Keys(), (uint8_t*)pairs, static_cast<uint32_t>(n));
        bool first = true;
        size_t a = ActiveIdx();
        for (size_t s = 0; s < a; ++s) {
            if (submapSize_[s] == 0)
                continue;
            if (first) {
                submaps_[s]->ContainsAsync(Keys(), Mask(), Extent{n}, stream); // 首个直接写 mask
                first = false;
            } else {
                submaps_[s]->ContainsAsync(Keys(), Tmp(), Extent{n}, stream);
                aclco::MergeContains<><<<aivCoreNum_, 0, stream>>>(Mask(), Tmp(), static_cast<uint32_t>(n));
            }
        }
    }
    bool HasMoreNonEmptyAfter(size_t base) const
    {
        for (size_t s = base + 1; s < submaps_.size(); ++s)
            if (submapSize_[s] != 0)
                return true;
        return false;
    }

    void EnsureScratch(size_t bytes)
    {
        if (scratchBytes_ >= bytes && dScratch_)
            return;
        void* p = nullptr;
        CheckRet(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "DynamicMap::EnsureScratch::aclrtMalloc");
        dScratch_.reset(p);
        scratchBytes_ = bytes;
    }

    /** @brief 活跃子表剩余可插入量。 */
    SizeType ActiveRoom() const
    {
        size_t a = ActiveIdx();
        SizeType usable = Usable(submaps_[a]->Capacity());
        return (usable > submapSize_[a]) ? (usable - submapSize_[a]) : 0;
    }

    /** @brief 后台流上预建下一子表（构造含全表 Clear，与插入流重叠，消增长停顿）。 */
    void StartPrepareNext(SizeType minUsable)
    {
        if (nextSubmap_ || prepStream_ == nullptr)
            return;
        SizeType newCap = nextCapacity_;
        while (Usable(newCap) < GrowTarget(minUsable))
            newCap *= GROWTH_FACTOR;
        nextSubmap_ = std::make_unique<MapType>(Extent{newCap}, emptyKey_, emptyValue_, pred_, probingScheme_, storage_,
                                                prepStream_);
        nextSubmap_->SetErasedKey(erasedKey_, prepStream_); // 启用墓碑删除
        nextSubmapCap_ = newCap;
        CheckRet(aclrtRecordEvent(prepEvent_, prepStream_), "DynamicMap::StartPrepareNext::aclrtRecordEvent");
    }

    /** @brief 激活预建子表（插入流等待预建完成事件），无预建则同步建。 */
    void ActivateNext(SizeType minUsable, aclrtStream stream)
    {
        if (nextSubmap_ && Usable(nextSubmapCap_) >= minUsable) {
            CheckRet(aclrtStreamWaitEvent(stream, prepEvent_), "DynamicMap::ActivateNext::aclrtStreamWaitEvent");
            submaps_.push_back(std::move(nextSubmap_));
            submapSize_.push_back(0);
            totalCapacity_ += submaps_.back()->Capacity();
            nextCapacity_ = nextSubmapCap_ * GROWTH_FACTOR;
            nextSubmapCap_ = 0; // 预建表已激活，清空其容量记录
            return;
        }
        nextSubmap_.reset(); // 容量不够的预建表丢弃
        SizeType newCap = nextCapacity_;
        while (Usable(newCap) < minUsable)
            newCap *= GROWTH_FACTOR;
        submaps_.push_back(
            std::make_unique<MapType>(Extent{newCap}, emptyKey_, emptyValue_, pred_, probingScheme_, storage_, stream));
        submaps_.back()->SetErasedKey(erasedKey_, stream); // 启用墓碑删除
        submapSize_.push_back(0);
        totalCapacity_ += submaps_.back()->Capacity();
        nextCapacity_ = newCap * GROWTH_FACTOR;
    }

    /** @brief 追加一个新子表（×2 增长，保证 usable > 0）。 */
    void GrowOneSubmap(SizeType minUsable, aclrtStream stream) { ActivateNext(GrowTarget(minUsable), stream); }

    /** @brief 增长目标：至少容纳剩余批量，且不低于当前规模 ×2（几何增长），
     *  让单次增长覆盖整轮 bulk 插入，子表数收敛（少一档读扇出）。 */
    SizeType GrowTarget(SizeType minUsable) const
    {
        SizeType byGrowth = size_ * GROWTH_FACTOR;
        return (byGrowth > minUsable) ? byGrowth : minUsable;
    }

    /** @brief 保证活跃(最后一个)子表还能容纳 n 个新元素，否则追加一个足够大的新子表。 */
    void EnsureActiveRoom(SizeType n, aclrtStream stream)
    {
        if (ActiveRoom() >= n)
            return;
        ActivateNext(GrowTarget(n), stream);
    }

    /** @brief 单块插入（chunk ≤ 活跃子表余量）：跨子表去重 + 活跃子表插入。 */
    SizeType InsertChunk(void* values, SizeType n, aclrtStream stream)
    {
        size_t a = ActiveIdx();
        SizeType ins;
        if (!HasNonEmptyBefore(a)) { // 单子表快路径：开销不变
            SizeType failed = submaps_[a]->Insert(values, Extent{n}, stream);
            ins = n - failed;
        } else {
            // mask[i]=1 表示 key 已在旧子表 → 只插 mask==0 的元素（活跃子表内 CAS 自带去重）。
            BuildContainsMask(values, n, stream);
            counter_.Initialize(0, stream);
            aclco::CountZero<>
                <<<aivCoreNum_, 0, stream>>>(Mask(), static_cast<uint32_t>(n), (uint32_t*)counter_.Data());
            SizeType failed = submaps_[a]->template InsertIf<uint8_t, MaskIsZero>(values, Mask(), Extent{n}, stream);
            SizeType candidates = static_cast<SizeType>(counter_.LoadToHost(stream)); // InsertIf 已同步
            ins = candidates - failed;
        }
        submapSize_[a] += ins;
        size_ += ins;
        return ins;
    }

    /** @brief 单块 IoA（chunk ≤ 活跃子表余量）：旧子表 assign-only + 活跃子表 Insert 计数 + Assign 覆盖。
     *  返回值只计真正新插入（assign 不计入）。 */
    SizeType InsertOrAssignChunk(void* values, SizeType n, aclrtStream stream)
    {
        size_t a = ActiveIdx();
        SizeType ins;
        if (!HasNonEmptyBefore(a)) { // 单子表快路径：计数版 IoA 单内核，assign 不计入返回值
            SizeType assigned = 0;
            SizeType failed = submaps_[a]->InsertOrAssign(values, Extent{n}, stream, assigned);
            ins = n - failed - assigned;
        } else {
            // mask!=0（命中旧子表）→ 各旧子表 assign-only；mask==0 → 活跃子表先 assign 再 InsertIf 计数。
            BuildContainsMask(values, n, stream);
            counter_.Initialize(0, stream);
            aclco::CountZero<>
                <<<aivCoreNum_, 0, stream>>>(Mask(), static_cast<uint32_t>(n), (uint32_t*)counter_.Data());
            for (size_t s = 0; s < a; ++s) {
                if (submapSize_[s] == 0)
                    continue;
                submaps_[s]->template AssignIfAsync<uint8_t, MaskNonZero>(values, Mask(), Extent{n}, stream);
            }
            submaps_[a]->template AssignIfAsync<uint8_t, MaskIsZero>(values, Mask(), Extent{n}, stream);
            SizeType failed = submaps_[a]->template InsertIf<uint8_t, MaskIsZero>(values, Mask(), Extent{n}, stream);
            SizeType candidates = static_cast<SizeType>(counter_.LoadToHost(stream)); // InsertIf 已同步
            ins = candidates - failed;
        }
        submapSize_[a] += ins;
        size_ += ins;
        return ins;
    }

    std::vector<std::unique_ptr<MapType>> submaps_;
    std::vector<SizeType> submapSize_;
    SizeType size_, totalCapacity_, nextCapacity_, minInsertSize_;
    float maxLoadFactor_;
    Key emptyKey_;
    T emptyValue_;
    Key erasedKey_;
    KeyEqual pred_;
    ProbingScheme probingScheme_;
    Storage storage_;

    struct AclStreamDeleter {
        void operator()(void* p) const noexcept
        {
            if (p)
                aclrtDestroyStream((aclrtStream)p);
        }
    };
    struct AclEventDeleter {
        void operator()(void* p) const noexcept
        {
            if (p)
                aclrtDestroyEvent((aclrtEvent)p);
        }
    };

    std::unique_ptr<void, AclFreeDeleter> dEmptyValue_; // emptyValue 的 device 副本
    std::unique_ptr<void, AclFreeDeleter> dScratch_;    // 合并临时缓冲（按需增长）
    std::unique_ptr<void, AclFreeDeleter> dDedup_;      // 去重 keys/mask/tmp 缓冲（按需增长）
    size_t scratchBytes_ = 0;
    size_t dedupBytes_ = 0, keysBytes_ = 0, maskBytes_ = 0;
    ArgStorage<uint32_t> counter_;        // 去重候选计数器
    std::unique_ptr<MapType> nextSubmap_; // 后台流预建的下一子表
    SizeType nextSubmapCap_ = 0;
    std::unique_ptr<void, AclStreamDeleter> prepStreamHolder_;
    std::unique_ptr<void, AclEventDeleter> prepEventHolder_;
    aclrtStream prepStream_ = nullptr;
    aclrtEvent prepEvent_ = nullptr;
    uint32_t aivCoreNum_ = 1;
};

} // namespace aclco
