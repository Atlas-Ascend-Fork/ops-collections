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
#include <type_traits>
#include <utility>

#include "bucket_storage.h"
#include "detail/storages/arg_storage.h"
#include "kernels.h"
#include "probing_scheme.h"
#include "storage.h"
#include "tiling/platform/platform_ascendc.h"
#include "utility/allocator.h"
#include "utility/kernel_launch_utils.h"
#include "utility/is_same.h"
#include "utility/traits.h"

namespace aclco {
struct AlwaysTrue {
  template <typename T>
  COLLECTION_SIMT_DEVICE bool operator()(T) const noexcept
  {
    return true;
  }
};

namespace detail {
template <typename T>
constexpr void ValidateDefaultInstance(T const& value, char const* name)
{
  if constexpr (IsEqualityComparable<T>::value) {
    if (!(value == T{})) {
      throw std::invalid_argument(std::string(name) +
                                  " carries instance state that cannot take effect on device");
    }
  }
}
} // namespace detail

template <class Key,
          class Value,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
class OpenAddressingImpl {
 public:
  static constexpr auto bucketSize = Storage::bucketSize;
  using KeyType = std::conditional_t<std::is_arithmetic_v<Key>, Key, UintBySizeT<sizeof(Key)>>;
  using ValueType = DeviceTypeT<Value>;
  using SizeType = typename Extent::ValueType;
  using ProbingSchemeType = ProbingScheme;
  using Hasher = typename ProbingSchemeType::Hasher;
  using StorageType = BucketStorage<ValueType, bucketSize, SizeType>;

  constexpr OpenAddressingImpl(Extent capacity,
                              Value emptyValue,
                              KeyEqual const& pred,
                              ProbingScheme const& probingScheme,
                              aclrtStream stream)
    : OpenAddressingImpl(capacity, emptyValue, emptyValue, pred, probingScheme, stream, true)
  {
  }

  constexpr OpenAddressingImpl(Extent capacity,
                              Value emptyValue,
                              Value erasedValue,
                              KeyEqual const& pred,
                              ProbingScheme const& probingScheme,
                              aclrtStream stream,
                              bool clearOnConstruction)
    : probingScheme_{ValidateInputs(probingScheme, pred)},
      storage_{MakeValidExtent<ProbingScheme, Storage>(capacity), DefaultAllocator<ValueType>()},
      emptyValue_{reinterpret_cast<ValueType&>(emptyValue)},
      predicate_{pred}
  {
    emptyValueStorage_.Initialize(reinterpret_cast<ValueType&>(emptyValue), stream);
    erasedValueStorage_.Initialize(reinterpret_cast<ValueType&>(erasedValue), stream);
    if (clearOnConstruction) {
      this->Clear(stream);
    }
  }

  /** @brief 配置删除墓碑值（key 应为保留的 erasedKey，与 emptyKey、任何有效 key 都不同）。 */
  void SetErasedValue(Value erasedValue, aclrtStream stream)
  {
    erasedValueStorage_.Initialize(reinterpret_cast<ValueType&>(erasedValue), stream);
  }

  SizeType Insert(void *values, Extent valueNum, aclrtStream stream)
  {
    return InsertIf<ValueType, AlwaysTrue>(values, values, valueNum, stream);
  }


  void InsertAsync(void *values, Extent valueNum, aclrtStream stream)
  {
    InsertIfAsync<ValueType, AlwaysTrue>(values, values, valueNum, stream);
  }

  SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream)
  {
    SizeType assigned = 0;
    return InsertOrAssign(values, valueNum, stream, assigned);
  }

  /** @brief 计数重载：assignedNum 输出已存在 key 被 assign 的个数（不计入插入）。 */
  SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream, SizeType &assignedNum)
  {
    assignedNum = 0;
    if (valueNum == 0) { return 0; }
    if (values == nullptr) { return valueNum; }
    // failed/assigned 打包进单个 64bit 计数（低 32 failed、高 32 assigned）：每批仅 1 次 init + 1 次回读
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    ioaCounterStorage_.Initialize(0, stream);
    aclco::InsertOrAssign<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize,
      (uint8_t*)ioaCounterStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream);
    CheckRetOrThrow(ret, "StaticMap::InsertOrAssign::aclrtSynchronizeStream");
    uint64_t packed = ioaCounterStorage_.LoadToHost(stream);
    assignedNum = static_cast<SizeType>(packed >> 32);
    return static_cast<SizeType>(packed & 0xFFFFFFFFu);
  }

  void InsertOrAssignAsync(void *values, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    aclco::InsertOrAssignAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  template <typename StencilT, typename Predicate>
  void AssignIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr || stencil == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    aclco::AssignIfAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }


  template <typename StencilT, typename Predicate>
  SizeType InsertIf(void *values, void *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return 0; }
    if (values == nullptr) { return valueNum; }
    if (stencil == nullptr) { return valueNum; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    argStorage_.InitializeZeroAsync(stream);
    aclco::InsertIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize,
      (uint8_t*)argStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream);
    if (isPairV<ValueType>) {
      CheckRetOrThrow(ret, "StaticMap::InsertIf::aclrtSynchronizeStream");
    }
    else {
      CheckRetOrThrow(ret, "StaticSet::InsertIf::aclrtSynchronizeStream");
    }
    return argStorage_.LoadToHost(stream);
  }

  template <typename StencilT, typename Predicate>
  void InsertIfAsync(void *values, void *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    if (stencil == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    aclco::InsertIfAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  void InsertAndFindAsync(void *values, void *outputFind, void* outputInsert, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(valueNum, aivCoreNum);
    aclco::InsertAndFindAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)outputFind,
      (uint8_t*)outputInsert,      
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return 0; }
    if (keys == nullptr) { return keyNum; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(keyNum, aivCoreNum);
    argStorage_.InitializeZeroAsync(stream);
    aclco::Erase<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      (uint8_t*)erasedValueStorage_.Data(),
      storage_.Capacity(),
      batchSize,
      (uint8_t*)argStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream); 
    if (isPairV<ValueType>) {
      CheckRetOrThrow(ret, "StaticMap::Erase::aclrtSynchronizeStream");
    }
    else {
      CheckRetOrThrow(ret, "StaticSet::Erase::aclrtSynchronizeStream");
    }
    return argStorage_.LoadToHost(stream);
  }

  void EraseAsync(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(keyNum, aivCoreNum);
    aclco::EraseAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      (uint8_t*)erasedValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  template <typename StencilT, typename Predicate>
  void FindIfAsync(void *keys, void *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    if (stencil == nullptr) { return; }
    if (outputValues == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(keyNum, aivCoreNum);
    aclco::FindIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    FindIfAsync<KeyType, AlwaysTrue>(keys, keys, outputValues, keyNum, stream);
  }

  template <typename StencilT, typename Predicate>
  void ContainsIfAsync(void *keys, void *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    if (stencil == nullptr) { return; }
    if (outputValues == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(keyNum, aivCoreNum);
    aclco::ContainsIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize);
  }

  void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    ContainsIfAsync<KeyType, AlwaysTrue>(keys, keys, outputValues, keyNum, stream);
  }

  template <typename CallbackOp>
  void ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto batchSize = CheckedBatchSize(keyNum, aivCoreNum);
    aclco::ForEach<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, CallbackOp><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      batchSize,
      (uint8_t*)callbackArgs);
  }

  bool ClearSimd(uint8_t *table, uint64_t tableCapacity, aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 统一以 32 位字为元素单位：Duplicate 仅支持到 32bit，更宽的槽位拆成 32 位字交错模式
    uint64_t tmp = ubSize / BLOCK_SIZE / BUFFER_NUM;
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
      uint64_t tb = tmp / sizeof(uint32_t);
      tileBlockNum = (tb == 0) ? 1U : tb;
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE;
    if (tileDataNum == 0U) tileDataNum = BLOCK_SIZE;

    // 总 block 数（向上取整）
    uint32_t totalBlockNum = static_cast<uint32_t>((tableCapacity + BLOCK_SIZE - 1ULL) / BLOCK_SIZE);

    uint32_t actualCoreNum = static_cast<uint32_t>(aivCoreNum);
    if (actualCoreNum > totalBlockNum) actualCoreNum = totalBlockNum;
    if (actualCoreNum == 0U) actualCoreNum = 1U; // 最少 1 core

    uint32_t eachCoreBlockNum = totalBlockNum / actualCoreNum; // 每一个核分到的基本块数
    uint32_t coresWithExtra = totalBlockNum % actualCoreNum; // 前 coresWithExtra 个核是 remain-core

    // common-core 数量（元素）
    uint32_t commonCoreDataNum = eachCoreBlockNum * BLOCK_SIZE;
    uint32_t commonTileNum = eachCoreBlockNum / tileBlockNum;
    uint32_t commonRemainBlock = eachCoreBlockNum % tileBlockNum;
    uint32_t commonAlignTileNum = commonTileNum + (commonRemainBlock > 0 ? 1 : 0);
    uint32_t commonTailDataNum = (commonRemainBlock > 0) ? (commonRemainBlock * BLOCK_SIZE) : 0;

    // remain-core 数量（每个多一个block）
    uint32_t earchRemainCoreBlockNum = eachCoreBlockNum + 1U;
    uint32_t remainCoreDataNum = earchRemainCoreBlockNum * BLOCK_SIZE;
    uint32_t remainTileNum = earchRemainCoreBlockNum / tileBlockNum;
    uint32_t remainRemainBlock = earchRemainCoreBlockNum % tileBlockNum;
    uint32_t remainAlignTileNum = remainTileNum + (remainRemainBlock > 0 ? 1 : 0);
    uint32_t remainTailDataNum = (remainRemainBlock > 0) ? (remainRemainBlock * BLOCK_SIZE) : 0;

    // pattern 取自 device 侧 emptyValue 的原始字节，按 32 位字交错模式填充；
    // 先化简为最小周期：全同时退化为纯 Duplicate（零标量补写），
    // 周期越短 device 侧 SetValue 补写越少（一次性开销随周期减小）
    uint32_t patternLen = static_cast<uint32_t>(sizeof(ValueType) / sizeof(uint32_t));
    uint32_t pattern[sizeof(ValueType) / sizeof(uint32_t)];
    {
      const uint32_t *p32 = reinterpret_cast<const uint32_t *>(&emptyValue_);
      for (uint32_t i = 0; i < patternLen; ++i) {
        pattern[i] = p32[i];
      }
    }
    for (uint32_t p = 1; p < patternLen; ++p) {
      if (patternLen % p != 0) {
        continue;
      }
      bool isPeriod = true;
      for (uint32_t i = p; i < patternLen; ++i) {
        if (pattern[i] != pattern[i % p]) {
          isPeriod = false;
          break;
        }
      }
      if (isPeriod) {
        patternLen = p;
        break;
      }
    }

    // 核/tile 偏移均为 BLOCK_SIZE 字对齐，patternLen 必须整除 BLOCK_SIZE
    // 才能保证 tile 内相位与全局相位一致，否则回退逐元素路径
    if (BLOCK_SIZE % patternLen != 0) {
      return false;
    }

    aclco::ClearSIMDU32<<<aivCoreNum, 0, stream>>>(
      table,
      reinterpret_cast<uint32_t*>(emptyValueStorage_.Data()),
      patternLen,
      commonCoreDataNum, commonAlignTileNum, commonTailDataNum,
      remainCoreDataNum, remainAlignTileNum, remainTailDataNum,
      tileDataNum, coresWithExtra);
    return true;
  }

  bool CanClearSimd() const
  {
    return sizeof(ValueType) <= 16 && sizeof(ValueType) % sizeof(uint32_t) == 0;
  }

  void Clear(aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    if (CanClearSimd() && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType) &&
        ClearSimd((uint8_t*)storage_.Data(), storage_.Capacity() * sizeof(ValueType) / sizeof(uint32_t), stream)) {
      uint32_t ret = aclrtSynchronizeStream(stream);
      CheckRet(ret, "StaticMap::Clear::aclrtSynchronizeStream");
    } else {
      storage_.Initialize(emptyValueStorage_.Data(), stream);
    }
  }

  void ClearAsync(aclrtStream stream) noexcept
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    if (CanClearSimd() && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType) &&
        ClearSimd((uint8_t*)storage_.Data(), storage_.Capacity() * sizeof(ValueType) / sizeof(uint32_t), stream)) {
      return;
    }
    storage_.InitializeAsync(emptyValueStorage_.Data(), stream);
  }

  SizeType Count(void *keys, Extent keyNum, aclrtStream stream)
  {
    const auto requested = static_cast<uint64_t>(static_cast<SizeType>(keyNum));
    if (requested == 0 || keys == nullptr) {
      return 0;
    }

    const auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    const auto neededBlocks = 1U + (requested - 1U) / COUNT_THREAD_NUM;
    const auto blocks = static_cast<uint32_t>(neededBlocks < aivCoreNum ? neededBlocks : aivCoreNum);
    const auto count = CheckedBatchSize(keyNum, blocks, COUNT_THREAD_NUM);
    if (count > COUNT_WARP_SIZE) {
      argStorage_.InitializeZeroAsync(stream);
    }
    aclco::Count<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<blocks, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint32_t*)argStorage_.Data(),
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      count);

    uint32_t ret = aclrtSynchronizeStream(stream); 
    if (isPairV<ValueType>) {
      CheckRetOrThrow(ret, "StaticMap::Count::aclrtSynchronizeStream");
    }
    else {
      CheckRetOrThrow(ret, "StaticSet::Count::aclrtSynchronizeStream");
    }
    return argStorage_.LoadToHost(stream);
  }

  constexpr auto Capacity() const noexcept
  {
    return storage_.Capacity();
  }

  ValueType* Data() const
  {
    return storage_.Data();
  }

  /**
   * @brief Gets the probing scheme.
   *
   * @return The probing scheme used for the container
   */
  [[nodiscard]] constexpr ProbingSchemeType const& GetProbingScheme() const noexcept
  {
    return probingScheme_;
  }

  /**
   * @brief Gets the key comparator.
   *
   * @return The comparator used to compare keys
   */
  [[nodiscard]] constexpr KeyEqual GetKeyEqual() const noexcept
  {
    return predicate_;
  }

  /**
   * @brief Gets the function(s) used to hash keys
   *
   * @return The function(s) used to hash keys
   */
  [[nodiscard]] constexpr Hasher HashFunction() const noexcept
  {
    return this->GetProbingScheme().HashFunction();
  }

 private:
  static constexpr ProbingScheme const& ValidateInputs(ProbingScheme const& probingScheme, KeyEqual const& pred)
  {
    detail::ValidateDefaultInstance(probingScheme, "ProbingScheme");
    detail::ValidateDefaultInstance(pred, "KeyEqual");
    return probingScheme;
  }

  static uint32_t CheckedBatchSize(Extent extent, uint32_t blockCount,
                                   uint32_t threadsPerBlock = std::max(DEFAULT_THREAD_NUM, THREAD_NUM_LAUNCH_BOUND))
  {
    const auto count = static_cast<uint64_t>(static_cast<SizeType>(extent));
    const auto stride = static_cast<uint64_t>(blockCount) * threadsPerBlock;
    // The final grid-stride increment must also fit in the 32-bit loop index.
    const auto maxBatchSize = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1 - stride;
    if (count > maxBatchSize) {
      throw std::length_error("OpenAddressing: batch size exceeds the kernel index range");
    }
    return static_cast<uint32_t>(count);
  }

 protected:
  ValueType emptyValue_;
  // The SIMT kernels update this counter through __gm__ uint32_t* (4-byte atomics).
  ArgStorage<uint32_t, uint32_t> argStorage_;
  static_assert(std::is_same_v<typename decltype(argStorage_)::ValueType, uint32_t>,
                "counter storage must match the 32-bit kernel counter");
  ArgStorage<uint64_t, uint64_t> ioaCounterStorage_;
  ArgStorage<ValueType> emptyValueStorage_;
  ArgStorage<ValueType> erasedValueStorage_;
  ProbingSchemeType probingScheme_;
  StorageType storage_;
  KeyEqual predicate_;
};
}