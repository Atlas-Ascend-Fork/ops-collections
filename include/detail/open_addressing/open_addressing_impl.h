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
    : probingScheme_{probingScheme},
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
    ioaCounterStorage_.Initialize(0, stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertOrAssign<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum,
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
    aclco::InsertOrAssignAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }

  template <typename StencilT, typename Predicate>
  void AssignIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr || stencil == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::AssignIfAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }


  template <typename StencilT, typename Predicate>
  SizeType InsertIf(void *values, void *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return 0; }
    if (values == nullptr) { return valueNum; }
    if (stencil == nullptr) { return valueNum; }
    argStorage_.InitializeZeroAsync(stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum,
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
    aclco::InsertIfAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }

  void InsertAndFindAsync(void *values, void *outputFind, void* outputInsert, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertAndFindAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)outputFind,
      (uint8_t*)outputInsert,      
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }

  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return 0; }
    if (keys == nullptr) { return keyNum; }
    argStorage_.InitializeZeroAsync(stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::Erase<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      (uint8_t*)erasedValueStorage_.Data(),
      storage_.Capacity(),
      keyNum,
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
    aclco::EraseAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      (uint8_t*)erasedValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
  }

  template <typename StencilT, typename Predicate>
  void FindIfAsync(void *keys, void *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    if (stencil == nullptr) { return; }
    if (outputValues == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::FindIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
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
    aclco::ContainsIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
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
    aclco::ForEach<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, CallbackOp><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum,
      (uint8_t*)callbackArgs);
  }

  template <typename KeyType>
  void ClearSimd(uint8_t *table, KeyType emptyValue, uint64_t tableCapacity, aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto totalDataNum = tableCapacity;
    uint64_t tmp = ubSize / BLOCK_SIZE / BUFFER_NUM;
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
      uint64_t tb = tmp / sizeof(KeyType);
      tileBlockNum = (tb == 0) ? 1U : tb;
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE;
    if (tileDataNum == 0U) tileDataNum = BLOCK_SIZE;

    // 总 block 数（向上取整）
    uint32_t totalBlockNum = (totalDataNum + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;

    uint32_t actualCoreNum = static_cast<uint32_t>(aivCoreNum);
    if (actualCoreNum > totalBlockNum) actualCoreNum = totalBlockNum;
    if (actualCoreNum == 0ULL) actualCoreNum = 1ULL; // 最少 1 core

    uint32_t eachCoreBlockNum = totalBlockNum / actualCoreNum; // 每一个核分到的基本块数
    uint32_t coresWithExtra = static_cast<uint32_t>(totalBlockNum % aivCoreNum); // 前 coresWithExtra 个核是 remain-core
    
    // common-core 数量（元素）
    uint32_t commonCoreDataNum = static_cast<uint32_t>(eachCoreBlockNum * BLOCK_SIZE);
    uint32_t commonTileNum = static_cast<uint32_t>(eachCoreBlockNum / tileBlockNum);
    uint32_t commonRemainBlock = eachCoreBlockNum % tileBlockNum;
    uint32_t commonAlignTileNum = commonTileNum + (commonRemainBlock > 0 ? 1 : 0);
    uint32_t commonTailDataNum = (commonRemainBlock > 0) ? (commonRemainBlock * BLOCK_SIZE) : 0;

    // remain-core 数量（每个多一个block）
    uint32_t earchRemainCoreBlockNum = eachCoreBlockNum + 1ULL;
    uint32_t remainCoreDataNum = static_cast<uint32_t>(earchRemainCoreBlockNum * BLOCK_SIZE);
    uint32_t remainTileNum = static_cast<uint32_t>(earchRemainCoreBlockNum / static_cast<uint32_t>(tileBlockNum));
    uint32_t remainRemainBlock = earchRemainCoreBlockNum % tileBlockNum;
    uint32_t remainAlignTileNum = remainTileNum + (remainRemainBlock > 0 ? 1 : 0);
    uint32_t remainTailDataNum = (remainRemainBlock > 0) ? (remainRemainBlock * BLOCK_SIZE) : 0;

    aclco::ClearSIMD<KeyType><<<aivCoreNum, 0, stream>>>(
      table,
      emptyValue,
      commonCoreDataNum, commonAlignTileNum, commonTailDataNum,
      remainCoreDataNum, remainAlignTileNum, remainTailDataNum,
      tileDataNum, coresWithExtra);
  }

  bool CanClearSimd() const
  {
    if constexpr (isPairV<ValueType>) {
      return sizeof(ValueType) <= 8 && sizeof(emptyValue_.first) == sizeof(emptyValue_.second) &&
             detail::Memcmp(&emptyValue_.first, &emptyValue_.second, sizeof(emptyValue_.first)) == 0;
    } else {
      return sizeof(ValueType) <= 8;
    }
  }

  void Clear(aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    if (CanClearSimd() && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType)) {
      if constexpr (isPairV<ValueType>) {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_.first,
                  storage_.Capacity() * sizeof(ValueType) / sizeof(typename ValueType::FirstType), stream);
        uint32_t ret = aclrtSynchronizeStream(stream); 
        CheckRet(ret, "StaticMap::Clear::aclrtSynchronizeStream");
      } else {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_, storage_.Capacity(), stream);
        uint32_t ret = aclrtSynchronizeStream(stream); 
        CheckRet(ret, "StaticSet::Clear::aclrtSynchronizeStream");
      }
    } else {
      storage_.Initialize(emptyValueStorage_.Data(), stream);
    }
  }

  void ClearAsync(aclrtStream stream) noexcept
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
     if (CanClearSimd() && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType)) {
      if constexpr (isPairV<ValueType>) {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_.first,
                  storage_.Capacity() * sizeof(ValueType) / sizeof(typename ValueType::FirstType), stream);
      } else {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_, storage_.Capacity(), stream);
      }
    } else {
      storage_.InitializeAsync(emptyValueStorage_.Data(), stream);
    }
  }

  SizeType Count(void *keys, Extent keyNum, aclrtStream stream)
  {
    auto count = static_cast<uint32_t>(static_cast<SizeType>(keyNum));
    if (count == 0 || keys == nullptr) {
      return 0;
    }
    if (count > COUNT_WARP_SIZE) {
      argStorage_.InitializeZeroAsync(stream);
    }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    auto neededBlocks = 1U + (count - 1U) / COUNT_THREAD_NUM;
    auto blocks = neededBlocks < aivCoreNum ? neededBlocks : aivCoreNum;
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