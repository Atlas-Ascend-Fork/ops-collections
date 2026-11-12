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
#include "static_map_ref.h"
#include "static_set_ref.h"
#include "detail/storages/bucket_storage_ref.h"
#include "probing_scheme.h"
#include "hash_functions.h"
#include "pair.h"
#include "utility/is_same.h"
#include "utility/traits.h"

namespace aclco {
constexpr uint32_t MAX_THREAD_NUM = 2048;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
constexpr uint32_t DEFAULT_THREAD_NUM = 1024;
constexpr uint32_t COUNT_THREAD_NUM = 256;
constexpr uint32_t COUNT_WARP_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename Value>
COLLECTION_SIMT_DEVICE void CopyEmptyValue(__gm__ Value* value, __gm__ Value* emptyValue)
{
  if constexpr (isPairV<Value>) {
    using FirstRawType = UintBySizeT<sizeof(typename Value::FirstType)>;
    using SecondRawType = UintBySizeT<sizeof(typename Value::SecondType)>;
    *reinterpret_cast<__gm__ FirstRawType*>(&(value->first)) = *reinterpret_cast<__gm__ FirstRawType*>(&(emptyValue->first));
    *reinterpret_cast<__gm__ SecondRawType*>(&(value->second)) = *reinterpret_cast<__gm__ SecondRawType*>(&(emptyValue->second));
  }
  else {
    *reinterpret_cast<__gm__ UintBySizeT<sizeof(Value)>*>(value) = *reinterpret_cast<__gm__ UintBySizeT<sizeof(Value)>*>(emptyValue);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum, __gm__ uint32_t *insertFailedNum)
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  // 每线程寄存器累加失败数，循环末一次全局原子加（避免高重复批次序列化）。
  uint32_t localFailed = 0;
  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    if (!pred(stencilValue)) { continue; }
    Value insertValue = *((__gm__ Value*)(values) + i);
    if (!ref.Insert(insertValue)) { 
      localFailed += addVal;
    }
  }
  if (localFailed != 0) {
    AscendC::Simt::AtomicAdd(insertFailedNum, localFailed);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertIfSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    if (!pred(stencilValue)) { continue; }
    Value insertValue = *((__gm__ Value*)(values) + i);
    ref.Insert(insertValue);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertOrAssignSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum, __gm__ uint32_t *failedAndAssignedNum)
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  // 失败/assign 计数均寄存器累加，线程末各一次全局原子（避免高重复批次序列化）。
  uint32_t localFailed = 0;
  uint32_t localAssigned = 0;
  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value insertValue = *((__gm__ Value*)(values) + i);
    InsertResult r = ref.InsertOrAssignVerdict(insertValue);
    if (r == InsertResult::FAILED) {
      localFailed += addVal;
    } else if (r == InsertResult::DUPLICATE) {
      localAssigned += addVal;
    }
  }
  if (localFailed != 0) {
    AscendC::Simt::AtomicAdd(failedAndAssignedNum, localFailed);
  }
  if (localAssigned != 0) {
    AscendC::Simt::AtomicAdd(failedAndAssignedNum + 1, localAssigned);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertOrAssignSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value insertValue = *((__gm__ Value*)(values) + i);
    ref.InsertOrAssign(insertValue);
  }
}

// assign-only（stencil 条件）：pred(stencil[i]) 为真才执行；key 已存在则更新 value，不存在不插入。
template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void AssignIfSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  Predicate pred = {};
  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    if (!pred(stencilValue)) { continue; }
    Value assignValue = *((__gm__ Value*)(values) + i);
    ref.Assign(assignValue);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertAndFindSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *outputFind, __gm__ uint8_t *outputInsert,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value value = *((__gm__ Value*)(values) + i);
    auto const [found, inserted] = ref.InsertAndFind(value);

    if constexpr (isPairV<Value>) {
      *((__gm__ typename Value::SecondType*)(outputFind) + i) = found;
    } else {
      *((__gm__ Value*)(outputFind) + i) = found;
    }
    *((__gm__ bool*)(outputInsert) + i) = inserted;
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void EraseSimt( // 如果传入的对象或结构体只支持传数据，则意味着上层软件设计必须考虑到编译器不支持的问题，设计上将数据和处理函数分开
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue, __gm__ uint8_t *erasedValue,
  uint32_t tableSize, uint32_t keyNum, __gm__ uint32_t *eraseFailedNum) // 这些参数待后续编译器支持结构体传参后整合成结构体
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);
  Value erasedSlot = *((__gm__ Value*)erasedValue);   // 墓碑值（key=erasedKey；未配置时=emptyValue）

  // 每线程在寄存器里累加失败数，循环结束后每线程仅做一次全局原子加，
  // 避免大规模 miss 时所有失败线程序列化到单一计数器地址。
  uint32_t localFailed = 0;
  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key eraseKey = *((__gm__ Key*)(keys) + i);
    if (!ref.Erase(eraseKey, erasedSlot)) {
       localFailed += addVal;
     }
  }
  if (localFailed != 0) {
    AscendC::Simt::AtomicAdd(eraseFailedNum, localFailed);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void EraseSimtAsync( // 如果传入的对象或结构体只支持传数据，则意味着上层软件设计必须考虑到编译器不支持的问题，设计上将数据和处理函数分开
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue, __gm__ uint8_t *erasedValue,
  uint32_t tableSize, uint32_t keyNum) // 这些参数待后续编译器支持结构体传参后整合成结构体
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);
  Value erasedSlot = *((__gm__ Value*)erasedValue);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key eraseKey = *((__gm__ Key*)(keys) + i);
    ref.Erase(eraseKey, erasedSlot);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void FindIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    Key findKey = *((__gm__ Key*)(keys) + i);
    if constexpr (isPairV<Value>) {
      *((__gm__ typename Value::SecondType*)(outputValues) + i) = pred(stencilValue)
        ? ref.Find(findKey) : ((__gm__ Value*)emptyValue)->second;
    } else {
      *((__gm__ Value*)(outputValues) + i) = pred(stencilValue)
        ? ref.Find(findKey) : *((__gm__ Value*)emptyValue);
    }
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ContainsIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    Key findKey = *((__gm__ Key*)(keys) + i);
    *((__gm__ bool*)(outputValues) + i) = pred(stencilValue) ? ref.Contains(findKey) : false;
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename CallbackOp>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ForEachSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t keyNum, __gm__ uint8_t *callbackArgs)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  CallbackOp callback = CallbackOp(callbackArgs);

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key probeKey = *((__gm__ Key*)(keys) + i);
    ref.ForEach(probeKey, callback);
  }
}

template <typename Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(MAX_THREAD_NUM) inline void ClearSimt(
  __gm__ uint8_t *table, uint32_t tableSize, __gm__ uint8_t *emptyValue) 
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using ValueType = Value;

  __gm__ ValueType* data = (__gm__ ValueType*)table;
  __gm__ ValueType* emptyVal = (__gm__ ValueType*)emptyValue;

  for (uint32_t i = globalThreadIdx; i < tableSize; i = i + totalThreadNum) {
    CopyEmptyValue(&data[i], emptyVal);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void CountSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint32_t *outputNum,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  uint32_t lane = AscendC::Simt::GetThreadIdx() % COUNT_WARP_SIZE;
  if (globalThreadIdx - lane >= keyNum) {
    return;
  }

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  uint32_t localCount = 0;
  for (uint32_t i = globalThreadIdx; i < keyNum; i += totalThreadNum) {
    Key targetKey = *((__gm__ Key*)(keys) + i);
    localCount += ref.Count(targetKey);
  }
  uint32_t warpCount = AscendC::Simt::WarpReduceAddSync(localCount);
  if (lane == 0) {
    if (keyNum <= COUNT_WARP_SIZE) {
      // A single warp owns the entire result, including a count of zero.
      *outputNum = warpCount;
    } else if (warpCount != 0) {
      AscendC::Simt::AtomicAdd(outputNum, warpCount);
    }
  }
}

template <typename Key>
__vector__ __global__ __aicore__ void ClearSIMD(__gm__ uint8_t *table, Key emptyKey,
                                                uint32_t commonCoreDataNum, uint32_t commonCoreAlignTileNum, uint32_t commonCoreTailDataNum,
                                                uint32_t remainCoreDataNum, uint32_t remainCoreAlignTileNum, uint32_t remainCoreTailDataNum,
                                                uint32_t tileDataNum, uint32_t remainBlockNum)
{
  // 只能为32bit, 因为当前Duplicate接口仅支持到32bit
  using KeyType = Key;

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> tableQueue;
  AscendC::GlobalTensor<KeyType> tableGM;

  uint32_t coreId = AscendC::GetBlockIdx();
  uint32_t globalBufferIdx;

  uint32_t coreDataNum;
  uint32_t tileNum;
  uint32_t tailDataNum;
  uint32_t processDataNum;

  if (coreId < remainBlockNum) {
    coreDataNum = remainCoreDataNum;
    tileNum = remainCoreAlignTileNum;
    tailDataNum = remainCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * coreId;
  } else {
    coreDataNum = commonCoreDataNum;
    tileNum = commonCoreAlignTileNum;
    tailDataNum = commonCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * remainBlockNum + (coreId - remainBlockNum) * commonCoreDataNum;
  }

  tableGM.SetGlobalBuffer((__gm__ KeyType*)table + globalBufferIdx, coreDataNum);
  pipe.InitBuffer(tableQueue, BUFFER_NUM, tileDataNum * sizeof(KeyType));

  uint32_t loopCount = tileNum;
  processDataNum = tileDataNum;
  AscendC::LocalTensor<KeyType> tableLocal = tableQueue.AllocTensor<KeyType>();
  AscendC::Duplicate<KeyType>(tableLocal, emptyKey, processDataNum);
  tableQueue.EnQue(tableLocal);
  tableLocal = tableQueue.DeQue<KeyType>();
  for (uint32_t i = 0; i < loopCount; i++) {
    if (i == tileNum - 1 && tailDataNum != 0) {
      processDataNum = tailDataNum;
    }
    AscendC::DataCopy(tableGM[i * tileDataNum], tableLocal, processDataNum);
  }
  tableQueue.FreeTensor(tableLocal);
}

// 通用 32 位字视图清除核：以 32 位字为填充单位（Duplicate 仅支持到 32bit），
// 更宽的槽位（如 16 字节 Pair）由 host 侧把 emptyValue 的原始字节展开为
// 32 位字交错模式（pattern，最多 4 个字，patternLen = sizeof(ValueType) / 4）。
// host 保证 patternLen 整除 BLOCK_SIZE，且核/tile 偏移均为 BLOCK_SIZE 字对齐，
// 因此 tile 内相位与全局相位一致。tile 造好后由下方循环反复 DataCopy 复用，
// 保持接近 memcpy 的写带宽。
__vector__ __global__ __aicore__ void ClearSIMDU32(__gm__ uint8_t *table, __gm__ uint32_t *pattern, uint32_t patternLen,
                                                   uint32_t commonCoreDataNum, uint32_t commonCoreAlignTileNum, uint32_t commonCoreTailDataNum,
                                                   uint32_t remainCoreDataNum, uint32_t remainCoreAlignTileNum, uint32_t remainCoreTailDataNum,
                                                   uint32_t tileDataNum, uint32_t remainBlockNum)
{
  using KeyType = uint32_t;

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> tableQueue;
  AscendC::GlobalTensor<KeyType> tableGM;

  uint32_t coreId = AscendC::GetBlockIdx();
  uint32_t globalBufferIdx;

  uint32_t coreDataNum;
  uint32_t tileNum;
  uint32_t tailDataNum;
  uint32_t processDataNum;

  if (coreId < remainBlockNum) {
    coreDataNum = remainCoreDataNum;
    tileNum = remainCoreAlignTileNum;
    tailDataNum = remainCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * coreId;
  } else {
    coreDataNum = commonCoreDataNum;
    tileNum = commonCoreAlignTileNum;
    tailDataNum = commonCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * remainBlockNum + (coreId - remainBlockNum) * commonCoreDataNum;
  }

  tableGM.SetGlobalBuffer((__gm__ KeyType*)table + globalBufferIdx, coreDataNum);
  pipe.InitBuffer(tableQueue, BUFFER_NUM, tileDataNum * sizeof(KeyType));

  uint32_t loopCount = tileNum;
  processDataNum = tileDataNum;
  AscendC::LocalTensor<KeyType> tableLocal = tableQueue.AllocTensor<KeyType>();
  // 先整体填 pattern[0]（PIPE_V），V_S 显式同步后再做标量补写（PIPE_S），
  // S->MTE3 由队列 EnQue/DeQue 同步
  KeyType fillValue = pattern[0];
  AscendC::Duplicate<KeyType>(tableLocal, fillValue, processDataNum);
  AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
  AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
  for (uint32_t j = 1; j < patternLen; j++) {
    KeyType value = pattern[j];
    for (uint32_t i = j; i < tileDataNum; i += patternLen) {
      tableLocal.SetValue(i, value);
    }
  }
  tableQueue.EnQue(tableLocal);
  tableLocal = tableQueue.DeQue<KeyType>();
  for (uint32_t i = 0; i < loopCount; i++) {
    if (i == tileNum - 1 && tailDataNum != 0) {
      processDataNum = tailDataNum;
    }
    AscendC::DataCopy(tableGM[i * tileDataNum], tableLocal, processDataNum);
  }
  tableQueue.FreeTensor(tableLocal);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_AIV_GLOBAL void InsertIf(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                         __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
                                                         uint32_t tableSize, uint32_t valueNum,
                                                         __gm__ uint8_t *insertFailedNum)
{
  AscendC::Simt::VF_CALL<InsertIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, stencil, emptyValue, tableSize, valueNum, (__gm__ uint32_t*)insertFailedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_AIV_GLOBAL void InsertIfAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                               __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
                                                               uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertIfSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{THREAD_NUM_LAUNCH_BOUND},
    table, values, stencil, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void InsertOrAssign(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                                __gm__ uint8_t *emptyValue,
                                                                uint32_t tableSize, uint32_t valueNum,
                                                                __gm__ uint8_t *failedAndAssignedNum)
{
  AscendC::Simt::VF_CALL<InsertOrAssignSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, tableSize, valueNum, (__gm__ uint32_t*)failedAndAssignedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void InsertOrAssignAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                                      __gm__ uint8_t *emptyValue,
                                                                      uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertOrAssignSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_AIV_GLOBAL void AssignIfAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                                __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
                                                                uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<AssignIfSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(
    AscendC::Simt::Dim3{DEFAULT_THREAD_NUM}, table, values, stencil, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void InsertAndFindAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                         __gm__ uint8_t *outputFind, __gm__ uint8_t *outputInsert,
                                                         __gm__ uint8_t *emptyValue, uint32_t tableSize,
                                                         uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertAndFindSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, outputFind, outputInsert, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void Erase(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                      __gm__ uint8_t *emptyValue, __gm__ uint8_t *erasedValue,
                                                      uint32_t tableSize, uint32_t valueNum,
                                                      __gm__ uint8_t *eraseFailedNum)
{
  AscendC::Simt::VF_CALL<EraseSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, erasedValue, tableSize, valueNum, (__gm__ uint32_t*)eraseFailedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void EraseAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                      __gm__ uint8_t *emptyValue, __gm__ uint8_t *erasedValue,
                                                      uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<EraseSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, erasedValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_AIV_GLOBAL void FindIf(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                       __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
                                                       __gm__ uint8_t *emptyValue,
                                                       uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<FindIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, stencil, outputValues, emptyValue, tableSize, keyNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
COLLECTION_AIV_GLOBAL void ContainsIf(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                           __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
                                                           __gm__ uint8_t *emptyValue,
                                                           uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<ContainsIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, stencil, outputValues, emptyValue, tableSize, keyNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename CallbackOp>
COLLECTION_AIV_GLOBAL void ForEach(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                        __gm__ uint8_t *emptyValue,
                                                        uint32_t tableSize, uint32_t keyNum,
                                                        __gm__ uint8_t *callbackArgs)
{
  AscendC::Simt::VF_CALL<ForEachSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, CallbackOp>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, emptyValue, tableSize, keyNum, callbackArgs);
}

template <typename Value>
COLLECTION_AIV_GLOBAL void Clear(__gm__ uint8_t *table, uint32_t tableSize,
                                                      __gm__ uint8_t *emptyValue)
{
  AscendC::Simt::VF_CALL<ClearSimt<Value>>(AscendC::Simt::Dim3{MAX_THREAD_NUM},
    table, tableSize, emptyValue);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
COLLECTION_AIV_GLOBAL void Count(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                       __gm__ uint32_t *outputNum, __gm__ uint8_t *emptyValue,
                                                       uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<CountSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{COUNT_THREAD_NUM},
    table, keys, outputNum, emptyValue, tableSize, keyNum);
}
}