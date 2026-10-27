/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "static_map.h"
#pragma once

namespace aclco {
template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
constexpr StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::
  StaticMap(Extent capacity,
            Key emptyKey,
            T emptyValue,
            KeyEqual const& pred,
            ProbingScheme const& probingScheme,
            Storage storage,
            aclrtStream stream)
  : impl_{std::make_unique<ImplType>(capacity,
                                     Pair{emptyKey, emptyValue},
                                     pred,
                                     probingScheme,
                                     stream)},
    emptyKey_{emptyKey},
    emptyValue_{emptyValue}
{
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::
  SetErasedKey(Key erasedKey, aclrtStream stream)
{
  // 墓碑槽值：key=erasedKey，value 任意（仅比较 key），取 emptyValue_ 即可。
  impl_->SetErasedValue(Pair{erasedKey, emptyValue_}, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Insert(void *values, Extent valueNum, aclrtStream stream)
{
  return impl_->Insert(values, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertAsync(void *values, Extent valueNum, aclrtStream stream)
{
  impl_->InsertAsync(values, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertOrAssign(void *values, Extent valueNum, aclrtStream stream)
{
  return impl_->InsertOrAssign(values, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertOrAssign(void *values, Extent valueNum,
                                                                            aclrtStream stream, SizeType &assignedNum)
{
  return impl_->InsertOrAssign(values, valueNum, stream, assignedNum);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertOrAssignAsync(void *values, Extent valueNum, aclrtStream stream)
{
  impl_->InsertOrAssignAsync(values, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::AssignIfAsync(
  void *values, StencilT *stencil, Extent valueNum, aclrtStream stream)
{
  impl_->template AssignIfAsync<StencilT, Predicate>(values, stencil, valueNum, stream);
}


template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertIf(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream)
{
  return impl_->template InsertIf<StencilT, Predicate>(values, stencil, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream)
{
  impl_->template InsertIfAsync<StencilT, Predicate>(values, stencil, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertAndFindAsync(void *values, void *outputFind, void *outputInsert, Extent valueNum, aclrtStream stream)
{
  impl_->InsertAndFindAsync(values, outputFind, outputInsert, valueNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::InsertAndFind(void *values, void *outputFind, void *outputInsert, Extent valueNum, aclrtStream stream)
{
  InsertAndFindAsync(values, outputFind, outputInsert, valueNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::InsertAndFind::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Erase(void *keys, Extent keyNum, aclrtStream stream)
{
  return impl_->Erase(keys, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::EraseAsync(void *keys, Extent keyNum, aclrtStream stream)
{
  impl_->EraseAsync(keys, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Clear(aclrtStream stream)
{
  impl_->Clear(stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ClearAsync(aclrtStream stream) noexcept
{
  impl_->ClearAsync(stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  FindAsync(keys, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::Find::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->FindAsync(keys, outputValues, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::FindIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  FindIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::FindIf::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::FindIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->template FindIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  ContainsAsync(keys, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::Contains::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->ContainsAsync(keys, outputValues, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ContainsIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  ContainsIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::ContainsIf::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ContainsIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->template ContainsIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
}

template <class Key, 
          class T, 
          class Extent, 
          class KeyEqual, 
          class ProbingScheme, 
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::SizeType          
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Count(void *keys, Extent keyNum, aclrtStream stream)
{
    return impl_->Count(keys, keyNum, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename CallbackOp>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ForEach(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
{
  ForEachAsync<CallbackOp>(keys, keyNum, callbackArgs, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticMap::ForEach::aclrtSynchronizeStream");
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename CallbackOp>
void StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
{
  impl_->template ForEachAsync<CallbackOp>(keys, keyNum, callbackArgs, stream);
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
constexpr auto StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Capacity() const noexcept
{
  return impl_->Capacity();
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::ValueType*
StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::Data() const
{
  return impl_->Data();
}

template <class Key,
          class T,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
constexpr KeyEqual StaticMap<Key, T, Extent, KeyEqual, ProbingScheme, Storage>::GetKeyEqual() const noexcept
{
  return impl_->GetKeyEqual();
}
}