/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "static_set.h"
#pragma once


namespace aclco {
template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
constexpr StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::
  StaticSet(Extent capacity,
            Key emptyKey,
            KeyEqual const& pred,
            ProbingScheme const& probingScheme,
            Storage storage,
            aclrtStream stream)
  : impl_{std::make_unique<ImplType>(capacity,
                                     emptyKey,
                                     pred,
                                     probingScheme,
                                     stream)},
    emptyKey_{emptyKey}
{
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Insert(void *keys, Extent keyNum, aclrtStream stream)
{
  return impl_->Insert(keys, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::InsertAsync(void *keys, Extent keyNum, aclrtStream stream)
{
  impl_->InsertAsync(keys, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
typename StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::InsertIf(void *keys, StencilT *stencil, Extent keyNum, aclrtStream stream)
{
  return impl_->template InsertIf<StencilT, Predicate>(keys, stencil, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::InsertIfAsync(void *keys, StencilT *stencil, Extent keyNum, aclrtStream stream)
{
  impl_->template InsertIfAsync<StencilT, Predicate>(keys, stencil, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::InsertAndFindAsync(void *keys, void *outputFind, void *outputInsert, Extent keyNum, aclrtStream stream)
{
  impl_->InsertAndFindAsync(keys, outputFind, outputInsert, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::InsertAndFind(void *keys, void *outputFind, void *outputInsert, Extent keyNum, aclrtStream stream)
{
  InsertAndFindAsync(keys, outputFind, outputInsert, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::InsertAndFind::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Erase(void *keys, Extent keyNum, aclrtStream stream)
{
  return impl_->Erase(keys, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::EraseAsync(void *keys, Extent keyNum, aclrtStream stream)
{
  impl_->EraseAsync(keys, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Clear(aclrtStream stream)
{
  impl_->Clear(stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ClearAsync(aclrtStream stream) noexcept
{
  impl_->ClearAsync(stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  FindAsync(keys, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::Find::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->FindAsync(keys, outputValues, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::FindIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  FindIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::FindIf::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::FindIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->template FindIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  ContainsAsync(keys, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::Contains::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->ContainsAsync(keys, outputValues, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ContainsIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  ContainsIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::ContainsIf::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename StencilT, typename Predicate>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ContainsIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
{
  impl_->template ContainsIfAsync<StencilT, Predicate>(keys, stencil, outputValues, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::SizeType
StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Count(void *keys, Extent keyNum, aclrtStream stream)
{
  return impl_->Count(keys, keyNum, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename CallbackOp>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ForEach(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
{
  ForEachAsync<CallbackOp>(keys, keyNum, callbackArgs, stream);
  CheckRetOrThrow(aclrtSynchronizeStream(stream), "StaticSet::ForEach::aclrtSynchronizeStream");
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
template <typename CallbackOp>
void StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
{
  impl_->template ForEachAsync<CallbackOp>(keys, keyNum, callbackArgs, stream);
}

template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
constexpr auto StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Capacity() const noexcept
{
  return impl_->Capacity();
}
template <class Key,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
typename StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::ValueType*
StaticSet<Key, Extent, KeyEqual, ProbingScheme, Storage>::Data() const
{
  return impl_->Data();
}
}