/*
 * Copyright (c) 2022-2025， NVIDIA CPRPORATION.
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

#include <cstddef>
#include <new>

#include "detail/open_addressing/kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utility/kernel_launch_utils.h"

namespace aclco {

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr BucketStorage<T, BucketSize, Extent, Allocator>::BucketStorage(
  Extent capacity, Allocator const& allocator)
  : extent_{capacity},
  allocator_{allocator},
  slots_{nullptr, SlotDeleterType{allocator_}}
{
  slots_.reset(allocator_.Allocate(this->Capacity()));
  if (static_cast<std::size_t>(this->Capacity()) != 0 && !slots_) {
    throw std::bad_alloc{};
  }
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr typename BucketStorage<T, BucketSize, Extent, Allocator>::ValueType*
BucketStorage<T, BucketSize, Extent, Allocator>::Data() const noexcept
{
  return slots_.get();
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void BucketStorage<T, BucketSize, Extent, Allocator>::Initialize(ValueType* key, aclrtStream stream)
{
  this->InitializeAsync(key, stream);
  uint32_t ret = aclrtSynchronizeStream(stream);
  CheckRet(ret, "aclrtSynchronizeStream in BucketStorage::Initialize");
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void BucketStorage<T, BucketSize, Extent, Allocator>::InitializeAsync(ValueType* key, aclrtStream stream)
{
  if (this->Capacity() == 0) {
    return;
  }
  // auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
  // 设置blockNum = 64时，性能有提升
  aclco::Clear<ValueType><<<64, 0, stream>>>(
    (uint8_t*)this->Data(),
    this->Capacity(),
    (uint8_t*)key);
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr typename BucketStorage<T, BucketSize, Extent, Allocator>::SizeType
BucketStorage<T, BucketSize, Extent, Allocator>::Capacity() const noexcept
{
  return static_cast<SizeType>(extent_);
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr typename BucketStorage<T, BucketSize, Extent, Allocator>::ExtentType
BucketStorage<T, BucketSize, Extent, Allocator>::GetExtent() const noexcept
{
  return extent_;
}

} // namespace aclco
