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

#include <type_traits>

#include "detail/storages/storage_base.h"
#include "utility/allocator.h"
#include "utility/kernel_launch_utils.h"

// 用于传递参数的storage类，例如传递emptyKey

namespace aclco {
static constexpr size_t defaultArgsStorageSize = 1;

template <typename T,
          typename Extent = uint32_t,
          typename Allocator = DefaultAllocator<T>>
class ArgStorage {
 public:
  static_assert(std::is_integral_v<Extent> && std::is_unsigned_v<Extent>,
                "SizeType must be an unsigned integral type to keep probe indexes valid");
  using SizeType = Extent;
  using ValueType = T;
  using AllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<ValueType>;
  using SlotDeleterType = CustomDeleter<AllocatorType>;

  explicit constexpr ArgStorage(Allocator const& allocator = {})
  : allocator_(allocator),
    capacity_(sizeof(ValueType)),
    slots_{allocator_.Allocate(defaultArgsStorageSize),
           SlotDeleterType{allocator_}}
  {
  }

  ArgStorage(ArgStorage&&) = default;
  ArgStorage& operator=(ArgStorage&&) = default;
  ~ArgStorage() = default;
  ArgStorage(ArgStorage const&) = delete;
  ArgStorage& operator=(ArgStorage const&) = delete;

  constexpr ValueType* Data() const noexcept
  {
    return slots_.get();
  }

  constexpr ValueType* Data() noexcept
  {
    return slots_.get();
  }

  constexpr ValueType LoadToHost(aclrtStream stream = nullptr) const
  {
    ValueType hCount{};
    uint32_t ret = aclrtMemcpy((void*)&hCount, sizeof(hCount), (void*)slots_.get(), sizeof(hCount),
                              ACL_MEMCPY_DEVICE_TO_HOST);
    CheckRetOrThrow(ret, "ArgStorage::LoadToHost");
    return hCount;
  }

  constexpr SizeType Capacity() const noexcept
  {
    return capacity_;
  }

  void Initialize(ValueType value, aclrtStream stream = nullptr)
  {
    uint32_t ret = aclrtMemcpy((void*)slots_.get(), capacity_, (void*)&value, capacity_, ACL_MEMCPY_HOST_TO_DEVICE);
    CheckRet(ret, "ArgStorage::Initialize::aclrtMemcpy");
  }

  // Enqueue counter reset before its consuming kernel on the same stream.
  void InitializeZeroAsync(aclrtStream stream = nullptr)
  {
    CheckRetOrThrow(aclrtMemsetAsync(slots_.get(), capacity_, 0, capacity_, stream),
                    "ArgStorage::InitializeZeroAsync::aclrtMemsetAsync");
  }

 private:
  SizeType capacity_;
  AllocatorType allocator_;
  std::unique_ptr<ValueType, SlotDeleterType> slots_;
};
}
