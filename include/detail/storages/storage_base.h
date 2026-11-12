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

namespace aclco {
template <typename Allocator>
struct CustomDeleter {
  using Pointer = typename Allocator::ValueType*;  ///< Value pointer type

  explicit constexpr CustomDeleter(Allocator const& allocator)
    : allocator_{allocator}
  {
  }

  void operator()(Pointer ptr) { allocator_.Deallocate(ptr); }

  Allocator allocator_;
};
}
