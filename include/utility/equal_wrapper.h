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

#include <cstdint>

#include "detail/bitwise_compare.h"
#include "macros.h"

namespace aclco {

enum class EqualResult : int32_t {UNEQUAL = 0, EQUAL = 1, EMPTY = 2, AVAILABLE = 3};
enum class IsInsert : bool {YES, NO};

template <typename T, typename Equal>
struct EqualWrapper {
  Equal equal_;

  COLLECTION_HOST_DEVICE constexpr EqualWrapper(Equal const& equal) noexcept : equal_{equal}
  {
  }

  COLLECTION_SIMT_DEVICE constexpr EqualResult EqualTo(T const& lhs, T const& rhs) const noexcept
  {
    return equal_(lhs, rhs) ? EqualResult::EQUAL : EqualResult::UNEQUAL;
  }

  COLLECTION_HOST_DEVICE constexpr bool IsSentinelValue(T const& value, T const& sentinel) const noexcept
  {
    if constexpr (aclco::isBitwiseComparableV<T>) {
      return detail::BitWiseCompare(value, sentinel);
    } else {
      return equal_(value, sentinel);
    }
  }

  template <IsInsert isInsert>
  COLLECTION_SIMT_DEVICE constexpr EqualResult operator()(T const& lhs, T const& rhs, T const& emptyValue) const noexcept
  {
    if constexpr (isInsert == IsInsert::YES) {
      return IsSentinelValue(rhs, emptyValue) ? EqualResult::AVAILABLE : this->EqualTo(lhs, rhs);
    } else {
      return IsSentinelValue(rhs, emptyValue) ? EqualResult::EMPTY : this->EqualTo(lhs, rhs);
    }
  }

};
}
