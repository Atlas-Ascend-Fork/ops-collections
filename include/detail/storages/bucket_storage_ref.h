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
#include <type_traits>

#include <acl/acl.h>

#include "detail/storages/storage_base.h"
#include "macros.h"

namespace aclco {
template<typename T,
         int32_t BucketSize,
         typename Extent = uint32_t>
class BucketStorageRef {
public:
    static_assert(std::is_integral_v<Extent> && std::is_unsigned_v<Extent>,
                  "SizeType must be an unsigned integral type to keep probe indexes valid");
    static constexpr uint32_t bucketSize = BucketSize; ///< Number of elements processed per bucket
    using SizeType = Extent; ///< Storage size type
    using ValueType = T; ///< Slot type
    using Iterator = __gm__ ValueType*;        ///< Iterator type
    using ConstIterator = __gm__ ValueType const*;  ///< Const forward iterator type
    /*static constexpr std::size_t alignment =
        std::min((std::size_t)(sizeof(T) * bucketSize), std::size_t{16});*/


    
    COLLECTION_SIMT_DEVICE explicit constexpr BucketStorageRef(Extent size, __gm__ ValueType* slots) noexcept;
    COLLECTION_SIMT_DEVICE constexpr Iterator End() noexcept;
    COLLECTION_SIMT_DEVICE constexpr Iterator End() const noexcept;
    COLLECTION_SIMT_DEVICE constexpr __gm__ ValueType* Data() noexcept;
    COLLECTION_SIMT_DEVICE constexpr __gm__ ValueType* Data() const noexcept;
    COLLECTION_SIMT_DEVICE constexpr SizeType NumBuckets() const noexcept;
    COLLECTION_SIMT_DEVICE constexpr SizeType Capacity() const noexcept;
private:
    SizeType capacity_;
    __gm__ ValueType* slots_;

};
} // namespace aclco
#include "detail/storages/bucket_storage_ref.inl"

