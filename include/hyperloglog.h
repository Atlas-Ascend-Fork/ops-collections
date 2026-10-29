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
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "extent.h"

namespace aclco {
/** Owning, synchronous HyperLogLog with one 8-bit register per sketch byte.
 * Calls on the same sketch must be externally serialized, including across streams.
 * The current ACL device must remain the device on which the sketch was created.
 */
template <class Key>
class HyperLogLog {
public:
    using KeyType = Key;
    using ExtentType = aclco::Extent<std::size_t>;
    static HyperLogLog CreateWithSketchSizeKB(std::uint32_t sketchSizeKb, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithStandardDeviation(double deviation, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithPrecision(std::uint32_t precision, aclrtStream stream = nullptr);
    HyperLogLog(HyperLogLog const&) = delete;
    HyperLogLog& operator=(HyperLogLog const&) = delete;
    HyperLogLog(HyperLogLog&& other) noexcept;
    HyperLogLog& operator=(HyperLogLog&& other) noexcept;
    ~HyperLogLog();
    std::uint32_t SketchSizeKB() const noexcept { return words_ ? (1U << precision_) / 1024U : 0U; }
    void Clear(aclrtStream stream = nullptr);
    void Add(void* keys, ExtentType count, aclrtStream stream = nullptr);
    void Merge(HyperLogLog const& other, aclrtStream stream = nullptr);
    std::uint64_t Estimate(aclrtStream stream = nullptr) const;
    /** Read-only register bytes; external kernels must obey the owner's stream/lifetime contract. */
    std::uint8_t const* Data() const noexcept { return reinterpret_cast<std::uint8_t const*>(words_); }

private:
    static_assert(std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::int64_t>,
                  "HyperLogLog supports int32_t and int64_t");
    explicit HyperLogLog(std::uint32_t precision, aclrtStream stream);
    void Validate() const;
    void Release() noexcept;
    std::uint32_t* words_{nullptr};
    std::uint32_t* histogram_{nullptr};
    std::uint32_t* addWorkspace_{nullptr};
    std::uint32_t addCores_{0};
    std::uint32_t precision_{0};
    std::uint32_t cores_{0};
    std::int32_t device_{-1};
    bool knownEmpty_{false};
    mutable std::vector<std::uint32_t> hostHistogram_;
};
} // namespace aclco
#include "detail/hyperloglog/hyperloglog.inl"
