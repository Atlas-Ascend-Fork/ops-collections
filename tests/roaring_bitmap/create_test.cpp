/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "roaring_bitmap.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/roaring_bitmap_factory.h"
#include "tests/common/test_print.h"

namespace {
struct OffsetAllocation {
    uint8_t* raw{};
    size_t bytes{};
    uint8_t* released{};
    size_t releaseCount{};
};

struct OffsetAllocator {
    using ValueType = uint8_t;
    OffsetAllocation* state;

    uint8_t* Allocate(size_t bytes) const
    {
        state->bytes = bytes;
        state->raw = aclco::DefaultAllocator<uint8_t>{}.Allocate(bytes + 1);
        return state->raw ? state->raw + 1 : nullptr;
    }

    void Deallocate(uint8_t* ptr)
    {
        state->released = ptr;
        ++state->releaseCount;
        aclco::DefaultAllocator<uint8_t>{}.Deallocate(state->raw);
    }
};
} // namespace

TEMPLATE_TEST_CASE("roaring_bitmap aligns metadata for a byte allocator", "[roaring_bitmap][create][alignment]",
                   uint32_t, uint64_t)
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    using Key = TestType;
    using Extent = aclco::Extent<size_t>;
    using Metadata = std::conditional_t<std::is_same_v<Key, uint32_t>, aclco::detail::RoaringBitmapMetadata32,
                                        aclco::detail::RoaringBitmapMetadata64>;
    const auto keys = aclco::test::roaring_bitmap_factory::AcceptanceKeys<Key>();
    const auto serialized = aclco::test::roaring_bitmap_factory::Serialize(keys);
    OffsetAllocation allocation;
    const uint8_t* data = nullptr;
    {
        aclco::RoaringBitmap<Key, Extent, OffsetAllocator> bitmap(serialized.data(), serialized.size(),
                                                                 OffsetAllocator{&allocation}, guard.stream);
        data = bitmap.Data();
        REQUIRE(data == allocation.raw + 1);
        REQUIRE(reinterpret_cast<uintptr_t>(data) % alignof(Metadata) == 1);

        // Locate the metadata by its serialized-size and key-count fields in the device allocation.
        REQUIRE(allocation.bytes >= serialized.size() + sizeof(Metadata));
        std::vector<uint8_t> tail(allocation.bytes - serialized.size());
        aclco::test::CheckAcl(aclrtMemcpy(tail.data(), tail.size(), data + serialized.size(), tail.size(),
                                        ACL_MEMCPY_DEVICE_TO_HOST), "read bitmap metadata");
        const std::array<uint64_t, 2> header{serialized.size(), keys.size()};
        auto prefix = reinterpret_cast<const uint8_t*>(header.data());
        auto found = std::search(tail.begin(), tail.end(), prefix, prefix + sizeof(header));
        REQUIRE(found != tail.end());
        auto metadataAddress = reinterpret_cast<uintptr_t>(data + serialized.size()) + (found - tail.begin());
        REQUIRE(metadataAddress % alignof(Metadata) == 0);
        if constexpr (std::is_same_v<Key, uint64_t>) {
            CHECK((metadataAddress + sizeof(Metadata)) % alignof(aclco::detail::RoaringBitmapBucket) == 0);
        }

        auto moved = std::move(bitmap);
        CHECK(bitmap.Data() == nullptr);
        CHECK(moved.Data() == data);
        const std::array<Key, 2> queries{keys.front(), 0};
        aclco::test::DeviceBuffer<Key> dKeys(queries.size());
        aclco::test::DeviceBuffer<bool> dResult(queries.size());
        dKeys.CopyFromHostAsync(queries.data(), queries.size(), guard.stream);
        moved.Contains(dKeys.Data(), dResult.Data(), Extent{queries.size()}, guard.stream);
        std::array<bool, 2> result{};
        dResult.CopyToHostAsync(result.data(), result.size(), guard.stream);
        aclco::test::Sync(guard.stream);
        CHECK(result[0]);
        CHECK_FALSE(result[1]);
    }
    CHECK(allocation.released == data);
    CHECK(allocation.releaseCount == 1);
}

TEMPLATE_TEST_CASE("roaring_bitmap create test", "[roaring_bitmap][create]", uint32_t, uint64_t)
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    using Key = TestType;
    auto keys = aclco::test::roaring_bitmap_factory::AcceptanceKeys<Key>();
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize(keys);

    PRINT_BEFORE_EXEC("create test", Key, Key, 1, keys.size(), 0);
    aclco::RoaringBitmap<Key> bitmap(serialized.data(), serialized.size(), {}, stream);
    REQUIRE_PRINT(bitmap.Size() == keys.size());
    REQUIRE_PRINT(!bitmap.Empty());
    REQUIRE_PRINT(bitmap.Data() != nullptr);
    REQUIRE_PRINT(bitmap.SizeBytes() == serialized.size());
}
