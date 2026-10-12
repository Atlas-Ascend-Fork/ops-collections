/*
 * Copyright (c) 2026 liaopinyi714.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

#include "static_set.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

TEST_CASE("Signed 16-bit updates preserve the neighboring occupied slot", "[static_set][insert][erase][signed-cas]")
{
    aclco::test::AclGlobalGuard aclGuard;
    aclco::test::AclStreamGuard streamGuard;
    auto const stream = streamGuard.stream;
    using Key = std::int16_t;
    using Probe = aclco::LinearProbing<aclco::murmurhash3_32<Key>>;
    using Set = aclco::StaticSet<Key, aclco::Extent<std::size_t>, aclco::EqualTo<Key>, Probe, aclco::Storage<1>>;
    constexpr std::size_t capacity = 2;
    constexpr Key empty = -1;
    Key const positive = 2;
    Key const negative = -2;

    // Establish that the two real hashes address opposite halfwords of one CAS word.
    Probe probing{};
    REQUIRE(*probing.MakeIterator<1>(positive, capacity) == 1);
    REQUIRE(*probing.MakeIterator<1>(negative, capacity) == 0);
    Set set(aclco::Extent<std::size_t>{capacity}, empty, {}, probing, {}, stream);
    REQUIRE(set.Capacity() == capacity);
    aclco::test::DeviceBuffer<Key> input(1);
    input.CopyFromHostAsync(&positive, 1, stream);
    REQUIRE(set.Insert(input.Data(), aclco::Extent<std::size_t>{1}, stream) == 0);
    input.CopyFromHostAsync(&negative, 1, stream);
    REQUIRE(set.Insert(input.Data(), aclco::Extent<std::size_t>{1}, stream) == 0);

    std::array<Key, 2> queries{positive, negative};
    std::array<bool, 2> observed{};
    aclco::test::DeviceBuffer<Key> keys(queries.size());
    aclco::test::DeviceBuffer<bool> output(observed.size());
    keys.CopyFromHostAsync(queries.data(), queries.size(), stream);
    set.Contains(keys.Data(), output.Data(), aclco::Extent<std::size_t>{queries.size()}, stream);
    output.CopyToHostAsync(observed.data(), observed.size(), stream);
    aclco::test::Sync(stream);
    REQUIRE(observed[0]);
    REQUIRE(observed[1]);

    REQUIRE(set.Erase(input.Data(), aclco::Extent<std::size_t>{1}, stream) == 0);
    set.Contains(keys.Data(), output.Data(), aclco::Extent<std::size_t>{queries.size()}, stream);
    output.CopyToHostAsync(observed.data(), observed.size(), stream);
    aclco::test::Sync(stream);
    REQUIRE(observed[0]);
    REQUIRE_FALSE(observed[1]);
}
