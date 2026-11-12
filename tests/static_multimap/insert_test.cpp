/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "common/static_multimap_test_common.h"

using namespace aclco::test::static_multimap;

TEMPLATE_TEST_CASE_SIG("static_multimap insert correctness", "[static_multimap][insert]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(128u, 1027u, 8192u, 65536u);
    auto occupancy = GENERATE(0.1, 0.5, 0.9);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto pair_num = static_cast<std::size_t>(capacity * occupancy);
    auto distinct = std::max<std::size_t>(pair_num / multiplicity, 1u);
    CAPTURE(capacity, occupancy, multiplicity, pair_num, distinct);

    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pairs = MakePairs<Key, Value>(distinct, multiplicity);
    if (pairs.size() > pair_num) {
        pairs.resize(pair_num);
    }
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    auto failures = map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);

    REQUIRE(failures == 0u);
    REQUIRE(map.Size(stream) == pairs.size());

    SECTION("identical pair is retained")
    {
        std::vector<aclco::Pair<Key, Value>> repeats(8u, pairs.front());
        aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_repeats(repeats.size());
        d_repeats.CopyFromHostAsync(repeats.data(), repeats.size(), stream);
        auto before = map.Size(stream);
        REQUIRE(map.Insert(d_repeats.Data(), aclco::Extent<std::size_t>(repeats.size()), stream) == 0u);
        REQUIRE(map.Size(stream) == before + repeats.size());
    }

    SECTION("zero input is a no-op")
    {
        auto before = map.Size(stream);
        REQUIRE(map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(0u), stream) == 0u);
        REQUIRE(map.Insert(nullptr, aclco::Extent<std::size_t>(0u), stream) == 0u);
        REQUIRE(map.Size(stream) == before);
    }

    SECTION("nonempty null input is rejected without changing size")
    {
        auto before = map.Size(stream);
        REQUIRE_THROWS_AS(map.Insert(nullptr, aclco::Extent<std::size_t>(3u), stream), std::invalid_argument);
        REQUIRE(map.Size(stream) == before);
    }
}

TEMPLATE_TEST_CASE_SIG("static_multimap insert capacity boundary", "[static_multimap][insert][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    constexpr std::size_t requested = 128u;
    auto map = MakeMultimap<Key, Value>(requested, stream);
    auto capacity = map.Capacity();
    REQUIRE_THROWS_AS(map.Insert(nullptr, aclco::Extent<std::size_t>(1u), stream), std::invalid_argument);
    REQUIRE(map.Size(stream) == 0u);
    auto pairs = MakePairs<Key, Value>(capacity + 5u, 1u);
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    auto failures = map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);
    REQUIRE(failures == pairs.size() - capacity);
    REQUIRE(map.Size(stream) == capacity);
    REQUIRE_THROWS_AS(map.Insert(nullptr, aclco::Extent<std::size_t>(1u), stream), std::invalid_argument);
    REQUIRE(map.Size(stream) == capacity);
}

TEMPLATE_TEST_CASE_SIG("static_multimap insert saturates with one duplicate key", "[static_multimap][insert][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    // Small dense tables force the stride phase: the 32-slot window of one key
    // saturates long before the table does, so the probe walk must still reach
    // every remaining slot as the load approaches capacity.
    constexpr std::size_t requested = 300u;
    auto map = MakeMultimap<Key, Value>(requested, stream);
    auto capacity = map.Capacity();
    std::vector<aclco::Pair<Key, Value>> pairs(capacity, {static_cast<Key>(1), static_cast<Value>(1025)});
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);

    SECTION("capacity-1 duplicates leave room for exactly one more")
    {
        auto failures = map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(capacity - 1u), stream);
        REQUIRE(failures == 0u);
        REQUIRE(map.Size(stream) == capacity - 1u);
        failures = map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(1u), stream);
        REQUIRE(failures == 0u);
        REQUIRE(map.Size(stream) == capacity);
    }
    SECTION("the full table accepts capacity identical pairs")
    {
        auto failures = map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(capacity), stream);
        REQUIRE(failures == 0u);
        REQUIRE(map.Size(stream) == capacity);
    }

    Key probe = static_cast<Key>(1);
    aclco::test::DeviceBuffer<Key> d_probe(1);
    d_probe.CopyFromHostAsync(&probe, 1, stream);
    REQUIRE(map.Count(d_probe.Data(), aclco::Extent<std::size_t>(1u), stream) == capacity);
}
