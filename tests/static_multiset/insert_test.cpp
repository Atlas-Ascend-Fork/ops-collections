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
#include "common/static_multiset_test_common.h"
#include <random>
#include <unordered_map>

using namespace aclco::test::static_multiset;

TEMPLATE_TEST_CASE_SIG("static_multiset insert correctness", "[static_multiset][insert]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(128u, 1027u, 8192u, 65536u);
    auto occupancy = GENERATE(0.1, 0.5, 0.9);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto key_num = static_cast<std::size_t>(capacity * occupancy);
    auto distinct = std::max<std::size_t>(key_num / multiplicity, 1u);
    CAPTURE(capacity, occupancy, multiplicity, key_num);

    auto set = MakeMultiset<Key>(capacity, stream);
    auto keys = MakeRepeatedKeys<Key>(distinct, multiplicity);
    if (keys.size() > key_num) {
        keys.resize(key_num);
    }
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    REQUIRE(set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream) == 0u);
    REQUIRE(set.Size(stream) == keys.size());

    SECTION("duplicate key is retained")
    {
        std::vector<Key> repeats(8u, keys.front());
        aclco::test::DeviceBuffer<Key> d_repeats(repeats.size());
        d_repeats.CopyFromHostAsync(repeats.data(), repeats.size(), stream);
        auto before = set.Size(stream);
        REQUIRE(set.Insert(d_repeats.Data(), aclco::Extent<std::size_t>(repeats.size()), stream) == 0u);
        REQUIRE(set.Size(stream) == before + repeats.size());
    }

    SECTION("zero and null inputs")
    {
        auto before = set.Size(stream);
        REQUIRE(set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(0u), stream) == 0u);
        REQUIRE(set.Size(stream) == before);
        REQUIRE(set.Insert(nullptr, aclco::Extent<std::size_t>(3u), stream) == 3u);
    }
}

TEMPLATE_TEST_CASE_SIG("static_multiset insert capacity boundary", "[static_multiset][insert][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto set = MakeMultiset<Key>(128u, stream);
    auto capacity = set.Capacity();
    auto keys = MakeRepeatedKeys<Key>(capacity + 5u, 1u);
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    auto failures = set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
    REQUIRE(failures == keys.size() - capacity);
    REQUIRE(set.Size(stream) == capacity);
}

TEMPLATE_TEST_CASE_SIG("static_multiset small insert preserves random accepted prefixes",
                       "[static_multiset][insert][small]", ((typename Key, int Dummy), Key, Dummy), (int32_t, 0),
                       (int64_t, 0))
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    auto stream = guard.stream;
    for (auto n : {1u, 16u, 31u, 32u, 33u, 256u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u}) {
        for (auto representation : {0u, 1u, 2u}) {
            auto capacity = n / 2u + 17u;
            CAPTURE(sizeof(Key), n, representation);
            auto set = MakeMultiset<Key>(capacity, stream);
            std::vector<Key> expected;
            if (representation != 0) {
                // Start from either a compact interval or materialized hash storage.
                expected = representation == 1 ? std::vector<Key>{1, 2, 3, 4} :
                                                 std::vector<Key>{-900001, 800003, -900001};
                auto seed = InsertKeys(set, expected, stream);
            }
            std::mt19937_64 rng(114);
            std::vector<Key> keys(n);
            std::size_t valid = 0;
            auto available = capacity - expected.size();
            for (std::size_t i = 0; i < n; ++i) {
                keys[i] = i % 7 == 0 ? EmptyKey<Key>() :
                                       static_cast<Key>((static_cast<int64_t>(rng() % 2001) - 1000) * 104729);
                if (keys[i] != EmptyKey<Key>()) {
                    if (valid < available)
                        expected.push_back(keys[i]);
                    ++valid;
                }
            }
            auto accepted = std::min<std::size_t>(valid, available);
            aclco::test::DeviceBuffer<Key> input(n), output(capacity);
            input.CopyFromHostAsync(keys.data(), keys.size(), stream);
            auto failures = set.Insert(input.Data(), aclco::Extent<std::size_t>{n}, stream);
            REQUIRE(failures == n - accepted);
            REQUIRE(set.Size(stream) == expected.size());
            REQUIRE(set.RetrieveAll(output.Data(), capacity, stream) == expected.size());
            auto actual = output.CopyToHost(stream);
            actual.resize(expected.size());
            REQUIRE(actual == expected);
            std::unordered_map<Key, uint64_t> counts;
            for (auto key : expected)
                ++counts[key];
            std::vector<uint64_t> golden;
            for (auto key : keys)
                golden.push_back(counts[key]);
            aclco::test::DeviceBuffer<uint64_t> result(n);
            set.CountEach(input.Data(), result.Data(), n, stream);
            REQUIRE(result.CopyToHost(stream) == golden);
        }
    }
}
