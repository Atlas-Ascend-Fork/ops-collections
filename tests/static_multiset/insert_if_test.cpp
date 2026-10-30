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

using namespace aclco::test::static_multiset;

namespace {
struct IsOdd {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const noexcept { return value % 2u != 0u; }
};
} // namespace

TEMPLATE_TEST_CASE_SIG("static_multiset insert_if correctness", "[static_multiset][insert_if]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto stencil_mode = GENERATE(0u, 1u, 2u);
    auto count = std::min<std::size_t>(capacity / 2u, 4096u);
    auto set = MakeMultiset<Key>(capacity, stream);
    auto keys = MakeRepeatedKeys<Key>(count, 1u);
    auto stencil = MakeStencil(count, stencil_mode);
    aclco::test::DeviceBuffer<Key> d_keys(count);
    aclco::test::DeviceBuffer<uint32_t> d_stencil(count);
    d_keys.CopyFromHostAsync(keys.data(), count, stream);
    d_stencil.CopyFromHostAsync(stencil.data(), count, stream);
    auto failed = set.template InsertIf<uint32_t, IsOdd>(static_cast<void*>(d_keys.Data()), d_stencil.Data(),
                                                         aclco::Extent<std::size_t>(count), stream);
    REQUIRE(failed == 0u);

    aclco::test::DeviceBuffer<unsigned char> d_output(count);
    d_output.MemsetZero(stream);
    set.Contains(d_keys.Data(), d_output.Data(), aclco::Extent<std::size_t>(count), stream);
    auto actual = d_output.CopyToHost(stream);
    for (std::size_t i = 0; i < count; ++i) {
        CAPTURE(i, capacity);
        REQUIRE((actual[i] != 0) == (stencil[i] % 2u != 0u));
    }
}

TEMPLATE_TEST_CASE_SIG("static_multiset insert_if boundary test", "[static_multiset][insert_if][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(128u, sg.stream);
    REQUIRE(set.template InsertIf<uint32_t, IsOdd>(nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream) == 0u);
    REQUIRE(set.Size(sg.stream) == 0u);
}

TEMPLATE_TEST_CASE_SIG("static_multiset small insert_if preserves filtered prefix order",
                       "[static_multiset][insert_if][small]", ((typename Key, int Dummy), Key, Dummy), (int32_t, 0),
                       (int64_t, 0))
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    auto stream = guard.stream;
    for (auto n : {1u, 16u, 31u, 32u, 33u, 256u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u}) {
        for (auto mode : {0u, 1u, 2u}) {
            auto capacity = n / 4u + 3u;
            CAPTURE(sizeof(Key), n, mode);
            auto set = MakeMultiset<Key>(capacity, stream);
            std::vector<Key> keys(n), expected;
            auto stencil = MakeStencil(n, mode);
            std::size_t invalid = 0, valid = 0;
            for (std::size_t i = 0; i < n; ++i) {
                keys[i] = i % 7 == 0 ? EmptyKey<Key>() :
                                       static_cast<Key>((static_cast<int64_t>((i * 7919) % 1009) - 500) * 65537);
                if (stencil[i] % 2 != 0) {
                    if (keys[i] == EmptyKey<Key>()) {
                        ++invalid;
                    } else {
                        if (expected.size() < capacity)
                            expected.push_back(keys[i]);
                        ++valid;
                    }
                }
            }
            aclco::test::DeviceBuffer<Key> input(n), output(capacity);
            aclco::test::DeviceBuffer<uint32_t> mask(n);
            input.CopyFromHostAsync(keys.data(), keys.size(), stream);
            mask.CopyFromHostAsync(stencil.data(), stencil.size(), stream);
            REQUIRE(set.template InsertIf<uint32_t, IsOdd>(input.Data(), mask.Data(), aclco::Extent<std::size_t>{n},
                                                           stream) == invalid + valid - expected.size());
            REQUIRE(set.Size(stream) == expected.size());
            REQUIRE(set.RetrieveAll(output.Data(), capacity, stream) == expected.size());
            auto actual = output.CopyToHost(stream);
            actual.resize(expected.size());
            REQUIRE(actual == expected);
        }
    }
}
