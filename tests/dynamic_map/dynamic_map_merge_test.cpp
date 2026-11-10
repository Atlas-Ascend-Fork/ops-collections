/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "dynamic_map.h"
#include "pair.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
}

TEMPLATE_TEST_CASE_SIG("dynamic_map multi non-empty submap merge", "[dynamic_map][merge][find][contains]",
                       ((typename K, typename V, int BS), K, V, BS), (uint32_t, uint32_t, 1), (uint64_t, uint64_t, 1))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;

    using Key = K;
    using Value = V;
    auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

    using DMap = aclco::DynamicMap<Key, Value, aclco::Extent<std::size_t>, aclco::EqualTo<Key>,
                                   aclco::LinearProbing<aclco::murmurhash3_32<Key>>, aclco::Storage<BS>>;

    std::size_t initCap = 40000;
    std::size_t per = 14000;
    std::size_t total = per * 2;

    auto all = aclco::test::MakeExamples<Key, Value>(/*seed=*/7u, total, sent, "uniform", /*unique=*/true);
    if (all.size() < total)
    {
        SKIP("not enough unique pairs for Key type");
    }

    std::unordered_map<Key, Value> golden;
    golden.reserve(total);
    for (auto& p : all) golden[p.first] = p.second;

    DMap map(aclco::Extent<std::size_t>(initCap), sent.emptyKey, sent.emptyValue, sent.erasedKey, {}, {}, {}, stream);

    aclco::test::DeviceBuffer<Pair<Key, Value>> dA(per);
    dA.CopyFromHostAsync(all.data(), per, stream);
    auto insA = map.Insert(static_cast<void*>(dA.Data()), aclco::Extent<std::size_t>(per), stream);
    REQUIRE(insA == static_cast<typename DMap::SizeType>(per));

    aclco::test::DeviceBuffer<Pair<Key, Value>> dB(per);
    dB.CopyFromHostAsync(all.data() + per, per, stream);
    auto insB = map.Insert(static_cast<void*>(dB.Data()), aclco::Extent<std::size_t>(per), stream);
    REQUIRE(insB == static_cast<typename DMap::SizeType>(per));

    REQUIRE(map.NumSubmaps() >= 2);
    REQUIRE(map.Size() == static_cast<typename DMap::SizeType>(total));

    std::vector<Key> keys(total);
    for (std::size_t i = 0; i < total; ++i) keys[i] = all[i].first;
    aclco::test::DeviceBuffer<Key> dKeys(total);
    dKeys.CopyFromHostAsync(keys.data(), total, stream);
    aclco::test::DeviceBuffer<Value> dVals(total);
    map.Find(static_cast<void*>(dKeys.Data()), static_cast<void*>(dVals.Data()), aclco::Extent<std::size_t>(total),
             stream);
    std::vector<Value> outVals(total);
    dVals.CopyToHostAsync(outVals.data(), total, stream);
    aclrtSynchronizeStream(stream);
    std::size_t findErr = 0;
    for (std::size_t i = 0; i < total; ++i)
        if (outVals[i] != golden[keys[i]]) ++findErr;
    REQUIRE(findErr == 0);

    std::vector<Key> q = keys;
    auto absent = aclco::test::MakeExamples<Key, Value>(/*seed=*/9999u, total, sent, "uniform", true);
    for (auto& p : absent)
        if (!golden.count(p.first)) q.push_back(p.first);
    std::size_t qn = q.size();
    aclco::test::DeviceBuffer<Key> dQ(qn);
    dQ.CopyFromHostAsync(q.data(), qn, stream);
    aclco::test::DeviceBuffer<uint8_t> dC(qn);
    map.Contains(static_cast<void*>(dQ.Data()), static_cast<void*>(dC.Data()), aclco::Extent<std::size_t>(qn), stream);
    std::vector<uint8_t> outC(qn);
    dC.CopyToHostAsync(outC.data(), qn, stream);
    aclrtSynchronizeStream(stream);
    std::size_t cErr = 0;
    for (std::size_t i = 0; i < qn; ++i)
    {
        bool exp = golden.count(q[i]) > 0;
        if ((outC[i] != 0) != exp) ++cErr;
    }
    REQUIRE(cErr == 0);
}

TEST_CASE("dynamic_map Find merges later hits with a floating-point empty value", "[dynamic_map][merge][float_sentinel]")
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    using Key = uint32_t;
    using Extent = aclco::Extent<std::size_t>;
    using Map = aclco::DynamicMap<Key, float, Extent, aclco::EqualTo<Key>,
                                 aclco::LinearProbing<aclco::murmurhash3_32<Key>>, aclco::Storage<1>>;
    const auto sent = aclco::test::MakeDefaultSentinels<Key, float>();
    const bool useNaN = GENERATE(false, true);
    const auto empty = useNaN ? std::numeric_limits<float>::quiet_NaN() : -999.0f;
    CAPTURE(useNaN);
    Map map(Extent{40000}, sent.emptyKey, empty, sent.erasedKey, {}, {}, {}, guard.stream);

    const std::array<Pair<Key, float>, 2> pairs{{{7, 11.0f}, {42, 42.0f}}};
    aclco::test::DeviceBuffer<Pair<Key, float>> dPairs(pairs.size());
    dPairs.CopyFromHostAsync(pairs.data(), pairs.size(), guard.stream);
    REQUIRE(map.Insert(dPairs.Data(), Extent{1}, guard.stream) == 1);
    REQUIRE(map.NumSubmaps() == 1);
    map.Reserve(20000, guard.stream);
    REQUIRE(map.NumSubmaps() == 2);
    // The keys are distinct. Append the second key directly to the new active submap.
    REQUIRE(map.Insert(dPairs.Data() + 1, Extent{1}, guard.stream, Map::InsertMode::kAppendUnique) == 1);

    constexpr std::array<Key, 3> keys{7, 42, 999};
    std::array<float, keys.size()> values{};
    aclco::test::DeviceBuffer<Key> dKeys(keys.size());
    aclco::test::DeviceBuffer<float> dValues(keys.size());
    dKeys.CopyFromHostAsync(keys.data(), keys.size(), guard.stream);
    map.Find(dKeys.Data(), dValues.Data(), Extent{keys.size()}, guard.stream);
    dValues.CopyToHostAsync(values.data(), values.size(), guard.stream);
    aclco::test::Sync(guard.stream);

    CHECK(values[0] == 11.0f);
    CHECK(values[1] == 42.0f);
    CHECK((useNaN ? std::isnan(values[2]) : values[2] == empty));
}
