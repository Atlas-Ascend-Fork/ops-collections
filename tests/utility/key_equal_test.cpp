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

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"
#include "tests/common/map_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/set_factory.h"
#include "tests/common/dmap_factory.h"
#include <cmath>
#include <limits>

static_assert(aclco::isBitwiseComparableV<float> && aclco::isBitwiseComparableV<double>);

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename T>
struct UnEqualTo
{
  COLLECTION_HOST_DEVICE constexpr bool operator()(T const& lhs, T const& rhs) const noexcept
  {
    return lhs != rhs;
  }
};
} // namespace

namespace {
struct PaddedKey {
  uint8_t a;
  uint16_t b;
};
static_assert(sizeof(PaddedKey) == 4 && !std::has_unique_object_representations_v<PaddedKey>);
struct PaddedEqual {
  uint8_t mask;
  COLLECTION_HOST_DEVICE bool operator()(PaddedKey lhs, PaddedKey rhs) const noexcept
  {
    return (lhs.a & mask) == (rhs.a & mask) && lhs.b == rhs.b;
  }
};
COLLECTION_SIMT_VF void CheckSentinelCompatibility(__gm__ uint32_t* result)
{
  aclco::EqualWrapper<PaddedKey, PaddedEqual> padded(PaddedEqual{1});
  PaddedKey value{1, 2}, sentinel{3, 2}, different{1, 4};
  result[0] = padded.IsSentinelValue(value, sentinel);
  result[1] = !padded.IsSentinelValue(different, sentinel);
  result[2] = padded.template operator()<aclco::IsInsert::YES>(value, sentinel, sentinel) == aclco::EqualResult::AVAILABLE;
  result[3] = padded.template operator()<aclco::IsInsert::NO>(value, sentinel, sentinel) == aclco::EqualResult::EMPTY;
  aclco::EqualWrapper<bool, aclco::EqualTo<bool>> boolean(aclco::EqualTo<bool>{});
  result[4] = boolean.IsSentinelValue(false, false) && !boolean.IsSentinelValue(true, false);
  uint32_t bits = 0x7fc01234u;
  float nan = *reinterpret_cast<float*>(&bits);
  aclco::EqualWrapper<float, aclco::EqualTo<float>> floating(aclco::EqualTo<float>{});
  result[5] = floating.IsSentinelValue(nan, nan);
  result[6] = floating.EqualTo(nan, nan) == aclco::EqualResult::UNEQUAL;
  result[7] = !floating.IsSentinelValue(-0.0f, 0.0f);
}
COLLECTION_AIV_GLOBAL void CheckSentinelCompatibilityKernel(__gm__ uint32_t* result)
{
  AscendC::Simt::VF_CALL<CheckSentinelCompatibility>(AscendC::Simt::Dim3{1}, result);
}
} // namespace

TEST_CASE("sentinel comparison preserves non-bitwise key compatibility", "[sentinel_compatibility]")
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  aclco::test::DeviceBuffer<uint32_t> result(8);
  CheckSentinelCompatibilityKernel<<<1, 0, stream.stream>>>(result.Data());
  for (auto flag : result.CopyToHost(stream.stream))
    REQUIRE(flag == 1);
}

template <typename K, typename Map>
void CheckNanEmptyMap(Map& map, aclrtStream stream, bool dynamic)
{
  using Extent = aclco::Extent<std::size_t>;
  std::vector<aclco::Pair<K, K>> pairs{{K(1), K(11)}, {K(2), K(22)}};
  std::vector<K> queries{K(1), K(2), K(999)};
  aclco::test::DeviceBuffer<aclco::Pair<K, K>> input(pairs.size());
  aclco::test::DeviceBuffer<K> keys(queries.size()), values(queries.size());
  aclco::test::DeviceBuffer<uint8_t> flags(queries.size());
  input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
  keys.CopyFromHostAsync(queries.data(), queries.size(), stream);
  REQUIRE(map.Insert(input.Data(), Extent(2), stream) == (dynamic ? 2 : 0));
  REQUIRE(map.Insert(input.Data(), Extent(2), stream) == (dynamic ? 0 : 2));
  map.Find(keys.Data(), values.Data(), Extent(3), stream);
  auto found = values.CopyToHost(stream);
  REQUIRE(found[0] == K(11));
  REQUIRE(found[1] == K(22));
  REQUIRE(found[2] == K(-1));
  map.Contains(keys.Data(), flags.Data(), Extent(3), stream);
  REQUIRE(flags.CopyToHost(stream) == std::vector<uint8_t>{1, 1, 0});
}

TEMPLATE_TEST_CASE("maps support NaN empty keys", "[nan_empty_key]", float)
{
  using K = TestType;
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  auto sent = aclco::test::MakeDefaultSentinels<K, K>();
  sent.emptyKey = std::numeric_limits<K>::quiet_NaN();
  sent.emptyValue = K(-1);
  SECTION("StaticMap") {
    auto map = aclco::test::map_factory::MakeStaticMap<K, K, 1,
      aclco::test::map_factory::LinearProbing<K>>(128, sent, stream.stream);
    CheckNanEmptyMap<K>(map, stream.stream, false);
  }
  SECTION("DynamicMap") {
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, K, 1,
      aclco::test::dmap_factory::LinearProbing<K>>(65536, sent, stream.stream);
    CheckNanEmptyMap<K>(map, stream.stream, true);
  }
}

TEMPLATE_TEST_CASE("StaticSet supports NaN empty keys", "[nan_empty_key]", float)
{
  using K = TestType;
  using Extent = aclco::Extent<std::size_t>;
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  auto set = aclco::test::set_factory::MakeStaticSet<K, 1,
    aclco::test::set_factory::LinearProbing<K>>(128, std::numeric_limits<K>::quiet_NaN(), stream.stream);
  std::vector<K> queries{K(1), K(2), K(999)};
  aclco::test::DeviceBuffer<K> keys(3), values(3);
  aclco::test::DeviceBuffer<uint8_t> flags(3);
  keys.CopyFromHostAsync(queries.data(), 3, stream.stream);
  REQUIRE(set.Insert(keys.Data(), Extent(2), stream.stream) == 0);
  REQUIRE(set.Insert(keys.Data(), Extent(2), stream.stream) == 2);
  set.Find(keys.Data(), values.Data(), Extent(3), stream.stream);
  auto found = values.CopyToHost(stream.stream);
  REQUIRE(found[0] == K(1));
  REQUIRE(found[1] == K(2));
  REQUIRE(std::isnan(found[2]));
  set.Contains(keys.Data(), flags.Data(), Extent(3), stream.stream);
  REQUIRE(flags.CopyToHost(stream.stream) == std::vector<uint8_t>{1, 1, 0});
}

TEMPLATE_TEST_CASE_SIG(
  "static_map default key equal",
  "[static_map][default_key_equal]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5)) 
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(10000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  //测试默认KeyEqual
  SECTION("default keyEqual insert") {
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE(fail == 0);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, golden, &diff);
    INFO(diff);
    REQUIRE(ok);
  }
}

  TEMPLATE_TEST_CASE_SIG(
  "static_map un equal key equal",
  "[static_map][un_equal_key_equal]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5)) 
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(10000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>, UnEqualTo<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  //测试UnEqualTo KeyEqual
  SECTION("default unEqual keyEqual insert") {
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE(fail == n);

  }
}

