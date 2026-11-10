/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"
#include "tests/common/map_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"
#include "hash_functions.h"
namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_or_assign correctness",
  "[static_map][insert_or_assign]",
  ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize, ProbingScheme),
  (uint32_t, uint32_t, 1, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::DoubleHashing<uint32_t>),
  (uint64_t, uint64_t, 1, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::LinearProbing<uint64_t>),
  (float, float, 1, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::DoubleHashing<float>),
  (aclco::fp16_t, aclco::fp16_t, 1, aclco::test::map_factory::LinearProbing<aclco::fp16_t>),
  (aclco::fp16_t, aclco::fp16_t, 5, aclco::test::map_factory::DoubleHashing<aclco::fp16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  capacity = map.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("insert_or_assign correctness", Key, Value, BS, capacity, n, params, Probe);

  SECTION("insert_or_assign updates existing keys") {
    PRINT_SECTION("insert_or_assign updates existing keys");
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);

    auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);

    std::vector<Pair<Key, Value>> updatePairs(n);
    for (std::size_t i = 0; i < n; i++) {
      updatePairs[i] = Pair<Key, Value>{hostPairs[i].first, static_cast<Value>(hostPairs[i].second * 2)};
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dUpdatePairs(n);
    dUpdatePairs.CopyFromHostAsync(updatePairs.data(), n, stream);

    auto updateFail = map.InsertOrAssign(static_cast<void*>(dUpdatePairs.Data()),
                                         aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(updateFail == 0);

    auto goldenAfterUpdate = aclco::test::GoldenInsertOrAssign<Key, Value>(golden, updatePairs, sent);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenAfterUpdate, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }

  SECTION("insert_or_assign inserts new keys") {
    PRINT_SECTION("insert_or_assign inserts new keys");
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    std::size_t halfN = n / 2;
    std::vector<Pair<Key, Value>> firstHalf(hostPairs.begin(), hostPairs.begin() + halfN);
    std::vector<Pair<Key, Value>> secondHalf(hostPairs.begin() + halfN, hostPairs.end());

    aclco::test::DeviceBuffer<Pair<Key, Value>> dFirst(halfN);
    dFirst.CopyFromHostAsync(firstHalf.data(), halfN, stream);

    auto fail = map.Insert(static_cast<void*>(dFirst.Data()),
                          aclco::Extent<std::size_t>(halfN), stream);
    REQUIRE_PRINT(fail == 0);

    auto golden = aclco::test::GoldenInsert<Key, Value>(firstHalf, sent);

    aclco::test::DeviceBuffer<Pair<Key, Value>> dSecond(secondHalf.size());
    dSecond.CopyFromHostAsync(secondHalf.data(), secondHalf.size(), stream);

    auto updateFail = map.InsertOrAssign(static_cast<void*>(dSecond.Data()),
                                         aclco::Extent<std::size_t>(secondHalf.size()), stream);
    REQUIRE_PRINT(updateFail == 0);

    auto goldenAfterUpdate = aclco::test::GoldenInsertOrAssign<Key, Value>(golden, secondHalf, sent);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenAfterUpdate, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }

  SECTION("insert_or_assign with duplicate keys") {
    PRINT_SECTION("insert_or_assign with duplicate keys");
    CAPTURE(seed, n, capacity, BS);
    std::size_t uniqueN = static_cast<std::size_t>(0.1 * capacity);
    auto hostPairs = aclco::test::MakeExamplesWithDuplicates<Key, Value>(seed, n, uniqueN, sent);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto updateFail = map.InsertOrAssign(static_cast<void*>(dPairs.Data()),
                                         aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(updateFail == 0);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::unordered_set<Key> uniqueKeys;
    for (auto const& p : hostPairs) {
      uniqueKeys.insert(p.first);
    }
    REQUIRE_PRINT(observed.size() == uniqueKeys.size());

    std::unordered_set<Key> observedKeys;
    for (auto const& p : observed) {
      observedKeys.insert(p.first);
      REQUIRE_PRINT(p.second != sent.emptyValue);
    }
    REQUIRE_PRINT(observedKeys.size() == uniqueKeys.size());

    for (auto const& k : uniqueKeys) {
      REQUIRE_PRINT(observedKeys.count(k) == 1);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_or_assign negative tests",
  "[static_map][insert_or_assign]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint32_t, uint64_t, 5),
  (uint64_t, uint32_t, 1),
  (float, float, 1),
  (uint64_t, float, 1),
  (aclco::fp16_t, aclco::fp16_t, 1),
  (aclco::fp16_t, aclco::fp16_t, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = 128;

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_WITH_PARAMS("insert_or_assign negative tests", Key, Value, BS, capacity, 0, params);

  SECTION("test n > capacity") {
    PRINT_SECTION("test n > capacity");
    std::size_t n = capacity + BS;
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.InsertOrAssign(static_cast<void*>(dPairs.Data()),
                                   aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n - map.Capacity());
  }

  SECTION("insert_or_assign nullptr") {
    PRINT_SECTION("insert_or_assign nullptr");
    std::size_t n = static_cast<std::size_t>(0.1 * capacity);
    CAPTURE(seed, n, capacity, BS);
    auto fail = map.InsertOrAssign(nullptr,
                                   aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n);
  }

  SECTION("test n == 0") {
    PRINT_SECTION("test n == 0");
    std::size_t n = 0;
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
    auto fail = map.InsertOrAssign(static_cast<void*>(dPairs.Data()),
                                   aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);
  }
}

TEST_CASE("StaticMap InsertOrAssign counts a reserved empty key as failed",
          "[static_map][insert_or_assign][empty_key]")
{
  using Key = std::uint32_t;
  using Extent = aclco::Extent<std::size_t>;
  const Key emptyKey = GENERATE(Key{0}, std::numeric_limits<Key>::max());
  constexpr Key emptyValue = 222;
  constexpr Key validKey = 12345;
  CAPTURE(emptyKey);

  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard guard;
  auto stream = guard.stream;
  aclco::StaticMap<Key, Key> map{Extent{1000}, emptyKey, emptyValue};
  Pair<Key, Key> input[] = {{emptyKey, 123}, {validKey, 777}};
  aclco::test::DeviceBuffer<Pair<Key, Key>> values(2);
  values.CopyFromHostAsync(input, 2, stream);
  std::size_t assigned = 0;
  CHECK(map.InsertOrAssign(values.Data(), Extent{2}, stream, assigned) == 1);
  CHECK(assigned == 0);

  input[1].second = 888;
  values.CopyFromHostAsync(input, 2, stream);
  CHECK(map.InsertOrAssign(values.Data(), Extent{2}, stream, assigned) == 1);
  CHECK(assigned == 1);

  const Key query[] = {emptyKey, validKey};
  aclco::test::DeviceBuffer<Key> keys(2), found(2);
  keys.CopyFromHostAsync(query, 2, stream);
  map.Find(keys.Data(), found.Data(), Extent{2}, stream);
  const auto output = found.CopyToHost(stream);
  CHECK(output[0] == emptyValue);
  CHECK(output[1] == input[1].second);
}
