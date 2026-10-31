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

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/map_factory.h"
#include "tests/common/set_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename Key, typename Value>
std::vector<Key> PickFirstKeys(const std::vector<aclco::Pair<Key, Value>>& hostPairs, std::size_t findN)
{
  std::vector<Key> keys;
  keys.reserve(findN);
  for (std::size_t i = 0; i < hostPairs.size() && keys.size() < findN; ++i)
  {
    keys.push_back(hostPairs[i].first);
  }
  return keys;
}

template <typename Key, typename Value>
std::vector<Value> ExpectedValuesForKeys(const std::vector<aclco::Pair<Key, Value>>& hostPairs,
                                         const std::vector<Key>& keys)
{
  std::unordered_map<Key, Value> pairMap;
  pairMap.reserve(hostPairs.size());
  for (const auto& pair : hostPairs) {
    pairMap[pair.first] = pair.second;
  }

  std::vector<Value> out;
  out.reserve(keys.size());
  for (const auto& k : keys)
  {
    auto it = pairMap.find(k);
    if (it != pairMap.end()) {
    out.push_back(it->second);
    } else {
      out.push_back(aclco::test::DefaultEmptyValue<Value>());
    }
  }
  return out;
}

} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map find correctness",
  "[static_map][findCorrect]",
  ((typename K, typename V, int BucketSize, typename ProbeScheme), K, V, BucketSize, ProbeScheme),
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

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;
  using Probe = ProbeScheme;
  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(157u, 100000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("find correctness", Key, Value, BS, capacity, n, params, Probe);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                        aclco::Extent<std::size_t>(hostPairs.size()),
                        stream);
  REQUIRE_PRINT(fail == 0);

  //全部都能找到
  SECTION("normal test find all") {
  PRINT_SECTION("normal test find all");
  
  std::size_t findN = static_cast<std::size_t>(0.5 * n);

  auto hostKeys = PickFirstKeys<Key, Value>(hostPairs, findN);
  auto expected  = ExpectedValuesForKeys<Key, Value>(hostPairs, hostKeys);

  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<Value> dVals(hostKeys.size());
  dVals.MemsetZero(stream);

  map.Find(static_cast<void*>(dKeys.Data()),
           static_cast<void*>(dVals.Data()),
           aclco::Extent<std::size_t>(hostKeys.size()),
           stream);

  aclco::test::Sync(stream);

  auto got = dVals.CopyToHost(stream);
  REQUIRE_PRINT(got.size() == expected.size());

  for (std::size_t i = 0; i < got.size(); ++i) {
    CAPTURE(i, hostKeys[i]);
    REQUIRE_PRINT(got[i] == expected[i]);
  }
 }

  //全部都找不到
  SECTION("find none") {
    PRINT_SECTION("find none");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }

  //能找到部分键
  SECTION("find some keys") {
    PRINT_SECTION("find some keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    findKeys.push_back(hostPairs[0].first);
    for (std::size_t i = 0; i < findN - 1; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == findN);

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map find negative test",
  "[static_map][findNegative]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5),
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
  auto n = GENERATE(5u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_WITH_PARAMS("find negative test", Key, Value, BS, capacity, n, params);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                        aclco::Extent<std::size_t>(hostPairs.size()),
                        stream);
  REQUIRE_PRINT(fail == 0);

  //查找重复键
  SECTION("find duplicate Keys") {
    PRINT_SECTION("find duplicate Keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(hostPairs[0].first);
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }

  //查找nullptr
  SECTION("find nullptr") {
    PRINT_SECTION("find nullptr");
    std::size_t findN = 1;
    std::vector<Key> findKeys;
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}



namespace
{
template <typename Key, typename Value, int BucketSize = 1>
void VerifyMapLookup(aclrtStream stream)
{
  using Probe = aclco::test::map_factory::LinearProbing<Key>;
  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();
  constexpr std::size_t kCapacity = 1000;
  constexpr std::size_t kInserted = 400;
  constexpr std::size_t kQueries = 800;

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BucketSize, Probe>(kCapacity, sent, stream);

  std::vector<Pair<Key, Value>> pairs;
  for (std::size_t i = 1; i <= kInserted; ++i) {
    pairs.push_back(Pair<Key, Value>{static_cast<Key>(i), static_cast<Value>(i * 17 + 1)});
  }
  aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(pairs.size());
  dPairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
  REQUIRE(map.Insert(dPairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream) == 0);

  std::vector<Key> keys(kQueries);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = static_cast<Key>(i + 1);
  }
  aclco::test::DeviceBuffer<Key> dKeys(keys.size());
  aclco::test::DeviceBuffer<Value> dValues(keys.size());
  aclco::test::DeviceBuffer<uint8_t> dFlags(keys.size());
  dKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

  map.FindAsync(dKeys.Data(), dValues.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
  auto values = dValues.CopyToHost(stream);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const Value expected = (i < kInserted) ? static_cast<Value>(keys[i] * 17 + 1) : sent.emptyValue;
    CAPTURE(i, keys[i]);
    REQUIRE(values[i] == expected);
  }

  map.ContainsAsync(dKeys.Data(), dFlags.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
  auto flags = dFlags.CopyToHost(stream);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    CAPTURE(i, keys[i]);
    REQUIRE(static_cast<bool>(flags[i]) == (i < kInserted));
  }

  REQUIRE(map.Count(dKeys.Data(), aclco::Extent<std::size_t>(keys.size()), stream) == kInserted);
}

template <typename Key, int BucketSize = 1>
void VerifySetLookup(aclrtStream stream)
{
  using Probe = aclco::test::set_factory::LinearProbing<Key>;
  constexpr std::size_t kCapacity = 1000;
  constexpr std::size_t kInserted = 400;
  constexpr std::size_t kQueries = 800;
  const Key emptyKey = aclco::test::DefaultEmptyValue<Key>();

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BucketSize, Probe>(kCapacity, emptyKey, stream);

  std::vector<Key> keys(kQueries);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = static_cast<Key>(i + 1);
  }
  aclco::test::DeviceBuffer<Key> dKeys(keys.size());
  aclco::test::DeviceBuffer<Key> dInserted(kInserted);
  dKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
  dInserted.CopyFromHostAsync(keys.data(), kInserted, stream);
  REQUIRE(set.Insert(dInserted.Data(), aclco::Extent<std::size_t>(kInserted), stream) == 0);

  aclco::test::DeviceBuffer<Key> dValues(kQueries);
  aclco::test::DeviceBuffer<uint8_t> dFlags(kQueries);
  set.FindAsync(dKeys.Data(), dValues.Data(), aclco::Extent<std::size_t>(kQueries), stream);
  auto values = dValues.CopyToHost(stream);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    CAPTURE(i, keys[i]);
    REQUIRE(values[i] == ((i < kInserted) ? keys[i] : emptyKey));
  }

  set.Contains(dKeys.Data(), dFlags.Data(), aclco::Extent<std::size_t>(kQueries), stream);
  auto flags = dFlags.CopyToHost(stream);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    CAPTURE(i, keys[i]);
    REQUIRE(static_cast<bool>(flags[i]) == (i < kInserted));
  }

  REQUIRE(set.Count(dKeys.Data(), aclco::Extent<std::size_t>(kQueries), stream) == kInserted);
}
}  // namespace

TEST_CASE("key-only lookup keeps map payloads and scalar set queries",
          "[static_map][static_set][find][contains][count][regression]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  // Single-slot buckets: the deferred-payload path is compiled out here.
  REQUIRE_NOTHROW(VerifyMapLookup<uint32_t, uint32_t>(stream));
  REQUIRE_NOTHROW(VerifyMapLookup<uint32_t, uint64_t>(stream));
  REQUIRE_NOTHROW(VerifyMapLookup<uint64_t, uint64_t>(stream));
  REQUIRE_NOTHROW(VerifySetLookup<uint32_t>(stream));
  REQUIRE_NOTHROW(VerifySetLookup<uint64_t>(stream));

  // Five-slot buckets: Map<uint32_t, uint64_t> is the one combination here that takes the
  // deferred-payload path, so it must read the key and the payload from the same slot. The
  // others stay on the single-load path and guard against the condition widening by mistake.
  REQUIRE_NOTHROW((VerifyMapLookup<uint32_t, uint64_t, 5>(stream)));
  REQUIRE_NOTHROW((VerifyMapLookup<uint32_t, uint32_t, 5>(stream)));
  REQUIRE_NOTHROW((VerifyMapLookup<uint64_t, uint64_t, 5>(stream)));
  REQUIRE_NOTHROW((VerifySetLookup<uint32_t, 5>(stream)));
  REQUIRE_NOTHROW((VerifySetLookup<uint64_t, 5>(stream)));

  // Three-slot buckets are excluded from the deferred path; keep them covered so a future
  // change to the bucket threshold shows up as a test failure rather than silently.
  REQUIRE_NOTHROW((VerifyMapLookup<uint32_t, uint64_t, 3>(stream)));
  REQUIRE_NOTHROW((VerifySetLookup<uint32_t, 3>(stream)));
}
