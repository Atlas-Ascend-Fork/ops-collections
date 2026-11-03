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
#include <limits>
#include <string>
#include <vector>
#include <unordered_set>
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/map_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h" 
#include "tests/common/dump_table.h"

TEMPLATE_TEST_CASE("arg storage asynchronous zero reset", "[arg_storage]", uint32_t, uint64_t)
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard guard;
  aclco::ArgStorage<TestType, TestType> counter;
  for (auto stream : {aclrtStream(nullptr), guard.stream}) {
    auto value = std::numeric_limits<TestType>::max();
    counter.Initialize(value, stream);
    REQUIRE(counter.LoadToHost(stream) == value);
    counter.InitializeZeroAsync(stream);
    aclco::test::Sync(stream);
    REQUIRE(counter.LoadToHost(stream) == 0);
    counter.Initialize(TestType(37), stream);
    REQUIRE(counter.LoadToHost(stream) == 37);
  }
}
namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename Key, typename Value>
std::size_t ExpectedNum(const std::vector<aclco::Pair<Key, Value>>& hostPairs, std::vector<Key> keys)
{
  std::size_t count = 0;
  std::unordered_set<Key> keySet;
  keySet.reserve(hostPairs.size());
  for(const auto& pair: hostPairs) {
    keySet.insert(pair.first);
  }
  for (auto i = 0; i < keys.size(); ++i) {
    Key currentKey = keys[i];
    if (keySet.find(currentKey) != keySet.end()) {
      ++count;
    }
  }
  return count;
}

template <typename Key, typename Value>
std::vector<Key> PickKeys(const std::vector<aclco::Pair<Key, Value>>& hostPairs, std::size_t startIdx, std::size_t countN)
{
  std::vector<Key> keys;
  keys.reserve(countN);
  for (std::size_t i = startIdx; i < hostPairs.size()&& i < startIdx + countN; ++i)
  {
    keys.push_back(hostPairs[i].first);
  }
  return keys;

}

} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map count correctness",
  "[static_map][countCorrect]",
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

  std::size_t capacity = GENERATE(164u, 100000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("count correctness", Key, Value, BS, capacity, n, params, Probe);

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

  SECTION("count keys in map") {
    PRINT_SECTION("count keys in map");

    std::size_t countN = static_cast<std::size_t>(0.5 * n);
    std::size_t startIdx = static_cast<std::size_t>(0.1 * n);
  
    auto hostKeys = PickKeys<Key, Value>(hostPairs, startIdx, countN);
    auto expected = ExpectedNum<Key, Value>(hostPairs, hostKeys);

    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

    auto result = map.Count(static_cast<void*>(dKeys.Data()),
                            aclco::Extent<std::size_t>(hostKeys.size()),
                            stream);

    REQUIRE_PRINT(result == expected);
  }

  SECTION("count keys not in map") {
    PRINT_SECTION("count keys not in map");
    std::vector<Key> hostKeys;
    std::size_t countN = static_cast<std::size_t>(0.5 * n);
    hostKeys.reserve(countN);
    for (std::size_t i = 1; i < countN; ++i) {
      hostKeys.push_back(static_cast<Key>(10000000000U + i));
    }

    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

    auto expected = ExpectedNum<Key, Value>(hostPairs, hostKeys);
    auto result = map.Count(static_cast<void*>(dKeys.Data()),
                            aclco::Extent<std::size_t>(hostKeys.size()),
                            stream);

    REQUIRE_PRINT(result == expected);
  }
}
 TEMPLATE_TEST_CASE_SIG(
  "static_map count negative tests",
  "[static_map][countNegative]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint64_t, 5),
  (uint64_t, uint32_t, 1),
  (float, float, 1),
  (uint8_t, uint32_t, 1),
  (int8_t, int32_t, 1),
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
  auto n = GENERATE(5u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_WITH_PARAMS("count negative tests", Key, Value, BS, capacity, n, params);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> dPairs(n);
  dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
  REQUIRE_PRINT(fail == 0);

  //重复计数同样的数据
  SECTION("count duplicate keys") {
    PRINT_SECTION("count duplicate keys");
    std::size_t countN = 3;

    std::vector<Key> hostKeys;
    hostKeys.reserve(countN);
    for (std::size_t i = 0; i < countN; ++i) {
      if(!hostPairs.empty()) {
        hostKeys.push_back(hostPairs[0].first);
      }
    }

    aclco::test::DeviceBuffer<Key> dKeys(countN);
    dKeys.CopyFromHostAsync(hostKeys.data(), countN, stream);

    auto expected = ExpectedNum<Key, Value>(hostPairs, hostKeys);

    auto result = map.Count(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(countN), stream);
    REQUIRE_PRINT(result == expected);
  }

  //传入nullptr
  SECTION("count nullptr") {
    PRINT_SECTION("count nullptr");
    std::size_t countN = 1;
    auto result = map.Count(nullptr, aclco::Extent<std::size_t>(countN), stream);
    REQUIRE_PRINT(result == 0);
  }
}
