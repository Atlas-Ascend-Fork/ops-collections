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
  "static_map insert_and_find test",
  "[static_map][insert_and_find]",
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

  std::size_t capacity = GENERATE(100u, 10000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  capacity = map.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("insert_and_find test", Key, Value, BS, capacity, n, params, Probe);

  SECTION("insert_and_find with unique keys") {
    PRINT_SECTION("insert_and_find with unique keys");
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    aclco::test::DeviceBuffer<Value> dOutputFind(n);
    aclco::test::DeviceBuffer<unsigned char> dOutputInsert(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    map.InsertAndFind(static_cast<void*>(dPairs.Data()),
                      static_cast<void*>(dOutputFind.Data()),
                      static_cast<void*>(dOutputInsert.Data()),
                      aclco::Extent<std::size_t>(n), stream);

    auto hostOutputFind = dOutputFind.CopyToHost(stream);
    auto hostOutputInsert = dOutputInsert.CopyToHost(stream);

    auto golden = aclco::test::GoldenInsertAndFind<Key, Value>(hostPairs, sent, capacity);

    REQUIRE_PRINT(hostOutputFind.size() == golden.size());
    REQUIRE_PRINT(hostOutputInsert.size() == golden.size());

    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE_PRINT(hostOutputFind[i] == golden[i].first);
      REQUIRE_PRINT((hostOutputInsert[i] !=0) == golden[i].second);
    }
  }

  //测试并行插入和查找是否出现冲突
  SECTION("insert_and_find with duplicate keys") {
    PRINT_SECTION("insert_and_find with duplicate keys");
    std::size_t uniqueN = static_cast<std::size_t>(0.1 * capacity);
    auto hostPairs = aclco::test::MakeExamplesWithDuplicates<Key, Value>(seed, n, uniqueN, sent, "uniform");
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    CAPTURE(seed, n, capacity, BS, uniqueN);

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    aclco::test::DeviceBuffer<Value> dOutputFind(n);
    aclco::test::DeviceBuffer<unsigned char> dOutputInsert(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    map.InsertAndFind(static_cast<void*>(dPairs.Data()),
                      static_cast<void*>(dOutputFind.Data()),
                      static_cast<void*>(dOutputInsert.Data()),
                      aclco::Extent<std::size_t>(n), stream);

    auto hostOutputFind = dOutputFind.CopyToHost(stream);
    auto hostOutputInsert = dOutputInsert.CopyToHost(stream);

    std::unordered_map<K, V> firstValue;
    for (std::size_t i = 0; i < n; ++i) {
      if (hostOutputInsert[i] != 0) {
        firstValue[hostPairs[i].first] = hostOutputFind[i];
      }
    }

    //验证键重复时返回的值是否是同一个，并且都等于insert = 1时对应的hostPairs插入的值
    for (std::size_t i = 0; i < n; ++i) {
      auto it = firstValue.find(hostPairs[i].first);
      if (it != firstValue.end()) {
        REQUIRE_PRINT(hostOutputFind[i] == it->second);
      }
      if (hostOutputInsert[i] == 1) {
        REQUIRE_PRINT(hostOutputFind[i] == hostPairs[i].second);
      }
    }
  }
}

TEST_CASE("StaticMap InsertAndFind rejects the reserved empty key", "[static_map][insert_and_find][empty_key]")
{
  using Key = std::uint32_t;
  using Extent = aclco::Extent<std::size_t>;
  const Key emptyKey = GENERATE(Key{0}, std::numeric_limits<Key>::max());
  const bool async = GENERATE(false, true);
  constexpr Key validKey = 12345;
  constexpr Key emptyValue = 222;
  constexpr Key validValue = 777;
  CAPTURE(emptyKey, async);

  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard guard;
  auto stream = guard.stream;
  aclco::StaticMap<Key, Key> table{Extent{1000}, emptyKey, emptyValue};
  const Key query[] = {emptyKey, validKey};
  const Pair<Key, Key> input[] = {{emptyKey, 123}, {validKey, validValue}};
  aclco::test::DeviceBuffer<Pair<Key, Key>> values(2);
  values.CopyFromHostAsync(input, 2, stream);
  aclco::test::DeviceBuffer<Key> keys(2), found(2);
  aclco::test::DeviceBuffer<unsigned char> inserted(2), present(2);
  keys.CopyFromHostAsync(query, 2, stream);

  if (async) {
    table.InsertAndFindAsync(values.Data(), found.Data(), inserted.Data(), Extent{2}, stream);
  } else {
    table.InsertAndFind(values.Data(), found.Data(), inserted.Data(), Extent{2}, stream);
  }
  const auto insertFlags = inserted.CopyToHost(stream);
  const auto insertValues = found.CopyToHost(stream);
  CHECK(insertFlags[0] == 0);
  CHECK(insertFlags[1] == 1);
  CHECK(insertValues[0] == emptyValue);
  CHECK(insertValues[1] == validValue);

  table.Find(keys.Data(), found.Data(), Extent{2}, stream);
  table.Contains(keys.Data(), present.Data(), Extent{2}, stream);
  const auto foundValues = found.CopyToHost(stream);
  const auto presentFlags = present.CopyToHost(stream);
  CHECK(foundValues[0] == emptyValue);
  CHECK(foundValues[1] == validValue);
  CHECK(presentFlags[0] == 0);
  CHECK(presentFlags[1] == 1);
  CHECK(table.Count(keys.Data(), Extent{1}, stream) == 0);
}
