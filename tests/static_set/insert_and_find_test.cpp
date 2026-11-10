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
#include "tests/common/set_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"
#include "hash_functions.h"

TEMPLATE_TEST_CASE_SIG(
  "static_set insert_and_find correctness",
  "[static_set][insert_and_find]",
  ((typename K, int BucketSize, typename ProbeScheme), K, BucketSize, ProbeScheme),
  (uint32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (int32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (int32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (aclco::fp16_t, 1, aclco::test::set_factory::DoubleHashing<uint16_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::LinearProbing<uint16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbeScheme;
  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, Probe>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("insert_and_find correctness", Key, BS, capacity, n, params, Probe);

  SECTION("insert_and_find with unique keys") {
    PRINT_SECTION("insert_and_find with unique keys");
    CAPTURE(seed, n, capacity, BS);
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Key> dKeys(n);
    aclco::test::DeviceBuffer<Key> dOutputFind(n);
    aclco::test::DeviceBuffer<unsigned char> dOutputInsert(n);

    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    set.InsertAndFind(static_cast<void*>(dKeys.Data()),
                      static_cast<void*>(dOutputFind.Data()),
                      static_cast<void*>(dOutputInsert.Data()),
                      aclco::Extent<std::size_t>(n), stream);

    auto hostOutputFind = dOutputFind.CopyToHost(stream);
    auto hostOutputInsert = dOutputInsert.CopyToHost(stream);

    auto golden = aclco::test::GoldenInsertAndFind<Key>(hostKeys, sent, capacity);

    REQUIRE_PRINT(hostOutputFind.size() == golden.size());
    REQUIRE_PRINT(hostOutputInsert.size() == golden.size());

    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE_PRINT(hostOutputFind[i] == golden[i].first);
      REQUIRE_PRINT((hostOutputInsert[i] != 0) == golden[i].second);
    }
  }

  SECTION("insert_and_find duplicate keys") {
    PRINT_SECTION("insert_and_find duplicate keys");
    std::size_t uniqueN = static_cast<std::size_t>(0.1 * capacity);
    auto uniqueKeys = aclco::test::MakeExamples<Key>(seed, uniqueN, sent, "uniform", true);
    if (uniqueKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
    }

    std::size_t dupN = std::min(n, uniqueN * 2);
    auto dupKeys = aclco::test::MakeDuplicateExamples<Key>(uniqueKeys, dupN, sent, seed);

    CAPTURE(seed, n, capacity, BS, uniqueN, dupN);

    aclco::test::DeviceBuffer<Key> dUniqueKeys(uniqueN);
    aclco::test::DeviceBuffer<Key> dUniqueOutputFind(uniqueN);
    aclco::test::DeviceBuffer<unsigned char> dUniqueOutputInsert(uniqueN);

    dUniqueKeys.CopyFromHostAsync(uniqueKeys.data(), uniqueN, stream);
    //先插入并查找唯一键
    set.InsertAndFind(static_cast<void*>(dUniqueKeys.Data()),
                      static_cast<void*>(dUniqueOutputFind.Data()),
                      static_cast<void*>(dUniqueOutputInsert.Data()),
                      aclco::Extent<std::size_t>(uniqueN), stream);

    auto uniqueOutputFind = dUniqueOutputFind.CopyToHost(stream);
    auto uniqueOutputInsert = dUniqueOutputInsert.CopyToHost(stream);

    auto goldenUnique = aclco::test::GoldenInsertAndFind<Key>(uniqueKeys, sent, capacity);

    for (std::size_t i = 0; i < uniqueN; ++i) {
      REQUIRE_PRINT(uniqueOutputFind[i] == goldenUnique[i].first);
      REQUIRE_PRINT((uniqueOutputInsert[i] != 0) == goldenUnique[i].second);
    }

    std::vector<Key> allKeys;
    allKeys.reserve(uniqueN + dupN);
    allKeys.insert(allKeys.end(), uniqueKeys.begin(), uniqueKeys.end());
    allKeys.insert(allKeys.end(), dupKeys.begin(), dupKeys.end());

    aclco::test::DeviceBuffer<Key> dDupKeys(dupN);
    aclco::test::DeviceBuffer<Key> dDupOutputFind(dupN);
    aclco::test::DeviceBuffer<unsigned char> dDupOutputInsert(dupN);

    dDupKeys.CopyFromHostAsync(dupKeys.data(), dupN, stream);

    //根据唯一键生成重复键并进行插入和查找
    set.InsertAndFind(static_cast<void*>(dDupKeys.Data()),
                      static_cast<void*>(dDupOutputFind.Data()),
                      static_cast<void*>(dDupOutputInsert.Data()),
                      aclco::Extent<std::size_t>(dupN), stream);

    auto dupOutputFind = dDupOutputFind.CopyToHost(stream);
    auto dupOutputInsert = dDupOutputInsert.CopyToHost(stream);

    auto goldenAll = aclco::test::GoldenInsertAndFind<Key>(allKeys, sent, capacity);

    for (std::size_t i = 0; i < dupN; ++i) {
      REQUIRE_PRINT(dupOutputFind[i] == goldenAll[uniqueN + i].first);
      REQUIRE_PRINT((dupOutputInsert[i] != 0) == goldenAll[uniqueN + i].second);
    }
  }
}

TEST_CASE("StaticSet InsertAndFind rejects the reserved empty key", "[static_set][insert_and_find][empty_key]")
{
  using Key = std::uint32_t;
  using Extent = aclco::Extent<std::size_t>;
  const Key emptyKey = GENERATE(Key{0}, std::numeric_limits<Key>::max());
  const bool async = GENERATE(false, true);
  constexpr Key validKey = 12345;
  CAPTURE(emptyKey, async);

  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard guard;
  auto stream = guard.stream;
  aclco::StaticSet<Key> table{Extent{1000}, emptyKey};
  const Key query[] = {emptyKey, validKey};
  aclco::test::DeviceBuffer<Key> keys(2), found(2);
  aclco::test::DeviceBuffer<unsigned char> inserted(2), present(2);
  keys.CopyFromHostAsync(query, 2, stream);

  if (async) {
    table.InsertAndFindAsync(keys.Data(), found.Data(), inserted.Data(), Extent{2}, stream);
  } else {
    table.InsertAndFind(keys.Data(), found.Data(), inserted.Data(), Extent{2}, stream);
  }
  const auto insertFlags = inserted.CopyToHost(stream);
  const auto insertValues = found.CopyToHost(stream);
  CHECK(insertFlags[0] == 0);
  CHECK(insertFlags[1] == 1);
  CHECK(insertValues[0] == emptyKey);
  CHECK(insertValues[1] == validKey);

  table.Find(keys.Data(), found.Data(), Extent{2}, stream);
  table.Contains(keys.Data(), present.Data(), Extent{2}, stream);
  const auto foundValues = found.CopyToHost(stream);
  const auto presentFlags = present.CopyToHost(stream);
  CHECK(foundValues[0] == emptyKey);
  CHECK(foundValues[1] == validKey);
  CHECK(presentFlags[0] == 0);
  CHECK(presentFlags[1] == 1);
  CHECK(table.Count(keys.Data(), Extent{1}, stream) == 0);
}
