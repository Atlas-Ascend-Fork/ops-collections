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

TEMPLATE_TEST_CASE_SIG(
  "static_set insert correctness",
  "[static_set][insert]",
  ((typename K,  int BucketSize, typename ProbeScheme), K, BucketSize, ProbeScheme),
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
  (aclco::fp16_t, 1, aclco::test::set_factory::LinearProbing<uint16_t>),
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
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("insert correctness", Key, BS, capacity, n, params, Probe);

  SECTION("normal insert unique keys") {
    PRINT_SECTION("normal insert unique keys");
    CAPTURE(seed, n, capacity, BS);
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
    }
    
    aclco::test::DeviceBuffer<Key> dKeys(n);
    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key>(hostKeys, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }

  SECTION("insert after clear") {
    PRINT_SECTION("insert after clear");
    CAPTURE(seed, n, capacity, BS);
    set.Clear(stream);
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
    }
    aclco::test::DeviceBuffer<Key> dKeys(n);

    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);
    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);
    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key>(hostKeys, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set insert negative tests",
  "[static_set][insert]",
  ((typename K,  int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (uint32_t, 5),
  (int32_t, 1),
  (int32_t, 5),
  (uint64_t, 1),
  (uint64_t, 5),
  (float, 1),
  (float, 5),
  (aclco::fp16_t, 1),
  (aclco::fp16_t, 5)) 
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = 128;

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("insert negative tests", Key, BS, capacity, 0, params);

  SECTION("test n > capacity") {
    PRINT_SECTION("test n > capacity");
    std::size_t n = capacity + BS;
    CAPTURE(seed, n, capacity, BS);
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Key> dKeys(n);

    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n - set.Capacity());
  }

  SECTION("insert duplicate keys") {
    PRINT_SECTION("insert duplicate keys");
    std::size_t n = static_cast<std::size_t>(0.2 * capacity);
    std::size_t uniqueN = static_cast<std::size_t>(0.1 * capacity);
    auto hostKeys = aclco::test::MakeExamplesWithDuplicates<Key>(seed, n, uniqueN, sent);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when pair number is bigger than the max exact integer of Key type!");
    }

    CAPTURE(seed, n, capacity, BS);
    aclco::test::DeviceBuffer<Key> dKeys(n);
    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n - uniqueN);
  }

  SECTION("insert nullptr") {
    PRINT_SECTION("insert nullptr");
    std::size_t n = static_cast<std::size_t>(0.1 * capacity);
    CAPTURE(seed, n, capacity, BS);
    auto fail = set.Insert(nullptr, 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n);
  }

  SECTION("test n == 0") {
    PRINT_SECTION("test n == 0");
    std::size_t n = 0;
    CAPTURE(seed, n, capacity, BS);
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Key> dKeys(n);
    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);
    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set insertAsync",
  "[static_set][insertAsync]",
  ((typename K,  int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (uint32_t, 5),
  (int32_t, 1),
  (int32_t, 5),
  (uint64_t, 1),
  (uint64_t, 5),
  (float, 1),
  (float, 5),
  (aclco::fp16_t, 1),
  (aclco::fp16_t, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("insertAysnc correctness", Key, BS, capacity, n, params);

  CAPTURE(seed, n, capacity, BS);
  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
  SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(n);
  dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);
  set.InsertAsync(static_cast<void*>(dKeys.Data()), 
                  aclco::Extent<std::size_t>(n), stream);

  auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
  auto golden   = aclco::test::GoldenInsert<Key>(hostKeys, sent);

  std::string diff;
  bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
  INFO(diff);
  REQUIRE_PRINT(ok);
}

TEST_CASE("StaticSet Insert counts a reserved empty key as failed", "[static_set][insert][empty_key]")
{
  using Key = std::uint32_t;
  using Extent = aclco::Extent<std::size_t>;
  const Key emptyKey = GENERATE(Key{0}, std::numeric_limits<Key>::max());
  constexpr Key validKey = 12345;
  CAPTURE(emptyKey);

  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard guard;
  auto stream = guard.stream;
  aclco::StaticSet<Key> set{Extent{1000}, emptyKey};
  const Key input[] = {emptyKey, validKey};
  aclco::test::DeviceBuffer<Key> keys(2);
  keys.CopyFromHostAsync(input, 2, stream);

  CHECK(set.Insert(keys.Data(), Extent{2}, stream) == 1);
  CHECK(set.Count(keys.Data(), Extent{1}, stream) == 0);
  CHECK(set.Count(keys.Data(), Extent{2}, stream) == 1);
}
