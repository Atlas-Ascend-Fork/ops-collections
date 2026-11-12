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
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pair.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/map_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/matchers.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

namespace aclco {
inline std::ostream& operator<<(std::ostream& os, Pair<uint32_t, uint32_t> const& p)
{
  return os << '{' << p.first << ", " << p.second << '}';
}
} // namespace aclco

TEMPLATE_TEST_CASE_SIG(
  "static_map clear correctness (sync Clear)",
  "[static_map][clear]",
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

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PARAMS("clear correctness (sync Clear)", Key, Value, BS, capacity, n, params);

  //不插入，直接清空
  SECTION("directly clear") {
    PRINT_SECTION("directly clear");
    map.Clear(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::unordered_map<Key, Value> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
  
  // 插入后多次清空
  SECTION("multiple clear") {
    PRINT_SECTION("multiple clear");
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

    map.Clear(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::unordered_map<Key, Value> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);

    map.Clear(stream);
    aclco::test::Sync(stream);

    bool ok2 = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok2);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map clear correctness (async ClearAsync)",
  "[static_map][clearAsync]",
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

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PARAMS("clear correctness (async ClearAsync)", Key, Value, BS, capacity, n, params);

  //不插入，直接清空
  SECTION("directly clear") {
    PRINT_SECTION("directly clear");
    map.ClearAsync(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::unordered_map<Key, Value> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
  
  // 插入后多次清空
  SECTION("multiple clear") {
    PRINT_SECTION("multiple clear");
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

    map.ClearAsync(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

    std::unordered_map<Key, Value> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);

    map.ClearAsync(stream);
    aclco::test::Sync(stream);

    bool ok2 = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok2);
  }
}

TEST_CASE("ClearSIMDU32 fills interleaved pattern including the tail tile",
          "[static_map][clear][clear_tail]")
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  constexpr uint32_t tileSize = 64;
  const uint32_t pattern[2] = {0x11111111u, 0x22222222u};
  const auto tail = GENERATE(0u, 32u);
  const auto count = tileSize + (tail == 0 ? tileSize : tail);
  std::vector<uint32_t> slots(count, 17u);
  aclco::test::DeviceBuffer<uint32_t> table(count);
  table.CopyFromHostAsync(slots.data(), count, stream);

  aclco::test::DeviceBuffer<uint32_t> patternBuf(2);
  patternBuf.CopyFromHostAsync(pattern, 2, stream);

  aclco::ClearSIMDU32<<<1, 0, stream>>>(reinterpret_cast<uint8_t*>(table.Data()),
      patternBuf.Data(), 2,
      count, 2, tail, 0, 0, 0, tileSize, 0);
  table.CopyToHostAsync(slots.data(), count, stream);
  aclco::test::Sync(stream);

  for (uint32_t i = 0; i < count; ++i) {
    INFO("slot " << i);
    REQUIRE(slots[i] == pattern[i % 2]);
  }
}

TEST_CASE("static_map 16-byte slot clear keeps simd bandwidth with interleaved pattern",
          "[static_map][clear][clear_u64]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = uint64_t;
  using Value = uint64_t;
  constexpr int BS = 1;

  // 空 sentinel 的低/高 32 位不同，覆盖 ClearSIMDU32 的 32 位字交错填充模式；
  // 大容量（> aivCoreNum * BLOCK_SIZE * sizeof(ValueType)）走 ClearSimd 快路径，
  // 小容量走逐元素路径，两条路径结果必须一致
  aclco::test::Sentinels<Key, Value> sent{};
  constexpr uint64_t sentinel = 0x1234567890abcdefULL;
  sent.emptyKey = sentinel;
  sent.emptyValue = sentinel;

  std::size_t capacity = GENERATE(100u, 1000000u);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  SECTION("sync Clear") {
    map.Clear(stream);
    aclco::test::Sync(stream);
  }
  SECTION("async ClearAsync") {
    map.ClearAsync(stream);
    aclco::test::Sync(stream);
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  std::unordered_map<Key, Value> goldenEmpty;
  std::string diff;
  bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
  INFO(diff);
  REQUIRE(ok);
}

TEST_CASE("ClearSIMDU32 fills 4-word interleaved pattern including the tail tile",
          "[static_map][clear][clear_pattern4]")
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  constexpr uint32_t tileSize = 64;
  const uint32_t pattern[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
  const auto tail = GENERATE(0u, 32u);
  const auto count = tileSize + (tail == 0 ? tileSize : tail);
  std::vector<uint32_t> slots(count, 17u);
  aclco::test::DeviceBuffer<uint32_t> table(count);
  table.CopyFromHostAsync(slots.data(), count, stream);

  aclco::test::DeviceBuffer<uint32_t> patternBuf(4);
  patternBuf.CopyFromHostAsync(pattern, 4, stream);

  aclco::ClearSIMDU32<<<1, 0, stream>>>(reinterpret_cast<uint8_t*>(table.Data()),
      patternBuf.Data(), 4,
      count, 2, tail, 0, 0, 0, tileSize, 0);
  table.CopyToHostAsync(slots.data(), count, stream);
  aclco::test::Sync(stream);

  for (uint32_t i = 0; i < count; ++i) {
    INFO("slot " << i);
    REQUIRE(slots[i] == pattern[i % 4]);
  }
}

TEST_CASE("static_map clear reduces equal 32-bit words to pure duplicate period",
          "[static_map][clear][clear_period]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = uint64_t;
  using Value = uint64_t;
  constexpr int BS = 1;

  // 低/高 32 位字相同，pattern 化简为周期 1 后退化为纯 Duplicate 快路径
  aclco::test::Sentinels<Key, Value> sent{};
  sent.emptyKey = 0x1234567890abcdefULL;
  sent.emptyValue = 0xcafebabeCAFEBABEULL;

  std::size_t capacity = GENERATE(100u, 1000000u);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  SECTION("sync Clear") {
    map.Clear(stream);
    aclco::test::Sync(stream);
  }
  SECTION("async ClearAsync") {
    map.ClearAsync(stream);
    aclco::test::Sync(stream);
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  std::unordered_map<Key, Value> goldenEmpty;
  std::string diff;
  bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
  INFO(diff);
  REQUIRE(ok);
}

TEST_CASE("static_map clear fills both pair halves with emptyValue when emptyKey differs from emptyValue",
          "[static_map][clear][clear_pair_sentinel]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = uint32_t;
  using Value = aclco::Pair<uint32_t, uint32_t>;
  constexpr int BS = 1;

  aclco::test::Sentinels<Key, Value> sent{};
  sent.emptyKey = 0x00000001u;
  sent.emptyValue.first = 0x00000001u;
  sent.emptyValue.second = 0xcafebabeu;

  std::size_t capacity = GENERATE(100u, 1000000u);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  SECTION("sync Clear") {
    map.Clear(stream);
    aclco::test::Sync(stream);
  }
  SECTION("async ClearAsync") {
    map.ClearAsync(stream);
    aclco::test::Sync(stream);
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  std::unordered_map<Key, Value> goldenEmpty;
  std::string diff;
  bool ok = aclco::test::EqualAsMap<Key, Value>(observed, goldenEmpty, &diff);
  INFO(diff);
  REQUIRE(ok);
}