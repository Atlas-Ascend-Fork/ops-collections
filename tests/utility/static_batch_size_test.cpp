/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "static_map.h"
#include "static_set.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace {
using Key = std::uint32_t;
using Map = aclco::StaticMap<Key, Key>;
using Set = aclco::StaticSet<Key>;
using Ext = aclco::Extent<std::size_t>;

// On the unguarded baseline this count truncates to one, so one device slot is sufficient.
constexpr Ext oversized{(std::uint64_t{1} << 32) + 1};

struct NoOp {
  COLLECTION_SIMT_DEVICE explicit NoOp(__gm__ uint8_t*) {}

  template <typename T>
  COLLECTION_SIMT_DEVICE void operator()(T) const noexcept {}
};
}

TEMPLATE_TEST_CASE("Static hash containers reject oversized batches", "[oversized_batch]", Map, Set)
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard streamGuard;
  auto stream = streamGuard.stream;
  auto table = [] {
    if constexpr (std::is_same_v<TestType, Map>) {
      return TestType{Ext{32}, 0, 0};
    } else {
      return TestType{Ext{32}, 0};
    }
  }();
  aclco::test::DeviceBuffer<typename TestType::ValueType> values(1);
  aclco::test::DeviceBuffer<Key> keys(1), output(1);
  aclco::test::DeviceBuffer<bool> flags(1);
  const Key key = 1;
  typename TestType::ValueType value{};
  if constexpr (std::is_same_v<TestType, Map>) {
    value = {key, 7};
  } else {
    value = key;
  }
  values.CopyFromHostAsync(&value, 1, stream);
  keys.CopyFromHostAsync(&key, 1, stream);

  CHECK_THROWS_AS(table.Insert(values.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS(table.InsertAsync(values.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS((table.template InsertIf<Key, aclco::AlwaysTrue>(
    values.Data(), keys.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS((table.template InsertIfAsync<Key, aclco::AlwaysTrue>(
    values.Data(), keys.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS(table.InsertAndFind(values.Data(), output.Data(), flags.Data(), oversized, stream),
                  std::length_error);
  CHECK_THROWS_AS(table.InsertAndFindAsync(values.Data(), output.Data(), flags.Data(), oversized, stream),
                  std::length_error);
  CHECK_THROWS_AS(table.Find(keys.Data(), output.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS(table.FindAsync(keys.Data(), output.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS((table.template FindIf<Key, aclco::AlwaysTrue>(
    keys.Data(), keys.Data(), output.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS((table.template FindIfAsync<Key, aclco::AlwaysTrue>(
    keys.Data(), keys.Data(), output.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS(table.Contains(keys.Data(), flags.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS(table.ContainsAsync(keys.Data(), flags.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS((table.template ContainsIf<Key, aclco::AlwaysTrue>(
    keys.Data(), keys.Data(), flags.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS((table.template ContainsIfAsync<Key, aclco::AlwaysTrue>(
    keys.Data(), keys.Data(), flags.Data(), oversized, stream)), std::length_error);
  CHECK_THROWS_AS(table.template ForEach<NoOp>(keys.Data(), oversized, nullptr, stream), std::length_error);
  CHECK_THROWS_AS(table.template ForEachAsync<NoOp>(keys.Data(), oversized, nullptr, stream), std::length_error);
  CHECK_THROWS_AS(table.Count(keys.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS(table.Count(keys.Data(), Ext{std::uint64_t{1} << 32}, stream), std::length_error);
  CHECK_THROWS_AS(table.Erase(keys.Data(), oversized, stream), std::length_error);
  CHECK_THROWS_AS(table.EraseAsync(keys.Data(), oversized, stream), std::length_error);

  if constexpr (std::is_same_v<TestType, Map>) {
    typename TestType::SizeType assigned = 0;
    CHECK_THROWS_AS(table.InsertOrAssign(values.Data(), oversized, stream), std::length_error);
    CHECK_THROWS_AS(table.InsertOrAssign(values.Data(), oversized, stream, assigned), std::length_error);
    CHECK_THROWS_AS(table.InsertOrAssignAsync(values.Data(), oversized, stream), std::length_error);
    CHECK_THROWS_AS((table.template AssignIfAsync<Key, aclco::AlwaysTrue>(
      values.Data(), keys.Data(), oversized, stream)), std::length_error);
  }
  aclco::test::Sync(stream);
}

TEST_CASE("StaticSet rejects a representable batch whose grid stride would wrap", "[batch_stride]")
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  using Ext32 = aclco::Extent<std::uint32_t>;
  aclco::StaticSet<Key, Ext32> set{Ext32{32}, 0};
  aclco::test::DeviceBuffer<Key> keys(1);
  CHECK_THROWS_AS(set.Count(keys.Data(), Ext32{std::numeric_limits<std::uint32_t>::max()}, stream.stream),
                  std::length_error);
}
