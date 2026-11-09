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
#include <type_traits>
#include <utility>
#include <vector>

#include "static_map.h"
#include "static_set.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace {
using Key = std::uint32_t;
using Pair = aclco::Pair<Key, std::uint32_t>;
using Equal = aclco::EqualTo<Key>;
using Probing = aclco::LinearProbing<aclco::murmurhash3_32<Key>>;
using Store = aclco::Storage<1>;

// Inspect the actual production member without adding a public testing accessor.
// A protected member may only be named through an object of the derived type
// (Bisheng/Clang 15 rejects `decltype(Base::argStorage_)` outright), so route the
// inspection through a CounterProbe member function.
template <typename Value, typename Ext>
struct CounterProbe : aclco::OpenAddressingImpl<Key, Value, Ext, Equal, Probing, Store> {
  static auto& CounterOf(CounterProbe& probe) { return probe.argStorage_; }
};

template <typename Value, typename Ext>
using ProbeCounterStorage = std::remove_reference_t<
    decltype(CounterProbe<Value, Ext>::CounterOf(std::declval<CounterProbe<Value, Ext>&>()))>;

struct Selected {
  COLLECTION_SIMT_DEVICE bool operator()(std::uint32_t value) const noexcept
  {
    return value != 0;
  }
};

template <typename Value, typename Ext>
void CheckCounterStorage()
{
  using Counter = ProbeCounterStorage<Value, Ext>;
  static_assert(std::is_same_v<typename Counter::ValueType, std::uint32_t>,
                "kernel atomics require a uint32_t counter even for narrow Extent");
  static_assert(sizeof(typename Counter::ValueType) == 4, "counter must occupy four bytes");
  static_assert(std::is_same_v<typename Counter::SizeType, std::uint32_t>,
                "counter readback must not use the container's narrow size type");
  Counter counter;
  REQUIRE(counter.Capacity() == 4);
}

template <typename Container, typename Ext>
void CheckOperations(Container& container, const std::vector<typename Container::ValueType>& values,
                     aclrtStream stream)
{
  aclco::test::DeviceBuffer<typename Container::ValueType> deviceValues(values.size());
  deviceValues.CopyFromHostAsync(values.data(), values.size(), stream);
  const std::vector<Key> keys{1, 2, 3, 4};
  const std::vector<Key> missing{99, 100};
  const std::vector<std::uint32_t> selectNew{0, 0, 1, 1};
  const std::vector<std::uint32_t> selectNone{0, 0, 0, 0};
  aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
  aclco::test::DeviceBuffer<Key> deviceMissing(missing.size());
  aclco::test::DeviceBuffer<std::uint32_t> deviceSelected(selectNew.size());
  aclco::test::DeviceBuffer<std::uint32_t> deviceNone(selectNone.size());
  deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
  deviceMissing.CopyFromHostAsync(missing.data(), missing.size(), stream);
  deviceSelected.CopyFromHostAsync(selectNew.data(), selectNew.size(), stream);
  deviceNone.CopyFromHostAsync(selectNone.data(), selectNone.size(), stream);
  aclco::test::Sync(stream);

  // Insert returns failures: duplicate insertions must produce a nonzero counter.
  REQUIRE(container.Insert(deviceValues.Data(), Ext{2}, stream) == 0);
  REQUIRE(container.Insert(deviceValues.Data(), Ext{2}, stream) == 2);
  REQUIRE(container.Count(deviceKeys.Data(), Ext{4}, stream) == 2);
  REQUIRE(container.Count(deviceMissing.Data(), Ext{2}, stream) == 0);

  // Reset a nonzero Count result before InsertIf, then reset duplicate failures.
  REQUIRE(container.Count(deviceKeys.Data(), Ext{4}, stream) == 2);
  const auto inserted = container.template InsertIf<std::uint32_t, Selected>(
    deviceValues.Data(), deviceSelected.Data(), Ext{4}, stream);
  REQUIRE(inserted == 0);
  const auto duplicates = container.template InsertIf<std::uint32_t, Selected>(
    deviceValues.Data(), deviceSelected.Data(), Ext{4}, stream);
  REQUIRE(duplicates == 2);
  const auto filtered = container.template InsertIf<std::uint32_t, Selected>(
    deviceValues.Data(), deviceNone.Data(), Ext{4}, stream);
  REQUIRE(filtered == 0);
  REQUIRE(container.Count(deviceKeys.Data(), Ext{4}, stream) == 4);

  // Erase also returns failures. A successful batch must clear prior miss counts.
  REQUIRE(container.Erase(deviceMissing.Data(), Ext{2}, stream) == 2);
  REQUIRE(container.Erase(deviceKeys.Data(), Ext{4}, stream) == 0);
  REQUIRE(container.Erase(deviceKeys.Data(), Ext{4}, stream) == 4);
  REQUIRE(container.Count(deviceKeys.Data(), Ext{4}, stream) == 0);
  REQUIRE(container.Insert(deviceValues.Data(), Ext{4}, stream) == 0);
  REQUIRE(container.Count(deviceKeys.Data(), Ext{4}, stream) == 4);
}
} // namespace

TEMPLATE_TEST_CASE("StaticMap counter uses four bytes with narrow Extent",
                   "[static_map][counter][narrow_extent]", std::uint8_t, std::uint16_t)
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  using Ext = aclco::Extent<TestType>;
  CheckCounterStorage<Pair, Ext>();
  aclco::StaticMap<Key, std::uint32_t, Ext, Equal, Probing, Store> map{Ext{32}, 0, 0};
  CheckOperations<decltype(map), Ext>(map, {{1, 11}, {2, 22}, {3, 33}, {4, 44}}, stream.stream);
}

TEMPLATE_TEST_CASE("StaticSet counter uses four bytes with narrow Extent",
                   "[static_set][counter][narrow_extent]", std::uint8_t, std::uint16_t)
{
  aclco::test::AclGlobalGuard acl;
  aclco::test::AclStreamGuard stream;
  using Ext = aclco::Extent<TestType>;
  CheckCounterStorage<Key, Ext>();
  aclco::StaticSet<Key, Ext, Equal, Probing, Store> set{Ext{32}, 0};
  CheckOperations<decltype(set), Ext>(set, {1, 2, 3, 4}, stream.stream);
}
