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
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hash_functions.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/map_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/matchers.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

TEST_CASE("static_map rejects probing scheme with instance state", "[static_map][create]")
{
  aclco::test::AclGlobalGuard g_acl;

  using Key = uint32_t;
  using Value = uint32_t;
  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  REQUIRE_THROWS_AS(
    (aclco::StaticMap<Key, Value>(100, sent.emptyKey, sent.emptyValue, aclco::EqualTo<Key>{},
      aclco::LinearProbing<aclco::murmurhash3_32<Key>>{aclco::murmurhash3_32<Key>{0xDEADBEEFu}})),
    std::invalid_argument);

  auto deflt = aclco::StaticMap<Key, Value>(100, sent.emptyKey, sent.emptyValue);
  REQUIRE(deflt.Capacity() >= 100);
}

TEMPLATE_TEST_CASE_SIG(
  "static_map Key == Value create test",
  "[static_map][sameKVCreate]",
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

  std::size_t capacity = 160000000;
  CAPTURE(capacity, BS);
  PRINT_BEFORE_EXEC("create test", Key, Value, BS, capacity, 0);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);
  
  std::size_t rest = capacity % BS;
  if (rest != 0) {
    REQUIRE_PRINT(map.Capacity() == capacity + BS - rest); 
  }
  else {
    REQUIRE_PRINT(map.Capacity() == capacity); 
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  REQUIRE_PRINT(observed.size() == 0);
}

TEMPLATE_TEST_CASE_SIG(
  "static_map Key != Value create test",
  "[static_map][diffKVCreate]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint64_t, 1),
  (uint32_t, int64_t, 5),
  (uint64_t, uint32_t, 1),
  (uint64_t, int32_t, 5),
  (float, uint32_t, 1),
  (float, int32_t, 5),
  (float, uint64_t, 1),
  (float, int64_t, 5),
  (aclco::fp16_t, uint32_t, 1),
  (aclco::fp16_t, uint64_t, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = 160000000;
  CAPTURE(capacity, BS);
  PRINT_BEFORE_EXEC("create test", Key, Value, BS, capacity, 0);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);
  
  std::size_t rest = capacity % BS;
  if (rest != 0) {
    REQUIRE_PRINT(map.Capacity() == capacity + BS - rest); 
  }
  else {
    REQUIRE_PRINT(map.Capacity() == capacity); 
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  REQUIRE_PRINT(observed.size() == 0);
}

TEMPLATE_TEST_CASE_SIG(
  "static_map extent storage create test",
  "[static_map][extentStorageCreate]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (int32_t, int32_t, 5),
  (uint64_t, uint64_t, 1),
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

  std::size_t capacity = 159;
  CAPTURE(capacity, BS);
  PRINT_BEFORE_EXEC("create test", Key, Value, BS, capacity, 0);
  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);
  
  std::size_t rest = capacity % BS;
  if (rest != 0) {
    REQUIRE_PRINT(map.Capacity() == capacity + BS - rest); 
  }
  else {
    REQUIRE_PRINT(map.Capacity() == capacity); 
  }

  auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);

  REQUIRE_PRINT(observed.size() == 0);
}
