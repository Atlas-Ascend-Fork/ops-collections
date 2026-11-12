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

#include "hash_functions.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/set_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/matchers.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

TEST_CASE("static_set rejects probing scheme with instance state", "[static_set][create]")
{
  aclco::test::AclGlobalGuard g_acl;

  using Key = uint32_t;
  constexpr Key emptyKey = 0;

  REQUIRE_THROWS_AS(
    (aclco::StaticSet<Key>(100, emptyKey, aclco::EqualTo<Key>{},
      aclco::DoubleHashing<aclco::xxhash_32<Key>>{aclco::xxhash_32<Key>{0xA5A5A5A5u}})),
    std::invalid_argument);

  auto deflt = aclco::StaticSet<Key>(100, emptyKey);
  REQUIRE(deflt.Capacity() >= 100);
}

TEMPLATE_TEST_CASE_SIG(
  "static_set create test",
  "[static_set][create]",
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

  std::size_t capacity = 160000000;
  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  CAPTURE(capacity, BS);  
  PRINT_BEFORE_EXEC_SET("create test", Key, BS, set.Capacity(), 0);  
  REQUIRE_PRINT(set.Capacity() >= capacity);

  auto observed = aclco::test::DumpTable<Key>(set, sent, stream);

  REQUIRE_PRINT(observed.size() == 0);
}

