/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <numeric>
#include <limits>

#include "probing_scheme.h"
#include "hash_functions.h"
#include "utility/math_utils.h"

template <typename Key, typename Hash, int BucketSize>
static void CheckLinearProbingIterator(Key probeKey, Key tableSzie)
{
  aclco::LinearProbing<Hash> probing{};
  auto hasher = probing.HashFunction();

  // MakeIterator computes:
  // init = sanitize(hash(probeKey)) % (upperBound / BucketSize) * BucketSize
  using SizeType = Key;
  auto const hashVal = static_cast<SizeType>(hasher(probeKey));
  auto const groups   = static_cast<SizeType>(tableSzie / BucketSize);
  auto const initExpected =
    static_cast<SizeType>((hashVal % groups) * static_cast<SizeType>(BucketSize));

  auto it = probing.template MakeIterator<BucketSize>(probeKey, tableSzie);

  REQUIRE(*it == initExpected);

  // Prefix ++ returns the incremented iterator
  auto itPrefix = it;
  ++itPrefix;
  REQUIRE(*itPrefix == (initExpected + BucketSize) % tableSzie);

  // Postfix ++ returns the old value and then increments
  auto itPost = it;
  auto old     = *itPost;
  auto retOld = *(itPost++);
  REQUIRE(retOld == old);
  REQUIRE(*itPost == (old + BucketSize) % tableSzie);
}

template <typename Key, int BucketSize>
static Key GetValidTableSize(Key initSize)
{
  auto extent = aclco::MakeValidExtent<BucketSize, Key, aclco::dynamicExtent>(
    aclco::Extent<Key, aclco::dynamicExtent>(initSize));
  return static_cast<Key>(extent);
}

template <typename Key, typename Hash1, typename Hash2, int BucketSize>
static void CheckDoubleHashingIterator(Key probeKey, Key initSize)
{
  Key tableSzie = GetValidTableSize<Key, BucketSize>(initSize);

  aclco::DoubleHashing<Hash1, Hash2> probing{};
  auto hasher = probing.HashFunction();
  // MakeIterator computes:
  // init = sanitize(hash(probeKey)) % (upperBound / BucketSize) * BucketSize
  using SizeType = Key;
  auto const hash1Val = static_cast<SizeType>(std::get<0>(hasher)(probeKey));
  auto const hash2Val = static_cast<SizeType>(std::get<1>(hasher)(probeKey));
  auto const groups   = static_cast<SizeType>(tableSzie / BucketSize);
  auto const initExpected = static_cast<SizeType>((hash1Val % groups) * static_cast<SizeType>(BucketSize));
  auto const stepExpected = static_cast<SizeType>((hash2Val % (groups - 1) + 1) * static_cast<SizeType>(BucketSize));
  auto it = probing.template MakeIterator<BucketSize>(probeKey, tableSzie);

  REQUIRE(*it == initExpected);

  // Prefix ++ returns the incremented iterator
  auto itPrefix = it;
  ++itPrefix;
  REQUIRE(*itPrefix == (initExpected + stepExpected) % tableSzie);

  // Postfix ++ returns the old value and then increments
  auto itPost = it;
  auto old     = *itPost;
  auto retOld = *(itPost++);
  REQUIRE(retOld == old);
  REQUIRE(*itPost == (old + stepExpected) % tableSzie);
}

template <typename Key, typename Hash1, typename Hash2, int BucketSize>
static void CheckDoubleHashingCollisions(Key initSize)
{
  Key tableSzie = GetValidTableSize<Key, BucketSize>(initSize);
  aclco::DoubleHashing<Hash1, Hash2> probing{};
  auto hasher = probing.HashFunction();
  using SizeType = Key;
  auto const groups   = static_cast<SizeType>(tableSzie / BucketSize);

  SECTION("Test covering all possible positions") {
    Key testKey = 100;
    auto it = probing.template MakeIterator<BucketSize>(testKey, tableSzie);

    std::vector<Key> visited;
    Key start = *it;
    for (Key i = 0; i < groups; ++i) {
      visited.push_back(*it);
      ++it;
    }
    std::vector<bool> positionVisited(tableSzie, false);
    for (Key pos : visited) {
      REQUIRE_FALSE(positionVisited[pos]);
      positionVisited[pos] = true;
    }
    REQUIRE(*it == start);
  }

  SECTION("Test step size coprime with table size") {
    Key testKey = 1000;
    auto hash2Val = static_cast<SizeType>(std::get<1>(hasher)(testKey));
    auto step = static_cast<SizeType>((hash2Val % (groups - 1) + 1));

    REQUIRE(step >= 1);
    REQUIRE(step <= groups - 1);
    REQUIRE(std::gcd(step, groups) == 1);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "LinearProbing iterator semantics match expected arithmetic progression",
  "[probing]",
  ((typename Key, typename Hash, int BucketSize), Key, Hash, BucketSize),
  (std::uint32_t, aclco::xxhash_32<std::uint32_t>, 1),
  (std::uint32_t, aclco::murmurhash3_32<std::uint32_t>, 1),
  (std::uint32_t, aclco::murmurhash3_fmix32<std::uint32_t>, 5),
  (std::uint64_t, aclco::murmurhash3_fmix64<std::uint64_t>, 5))
{
  CheckLinearProbingIterator<Key, Hash, BucketSize>(static_cast<Key>(6), static_cast<Key>(40));
}

TEMPLATE_TEST_CASE_SIG(
  "DoubleHashing iterator semantics match expected arithmetic progression",
  "[probing]",
  ((typename Key, typename Hash1, typename Hash2, int BucketSize), Key, Hash1, Hash2, BucketSize),
  (std::uint32_t, aclco::xxhash_32<std::uint32_t>, aclco::xxhash_32<std::uint32_t>, 5))
{
  CheckDoubleHashingIterator<Key, Hash1, Hash2, BucketSize>(static_cast<Key>(6), static_cast<Key>(40));
}

TEMPLATE_TEST_CASE_SIG(
  "DoubleHashing collision test",
  "[probing][doublehashing][collision]",
  ((typename Key, typename Hash1, typename Hash2, int BucketSize), Key, Hash1, Hash2, BucketSize),
  (std::uint32_t, aclco::xxhash_32<std::uint32_t>, aclco::xxhash_32<std::uint32_t>, 1),
  (std::uint32_t, aclco::murmurhash3_32<std::uint32_t>, aclco::murmurhash3_32<std::uint32_t>, 1),
  (std::uint32_t, aclco::murmurhash3_fmix32<std::uint32_t>, aclco::murmurhash3_fmix32<std::uint32_t>, 5),
  (std::uint64_t, aclco::murmurhash3_fmix64<std::uint64_t>, aclco::murmurhash3_fmix64<std::uint64_t>, 5))
{
  CheckDoubleHashingCollisions<Key, Hash1, Hash2, BucketSize>(static_cast<Key>(100));
}

namespace {
struct StatefulHash {
    std::uint32_t offset;

    COLLECTION_HOST_DEVICE constexpr std::uint32_t operator()(std::uint32_t key) const { return key + offset; }
};
} // namespace

TEST_CASE("DoubleHashing can be reconstructed from HashFunction", "[probing][doublehashing]")
{
    aclco::DoubleHashing<StatefulHash, StatefulHash> original(StatefulHash{3}, StatefulHash{11});
    auto reconstructed = aclco::DoubleHashing<StatefulHash, StatefulHash>(original.HashFunction());
    auto hashes = reconstructed.HashFunction();

    REQUIRE(std::get<0>(hashes)(5) == 8);
    REQUIRE(std::get<1>(hashes)(5) == 16);
}

TEMPLATE_TEST_CASE("ProbingIterator wraps without overflowing", "[probing][boundary]",
                   std::uint32_t, std::uint64_t)
{
    using T = TestType;
    for (T bound = 1; bound <= 24; ++bound) {
        for (T step = 1; step <= bound; ++step) {
            for (T start = 0; start < bound; ++start) {
                aclco::detail::ProbingIterator<T> it(start, step, bound);
                T expected = start;
                for (T i = 0; i < 2 * bound; ++i) {
                    REQUIRE(*(it++) == expected);
                    expected = (expected + step) % bound;
                    REQUIRE(*it == expected);
                }
            }
        }
    }
    auto const max = std::numeric_limits<T>::max();
    for (T bound : {T(max / 2 + 1), T(max - 1), max}) {
        for (T step : {T(1), T(bound / 2), T(bound - 1), bound}) {
            for (T start : {T(0), T(bound / 2), T(bound - 1)}) {
                aclco::detail::ProbingIterator<T> it(start, step, bound);
                auto expected = static_cast<unsigned __int128>(start);
                for (int i = 0; i < 32; ++i) {
                    expected = (expected + step) % bound;
                    REQUIRE(*++it == static_cast<T>(expected));
                }
            }
        }
    }
}
