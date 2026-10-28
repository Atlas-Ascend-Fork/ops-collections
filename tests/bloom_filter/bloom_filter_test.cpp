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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"
#include "tests/common/key_values.h"
#include "tests/common/object_representation.h"

namespace {

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count, std::uint64_t offset = 0)
{
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t const value = ((i + offset) * 2654435761ull + 17ull) % 10000019ull;
        keys.push_back(static_cast<Key>(value));
    }
    if constexpr (std::is_same_v<Key, float>) {
        if (count > 5) {
            keys[0] = 0.0f;
            keys[1] = -0.0f;
            keys[2] = std::numeric_limits<float>::infinity();
            keys[3] = -std::numeric_limits<float>::infinity();
            std::uint32_t const nanBits1 = 0x7fc00001u;
            std::uint32_t const nanBits2 = 0x7fc01234u;
            keys[4] = aclco::test::ObjectRepresentationCast<float>(nanBits1);
            keys[5] = aclco::test::ObjectRepresentationCast<float>(nanBits2);
        }
    } else if constexpr (std::is_integral_v<Key> && std::is_signed_v<Key>) {
        aclco::test::SetSignedBoundaryValues(keys);
    }
    return keys;
}

template <typename Word>
std::vector<Word> CopyFilterWords(Word const* device, std::size_t count, aclrtStream stream)
{
    std::vector<Word> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(Word), device, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter D2H");
    aclco::test::Sync(stream);
    return host;
}

bool AllBytesZero(std::vector<std::uint8_t> const& values)
{
    return std::all_of(values.begin(), values.end(), [](std::uint8_t value) { return value == 0; });
}

template <typename Word>
bool AllWordsZero(Word const* device, std::size_t count, aclrtStream stream)
{
    auto const words = CopyFilterWords(device, count, stream);
    return std::all_of(words.begin(), words.end(), [](Word value) { return value == Word{0}; });
}

struct EmptyFilterTestData {
    static constexpr std::size_t numBlocks = 257;
    static constexpr std::size_t keyCount = 128;

    aclrtStream stream;
    std::vector<std::uint32_t> keys;
    std::vector<std::uint8_t> stencil;
    aclco::test::DeviceBuffer<std::uint32_t> deviceKeys;
    aclco::test::DeviceBuffer<std::uint8_t> deviceStencil;
    aclco::test::DeviceBuffer<std::uint8_t> deviceOutput;

    explicit EmptyFilterTestData(aclrtStream s)
        : stream(s), keys(MakeKeys<std::uint32_t>(keyCount)), stencil(keyCount, 1), deviceKeys(keyCount),
          deviceStencil(keyCount), deviceOutput(keyCount)
    {
        deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
        deviceStencil.CopyFromHostAsync(stencil.data(), stencil.size(), stream);
        deviceOutput.MemsetZero(stream);
        aclco::test::Sync(stream);
    }

    aclco::Extent<std::size_t> KeyExtent() const { return aclco::Extent<std::size_t>{keyCount}; }
};

template <typename Filter>
void RequireInitialState(Filter const& filter, std::size_t numBlocks, aclrtStream stream)
{
    using Word = typename Filter::WordType;
    REQUIRE(static_cast<std::size_t>(filter.BlockExtent()) == numBlocks);
    REQUIRE(filter.NumWords() == numBlocks * Filter::wordsPerBlock);
    REQUIRE(filter.SizeBytes() == filter.NumWords() * sizeof(Word));
    auto const observedWords = CopyFilterWords(filter.Data(), static_cast<std::size_t>(filter.NumWords()), stream);
    REQUIRE(std::all_of(observedWords.begin(), observedWords.end(), [](Word value) { return value == Word{0}; }));
}

template <typename Key, typename Filter>
std::vector<std::uint8_t> RequireContainsMatchesGolden(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceQueries,
                                                       aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput,
                                                       std::vector<Key> const& queries,
                                                       std::vector<typename Filter::WordType> const& expectedWords,
                                                       std::size_t numBlocks, aclrtStream stream)
{
    filter.Contains(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, deviceOutput.Data(), stream);
    auto output = deviceOutput.CopyToHost(stream);
    REQUIRE(output.size() == queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        bool const expected = aclco::test::BloomGoldenContains<Key>(expectedWords, numBlocks, queries[i]);
        CAPTURE(i);
        REQUIRE((output[i] != 0) == expected);
    }
    return output;
}

template <typename Key, typename Filter>
void RequireCompatibilityContainsOverloads(Filter const& filter, aclco::test::DeviceBuffer<Key>& deviceQueries,
                                           aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput,
                                           std::size_t queryCount, std::vector<std::uint8_t> const& expectedOutput,
                                           aclrtStream stream)
{
    deviceOutput.MemsetZero(stream);
    filter.Contains(deviceQueries.Data(), deviceOutput.Data(), aclco::Extent<std::size_t>{queryCount}, stream);
    auto compatibilityOutput = deviceOutput.CopyToHost(stream);
    REQUIRE(compatibilityOutput == expectedOutput);

    deviceOutput.MemsetZero(stream);
    filter.ContainsAsync(deviceQueries.Data(), deviceOutput.Data(), aclco::Extent<std::size_t>{queryCount}, stream);
    compatibilityOutput = deviceOutput.CopyToHost(stream);
    REQUIRE(compatibilityOutput == expectedOutput);
}

template <typename Key, typename Filter>
void RequireEmptyAndInvalidArguments(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceQueries,
                                     aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput, aclrtStream stream)
{
    filter.Add(nullptr, aclco::Extent<std::size_t>{0}, stream);
    filter.Contains(nullptr, aclco::Extent<std::size_t>{0}, nullptr, stream);
    filter.AddAsync(nullptr, aclco::Extent<std::size_t>{0}, stream);
    filter.ContainsAsync(nullptr, aclco::Extent<std::size_t>{0}, nullptr, stream);
    std::as_const(filter).Contains(nullptr, nullptr, aclco::Extent<std::size_t>{0}, stream);
    std::as_const(filter).ContainsAsync(nullptr, nullptr, aclco::Extent<std::size_t>{0}, stream);
    REQUIRE_THROWS_AS(filter.Add(nullptr, aclco::Extent<std::size_t>{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(filter.AddAsync(nullptr, aclco::Extent<std::size_t>{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(filter.Contains(nullptr, aclco::Extent<std::size_t>{1}, deviceOutput.Data(), stream),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(filter.Contains(deviceQueries.Data(), aclco::Extent<std::size_t>{1}, nullptr, stream),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(filter.ContainsAsync(nullptr, aclco::Extent<std::size_t>{1}, deviceOutput.Data(), stream),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(filter.ContainsAsync(deviceQueries.Data(), aclco::Extent<std::size_t>{1}, nullptr, stream),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).Contains(nullptr, deviceOutput.Data(), aclco::Extent<std::size_t>{1}, stream),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).Contains(deviceQueries.Data(), nullptr, aclco::Extent<std::size_t>{1}, stream),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).ContainsAsync(nullptr, deviceOutput.Data(), aclco::Extent<std::size_t>{1}, stream),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).ContainsAsync(deviceQueries.Data(), nullptr, aclco::Extent<std::size_t>{1}, stream),
        std::invalid_argument);
}

template <typename Key, typename Filter>
void RequireClearAndAsyncOrdering(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceKeys,
                                  aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput, std::size_t keyCount,
                                  aclrtStream stream)
{
    using Word = typename Filter::WordType;
    filter.Clear(stream);
    auto const observedWords = CopyFilterWords(std::as_const(filter).Data(),
                                               static_cast<std::size_t>(filter.NumWords()), stream);
    REQUIRE(std::all_of(observedWords.begin(), observedWords.end(), [](Word value) { return value == Word{0}; }));

    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, stream);
    filter.ClearAsync(stream);
    filter.ContainsAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, deviceOutput.Data(), stream);
    aclco::test::Sync(stream);
    auto const output = deviceOutput.CopyToHost(stream);
    REQUIRE(std::all_of(output.begin(), output.begin() + keyCount, [](std::uint8_t value) { return value == 0; }));
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter Add, Contains and Clear are bit-exact", "[bloom_filter][correctness]", std::int32_t,
                   std::uint32_t, std::int64_t, std::uint64_t, float)
{
    using Key = TestType;
    using Filter = aclco::BloomFilter<Key>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 257;
    auto keys = MakeKeys<Key>(4099);
    keys.insert(keys.end(), keys.begin(), keys.begin() + 64);

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, typename Filter::PolicyType{},
                  typename Filter::AllocatorType{}, stream);
    RequireInitialState(filter, numBlocks, stream);

    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);

    auto expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
    auto const observedWords = CopyFilterWords(std::as_const(filter).Data(),
                                               static_cast<std::size_t>(filter.NumWords()), stream);
    REQUIRE(observedWords == expectedWords);

    auto queries = keys;
    auto missing = MakeKeys<Key>(513, 100000);
    queries.insert(queries.end(), missing.begin(), missing.end());
    aclco::test::DeviceBuffer<Key> deviceQueries(queries.size());
    aclco::test::DeviceBuffer<std::uint8_t> deviceOutput(queries.size());
    deviceQueries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    deviceOutput.MemsetZero(stream);

    auto const output = RequireContainsMatchesGolden(filter, deviceQueries, deviceOutput, queries, expectedWords,
                                                     numBlocks, stream);
    RequireCompatibilityContainsOverloads(std::as_const(filter), deviceQueries, deviceOutput, queries.size(), output,
                                          stream);
    RequireEmptyAndInvalidArguments(filter, deviceQueries, deviceOutput, stream);
    RequireClearAndAsyncOrdering(filter, deviceKeys, deviceOutput, keys.size(), stream);
}

TEST_CASE("BloomFilter rejects a zero block extent", "[bloom_filter][validation]")
{
    REQUIRE_THROWS_AS((aclco::BloomFilter<std::uint32_t>{aclco::Extent<std::size_t>{0}}), std::invalid_argument);

    if constexpr (std::numeric_limits<std::size_t>::max() > std::numeric_limits<std::uint32_t>::max()) {
        std::size_t const tooManyBlocks = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
        REQUIRE_THROWS_AS((aclco::BloomFilter<std::uint32_t>{aclco::Extent<std::size_t>{tooManyBlocks}}),
                          std::length_error);
    }

    using SmallExtent = aclco::Extent<std::uint32_t>;
    using SmallFilter = aclco::BloomFilter<std::uint32_t, SmallExtent>;
    std::uint32_t const wordCountOverflow = std::numeric_limits<std::uint32_t>::max() / SmallFilter::wordsPerBlock + 1;
    REQUIRE_THROWS_AS((SmallFilter{SmallExtent{wordCountOverflow}}), std::length_error);
}

TEST_CASE("BloomFilter empty-filter sync fast paths preserve semantics", "[bloom_filter][empty]")
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    EmptyFilterTestData data(stream);
    using Filter = aclco::BloomFilter<std::uint32_t>;
    auto const extent = aclco::Extent<std::size_t>{EmptyFilterTestData::numBlocks};

    Filter emptyA(extent, typename Filter::PolicyType{}, stream);
    Filter emptyB(extent, typename Filter::PolicyType{}, stream);
    emptyA.Contains(data.deviceKeys.Data(), data.KeyExtent(), data.deviceOutput.Data(), stream);
    REQUIRE(AllBytesZero(data.deviceOutput.CopyToHost(stream)));

    data.deviceOutput.MemsetZero(stream);
    emptyA.ContainsIf(data.deviceKeys.Data(), data.deviceStencil.Data(), data.deviceOutput.Data(), data.KeyExtent(),
                      stream);
    REQUIRE(AllBytesZero(data.deviceOutput.CopyToHost(stream)));

    emptyA.Clear(stream);
    REQUIRE(AllWordsZero(std::as_const(emptyA).Data(), static_cast<std::size_t>(emptyA.NumWords()), stream));
    emptyA.Merge(emptyB, stream);
    REQUIRE(AllWordsZero(std::as_const(emptyA).Data(), static_cast<std::size_t>(emptyA.NumWords()), stream));
    emptyA.Intersect(emptyB, stream);
    REQUIRE(AllWordsZero(std::as_const(emptyA).Data(), static_cast<std::size_t>(emptyA.NumWords()), stream));
}

TEST_CASE("BloomFilter empty-filter async and mixed-source semantics", "[bloom_filter][empty]")
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    EmptyFilterTestData data(stream);
    using Filter = aclco::BloomFilter<std::uint32_t>;
    auto const extent = aclco::Extent<std::size_t>{EmptyFilterTestData::numBlocks};

    Filter empty(extent, typename Filter::PolicyType{}, stream);
    Filter nonEmpty(extent, typename Filter::PolicyType{}, stream);
    empty.ClearAsync(stream);
    empty.ContainsAsync(data.deviceKeys.Data(), data.KeyExtent(), data.deviceOutput.Data(), stream);
    aclco::test::Sync(stream);
    REQUIRE(AllBytesZero(data.deviceOutput.CopyToHost(stream)));

    nonEmpty.Add(data.deviceKeys.Data(), data.KeyExtent(), stream);
    auto const beforeMerge = CopyFilterWords(std::as_const(nonEmpty).Data(),
                                             static_cast<std::size_t>(nonEmpty.NumWords()), stream);
    nonEmpty.Merge(empty, stream);
    REQUIRE(CopyFilterWords(std::as_const(nonEmpty).Data(),
                            static_cast<std::size_t>(nonEmpty.NumWords()), stream) == beforeMerge);

    nonEmpty.Intersect(empty, stream);
    REQUIRE(AllWordsZero(std::as_const(nonEmpty).Data(), static_cast<std::size_t>(nonEmpty.NumWords()), stream));
    data.deviceOutput.MemsetZero(stream);
    nonEmpty.Contains(data.deviceKeys.Data(), data.KeyExtent(), data.deviceOutput.Data(), stream);
    REQUIRE(AllBytesZero(data.deviceOutput.CopyToHost(stream)));
}
