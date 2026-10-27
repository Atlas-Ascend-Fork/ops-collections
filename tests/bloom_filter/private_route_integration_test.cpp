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
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"

namespace {

template <typename Key>
using H8V1Policy = aclco::BloomFilterPolicy<Key, aclco::xxhash_64<Key>, std::uint32_t, 8, 8, 8, 1, 8, 1>;

template <typename Key>
void AppendBucketTileKeys(std::vector<Key>& keys, std::uint64_t& candidate, std::uint32_t bucket, std::uint32_t tile,
                          std::size_t count, std::vector<std::uint64_t>* hashes = nullptr)
{
    H8V1Policy<Key> const policy{};
    std::size_t accepted = 0;
    while (accepted < count) {
        Key const key = static_cast<Key>(candidate++);
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        if ((upperHash >> 17) == bucket && ((upperHash >> 16) & 1U) == tile) {
            keys.push_back(key);
            if (hashes != nullptr) {
                hashes->push_back(hash);
            }
            ++accepted;
        }
    }
}

template <typename Key>
struct ScaledRouteInputs {
    std::vector<Key> firstKeys;
    std::vector<Key> secondKeys;
    std::vector<std::uint64_t> secondHashes;
};

template <typename Key>
ScaledRouteInputs<Key> MakeScaledRouteInputs()
{
    constexpr std::size_t recordsPerSegment = aclco::detail::bloom_filter::finePrivateRouteRecordsPerSegment;
    ScaledRouteInputs<Key> inputs;
    std::uint64_t candidate = 1;
    inputs.firstKeys.reserve(529);
    for (std::uint32_t bucket = 0; bucket < 4; ++bucket) {
        AppendBucketTileKeys(inputs.firstKeys, candidate, bucket, 0, 48);
        AppendBucketTileKeys(inputs.firstKeys, candidate, bucket, 1, 48);
    }
    std::uint32_t const lastBucket = aclco::detail::bloom_filter::finePrivateRouteBucketCount - 1U;
    AppendBucketTileKeys(inputs.firstKeys, candidate, lastBucket, 0, 72);
    AppendBucketTileKeys(inputs.firstKeys, candidate, lastBucket, 1, 73);
    REQUIRE(inputs.firstKeys.size() == 529);

    inputs.secondKeys.reserve(recordsPerSegment);
    inputs.secondHashes.reserve(recordsPerSegment);
    AppendBucketTileKeys(inputs.secondKeys, candidate, 4, 0, 40, &inputs.secondHashes);
    AppendBucketTileKeys(inputs.secondKeys, candidate, 4, 1, 40, &inputs.secondHashes);
    REQUIRE(inputs.secondKeys.size() == recordsPerSegment);
    return inputs;
}

template <typename T>
std::vector<T> CopyDeviceValues(T const* device, std::size_t count, aclrtStream stream)
{
    std::vector<T> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(T), device, count * sizeof(T),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter private route D2H");
    aclco::test::Sync(stream);
    return host;
}

template <typename Key>
void LaunchScaledPrivateRoute(aclco::test::DeviceBuffer<std::uint32_t>& filter, aclco::test::DeviceBuffer<Key>& keys,
                              aclco::test::DeviceBuffer<std::uint32_t>& workspace, std::size_t keyCount,
                              bool preserveExisting, aclrtStream stream)
{
    using Policy = H8V1Policy<Key>;
    constexpr std::uint32_t bucketBits = aclco::detail::bloom_filter::finePrivateRouteBucketBits;
    constexpr std::uint32_t recordsPerSegment = aclco::detail::bloom_filter::finePrivateRouteRecordsPerSegment;
    constexpr std::uint64_t numBlocks = 1ULL << 16;
    constexpr std::uint32_t producerCount = 1;
    using RouteStorage = aclco::detail::bloom_filter::PrivateRouteStorage<bucketBits, true>;
    using ApplyStorage = aclco::detail::bloom_filter::PrivateApplyStorage<
        aclco::detail::bloom_filter::privateSparseSlotsPerBlock, 1>;

    aclco::BloomFilterAddPrivateRoute<Key, Policy, bucketBits, recordsPerSegment, true>
        <<<producerCount, RouteStorage::launchBytes, stream>>>(
            reinterpret_cast<std::uint8_t*>(filter.Data()), reinterpret_cast<std::uint8_t*>(keys.Data()),
            reinterpret_cast<std::uint8_t*>(workspace.Data()), numBlocks, keyCount, 0, producerCount);
    if (preserveExisting) {
        aclco::BloomFilterAddPrivateApply<Policy, bucketBits, recordsPerSegment, true,
                                          aclco::detail::bloom_filter::privateSparseSlotsPerBlock, 16, 1, 2, true>
            <<<1, ApplyStorage::launchBytes, stream>>>(reinterpret_cast<std::uint8_t*>(filter.Data()),
                                                       reinterpret_cast<std::uint8_t*>(workspace.Data()), 0,
                                                       producerCount);
    } else {
        aclco::BloomFilterAddPrivateApply<Policy, bucketBits, recordsPerSegment, true,
                                          aclco::detail::bloom_filter::privateSparseSlotsPerBlock, 16, 1, 2, false>
            <<<1, ApplyStorage::launchBytes, stream>>>(reinterpret_cast<std::uint8_t*>(filter.Data()),
                                                       reinterpret_cast<std::uint8_t*>(workspace.Data()), 0,
                                                       producerCount);
    }
}

std::vector<std::uint32_t> MakeCompactBlockKeys(std::uint64_t& candidate, std::uint32_t bucket,
                                                std::uint32_t localBlock, std::size_t count)
{
    using Policy = H8V1Policy<std::uint32_t>;
    Policy const policy{};
    std::vector<std::uint32_t> keys;
    keys.reserve(count);
    while (keys.size() < count) {
        std::uint32_t const key = static_cast<std::uint32_t>(candidate++);
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        if ((upperHash >> 29) == bucket && ((upperHash >> 27) & 3U) == localBlock) {
            keys.push_back(key);
        }
    }
    return keys;
}

void WriteCompactSegment(std::vector<std::uint32_t>& workspace, std::uint32_t bucket,
                         std::vector<std::uint32_t> const& keys, bool overflow)
{
    using Policy = H8V1Policy<std::uint32_t>;
    constexpr std::uint32_t counterWords = 2;
    constexpr std::uint32_t recordsPerSegment = 8;
    std::uint32_t const shift = (bucket & 3U) * 8U;
    workspace[bucket >> 2] |= (static_cast<std::uint32_t>(keys.size()) | (overflow ? 0x80U : 0U)) << shift;
    Policy const policy{};
    for (std::size_t ticket = 0; ticket < keys.size(); ++ticket) {
        std::uint64_t const hash = policy.HashKey(keys[ticket]);
        std::size_t const offset = counterWords + 2ULL * (bucket * recordsPerSegment + ticket);
        workspace[offset] = static_cast<std::uint32_t>(hash);
        workspace[offset + 1] = static_cast<std::uint32_t>(hash >> 32);
    }
}

template <bool PreserveExisting>
void LaunchScaledCompactApply(aclco::test::DeviceBuffer<std::uint32_t>& filter,
                              aclco::test::DeviceBuffer<std::uint32_t>& workspace, aclrtStream stream)
{
    using Policy = H8V1Policy<std::uint32_t>;
    using Storage = aclco::detail::bloom_filter::PrivateCompactSparseApplyStorage<4, 2>;
    aclco::BloomFilterAddPrivateCompactSparseApply<Policy, PreserveExisting, 3, 8, 27, 4, 2>
        <<<2, Storage::launchBytes, stream>>>(reinterpret_cast<std::uint8_t*>(filter.Data()),
                                              reinterpret_cast<std::uint8_t*>(workspace.Data()), 0, 1);
}

template <bool PreserveExisting>
void LaunchScaledCompactRouteAdd(aclco::test::DeviceBuffer<std::uint32_t>& filter,
                                 aclco::test::DeviceBuffer<std::uint32_t>& keys,
                                 aclco::test::DeviceBuffer<std::uint32_t>& workspace, std::size_t keyCount,
                                 aclrtStream stream)
{
    using Policy = H8V1Policy<std::uint32_t>;
    constexpr std::uint32_t producerCount = 2;
    using RouteStorage = aclco::detail::bloom_filter::PrivateRouteStorage<3, true>;
    using ApplyStorage = aclco::detail::bloom_filter::PrivateCompactSparseApplyStorage<4, 2>;
    aclco::BloomFilterAddPrivateRoute<std::uint32_t, Policy, 3, 8, true>
        <<<producerCount, RouteStorage::launchBytes, stream>>>(
            reinterpret_cast<std::uint8_t*>(filter.Data()), reinterpret_cast<std::uint8_t*>(keys.Data()),
            reinterpret_cast<std::uint8_t*>(workspace.Data()), 32, keyCount, 0, producerCount);
    aclco::BloomFilterAddPrivateCompactSparseApply<Policy, PreserveExisting, 3, 8, 27, 4, 2>
        <<<2, ApplyStorage::launchBytes, stream>>>(reinterpret_cast<std::uint8_t*>(filter.Data()),
                                                   reinterpret_cast<std::uint8_t*>(workspace.Data()), 0, producerCount);
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter scaled private route preserves every record",
                   "[bloom_filter][correctness][routed_add][private_route][integration]", std::uint32_t, std::uint64_t)
{
    using Key = TestType;
    using Policy = H8V1Policy<Key>;
    constexpr std::size_t numBlocks = 1U << 16;
    constexpr std::size_t wordsPerBlock = Policy::wordsPerBlock;
    constexpr std::size_t counterWords = aclco::detail::bloom_filter::privateRoutePackedCounterWords;
    constexpr std::size_t recordsPerSegment = aclco::detail::bloom_filter::finePrivateRouteRecordsPerSegment;
    constexpr std::size_t workspaceWords = counterWords + 2ULL *
                                                              aclco::detail::bloom_filter::finePrivateRouteBucketCount *
                                                              recordsPerSegment;

    auto inputs = MakeScaledRouteInputs<Key>();
    auto& firstKeys = inputs.firstKeys;
    auto& secondKeys = inputs.secondKeys;
    auto& secondHashes = inputs.secondHashes;

    std::vector<Key> allKeys = firstKeys;
    allKeys.insert(allKeys.end(), secondKeys.begin(), secondKeys.end());
    auto const expected = aclco::test::MakeBloomGolden<Key>(numBlocks, allKeys);

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream const stream = streamGuard.stream;
    aclco::test::DeviceBuffer<std::uint32_t> filter(numBlocks * wordsPerBlock);
    aclco::test::DeviceBuffer<std::uint32_t> workspace(workspaceWords);
    aclco::test::DeviceBuffer<Key> firstDevice(firstKeys.size());
    aclco::test::DeviceBuffer<Key> secondDevice(secondKeys.size());
    filter.MemsetZero(stream);
    firstDevice.CopyFromHostAsync(firstKeys.data(), firstKeys.size(), stream);
    secondDevice.CopyFromHostAsync(secondKeys.data(), secondKeys.size(), stream);

    // Keep both route/apply pairs queued on one stream. The second Route must
    // overwrite every packed count word and the second Apply must preserve the
    // first pair's filter contents.
    LaunchScaledPrivateRoute(filter, firstDevice, workspace, firstKeys.size(), false, stream);
    LaunchScaledPrivateRoute(filter, secondDevice, workspace, secondKeys.size(), true, stream);

    REQUIRE(CopyDeviceValues(filter.Data(), filter.Size(), stream) == expected);
    auto const counters = CopyDeviceValues(workspace.Data(), counterWords, stream);
    for (std::size_t word = 0; word < counters.size(); ++word) {
        CAPTURE(word);
        REQUIRE(counters[word] == (word == 1 ? 0x50U : 0U));
    }

    auto const* records = reinterpret_cast<std::uint64_t const*>(workspace.Data() + counterWords);
    auto observedHashes = CopyDeviceValues(records + 4 * recordsPerSegment, recordsPerSegment, stream);
    std::sort(observedHashes.begin(), observedHashes.end());
    std::sort(secondHashes.begin(), secondHashes.end());
    REQUIRE(observedHashes == secondHashes);
}

TEST_CASE("BloomFilter compact sparse Apply preserves local and route overflow",
          "[bloom_filter][correctness][routed_add][private_route][compact_apply]")
{
    constexpr std::size_t numBlocks = 32;
    constexpr std::size_t wordsPerBlock = 8;
    constexpr std::size_t workspaceWords = 2 + 2 * 8 * 8;

    std::uint64_t candidate = 1;
    auto const localOverflowKeys = MakeCompactBlockKeys(candidate, 0, 0, 8);
    auto const routedKeys = MakeCompactBlockKeys(candidate, 1, 0, 3);
    auto const activeRouteOverflow = MakeCompactBlockKeys(candidate, 1, 0, 1);
    auto const routeOverflowOnly = MakeCompactBlockKeys(candidate, 1, 1, 1);
    auto const secondKeys = MakeCompactBlockKeys(candidate, 2, 2, 5);

    std::vector<std::uint32_t> firstWorkspace(workspaceWords, 0);
    WriteCompactSegment(firstWorkspace, 0, localOverflowKeys, false);
    WriteCompactSegment(firstWorkspace, 1, routedKeys, true);

    std::vector<std::uint32_t> initialKeys = activeRouteOverflow;
    initialKeys.insert(initialKeys.end(), routeOverflowOnly.begin(), routeOverflowOnly.end());
    auto initialFilter = aclco::test::MakeBloomGolden<std::uint32_t>(numBlocks, initialKeys);

    std::vector<std::uint32_t> expectedKeys = initialKeys;
    expectedKeys.insert(expectedKeys.end(), localOverflowKeys.begin(), localOverflowKeys.end());
    expectedKeys.insert(expectedKeys.end(), routedKeys.begin(), routedKeys.end());
    auto expected = aclco::test::MakeBloomGolden<std::uint32_t>(numBlocks, expectedKeys);

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream const stream = streamGuard.stream;
    aclco::test::DeviceBuffer<std::uint32_t> filter(numBlocks * wordsPerBlock);
    aclco::test::DeviceBuffer<std::uint32_t> workspace(workspaceWords);
    filter.CopyFromHostAsync(initialFilter.data(), initialFilter.size(), stream);
    workspace.CopyFromHostAsync(firstWorkspace.data(), firstWorkspace.size(), stream);
    LaunchScaledCompactApply<false>(filter, workspace, stream);
    REQUIRE(filter.CopyToHost(stream) == expected);

    std::vector<std::uint32_t> secondWorkspace(workspaceWords, 0);
    WriteCompactSegment(secondWorkspace, 2, secondKeys, false);
    workspace.CopyFromHostAsync(secondWorkspace.data(), secondWorkspace.size(), stream);
    LaunchScaledCompactApply<true>(filter, workspace, stream);
    expectedKeys.insert(expectedKeys.end(), secondKeys.begin(), secondKeys.end());
    expected = aclco::test::MakeBloomGolden<std::uint32_t>(numBlocks, expectedKeys);
    REQUIRE(filter.CopyToHost(stream) == expected);
}

TEST_CASE("BloomFilter compact sparse Route and Apply are reusable without host synchronization",
          "[bloom_filter][correctness][routed_add][private_route][compact_apply][async]")
{
    constexpr std::size_t numBlocks = 32;
    constexpr std::size_t wordsPerBlock = 8;
    constexpr std::size_t producerCount = 2;
    constexpr std::size_t counterWords = 2 * producerCount;
    constexpr std::size_t workspaceWords = counterWords + 2 * 8 * producerCount * 8;

    std::uint64_t candidate = 1;
    auto firstKeys = MakeCompactBlockKeys(candidate, 0, 0, 12);
    while (firstKeys.size() < 513) {
        firstKeys.push_back(static_cast<std::uint32_t>(candidate++));
    }
    auto secondKeys = MakeCompactBlockKeys(candidate, 0, 0, 3);
    auto const lastBlockKeys = MakeCompactBlockKeys(candidate, 7, 3, 8);
    secondKeys.insert(secondKeys.end(), lastBlockKeys.begin(), lastBlockKeys.end());
    auto expectedKeys = firstKeys;
    expectedKeys.insert(expectedKeys.end(), secondKeys.begin(), secondKeys.end());
    auto const expected = aclco::test::MakeBloomGolden<std::uint32_t>(numBlocks, expectedKeys);

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream const stream = streamGuard.stream;
    aclco::test::DeviceBuffer<std::uint32_t> filter(numBlocks * wordsPerBlock);
    aclco::test::DeviceBuffer<std::uint32_t> workspace(workspaceWords);
    aclco::test::DeviceBuffer<std::uint32_t> firstDevice(firstKeys.size());
    aclco::test::DeviceBuffer<std::uint32_t> secondDevice(secondKeys.size());
    filter.MemsetZero(stream);
    firstDevice.CopyFromHostAsync(firstKeys.data(), firstKeys.size(), stream);
    secondDevice.CopyFromHostAsync(secondKeys.data(), secondKeys.size(), stream);

    // Both pairs remain queued on one stream. The first Route exceeds its
    // per-producer segment capacity, and the second pair immediately reuses
    // every counter word while preserving the first filter contents.
    LaunchScaledCompactRouteAdd<false>(filter, firstDevice, workspace, firstKeys.size(), stream);
    LaunchScaledCompactRouteAdd<true>(filter, secondDevice, workspace, secondKeys.size(), stream);
    REQUIRE(filter.CopyToHost(stream) == expected);
}
