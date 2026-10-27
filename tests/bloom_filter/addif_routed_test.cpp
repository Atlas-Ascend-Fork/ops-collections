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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"

namespace {

using Key = std::uint32_t;
using Filter = aclco::BloomFilter<Key>;
using Extent = typename Filter::ExtentType;
using Word = typename Filter::WordType;

std::vector<Key> MakeKeys(std::size_t count)
{
    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i) {
        keys[i] = static_cast<Key>(i * 2654435761ULL + 17ULL);
    }
    return keys;
}

std::vector<Word> ReadWords(Filter const& filter)
{
    std::vector<Word> host(static_cast<std::size_t>(filter.NumWords()));
    aclco::test::CheckAcl(aclrtMemcpy(host.data(), host.size() * sizeof(Word), filter.Data(),
                                      host.size() * sizeof(Word), ACL_MEMCPY_DEVICE_TO_HOST),
                          "read BloomFilter words");
    return host;
}

} // namespace

TEST_CASE("BloomFilter AddIf routed path matches Add for large batches", "[bloom_filter][addif][routed]")
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 1U << 20;
    constexpr std::size_t keyCount = 2000000;
    std::vector<Key> const keys = MakeKeys(keyCount);
    std::vector<std::uint8_t> stencil(keyCount, 1);
    for (std::size_t i = 0; i < keyCount; ++i) {
        stencil[i] = static_cast<std::uint8_t>((i % 3) != 1);
    }
    std::vector<Key> selected;
    selected.reserve(keyCount);
    for (std::size_t i = 0; i < keyCount; ++i) {
        if (stencil[i] != 0) {
            selected.push_back(keys[i]);
        }
    }

    aclco::test::DeviceBuffer<Key> deviceKeys(keyCount);
    aclco::test::DeviceBuffer<std::uint8_t> deviceStencil(keyCount);
    aclco::test::DeviceBuffer<Key> deviceSelected(selected.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    deviceStencil.CopyFromHostAsync(stencil.data(), stencil.size(), stream);
    deviceSelected.CopyFromHostAsync(selected.data(), selected.size(), stream);
    aclco::test::Sync(stream);

    Filter ifFilter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    Filter addFilter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    ifFilter.AddIf(deviceKeys.Data(), deviceStencil.Data(), Extent{keyCount}, stream);
    addFilter.Add(deviceSelected.Data(), Extent{selected.size()}, stream);
    REQUIRE(ReadWords(ifFilter) == ReadWords(addFilter));
    REQUIRE(ReadWords(ifFilter) == aclco::test::MakeBloomGolden<Key>(numBlocks, selected));

    std::vector<std::uint8_t> zeroStencil(keyCount, 0);
    aclco::test::DeviceBuffer<std::uint8_t> deviceZeroStencil(keyCount);
    deviceZeroStencil.CopyFromHostAsync(zeroStencil.data(), zeroStencil.size(), stream);
    aclco::test::Sync(stream);
    Filter zeroFilter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    zeroFilter.AddIf(deviceKeys.Data(), deviceZeroStencil.Data(), Extent{keyCount}, stream);
    std::vector<Word> const zeroWords = ReadWords(zeroFilter);
    REQUIRE(std::all_of(zeroWords.begin(), zeroWords.end(), [](Word word) { return word == Word{0}; }));

    std::vector<std::uint8_t> oneStencil(keyCount, 1);
    aclco::test::DeviceBuffer<std::uint8_t> deviceOneStencil(keyCount);
    deviceOneStencil.CopyFromHostAsync(oneStencil.data(), oneStencil.size(), stream);
    aclco::test::Sync(stream);
    Filter oneIfFilter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    Filter oneAddFilter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    oneIfFilter.AddIf(deviceKeys.Data(), deviceOneStencil.Data(), Extent{keyCount}, stream);
    oneAddFilter.Add(deviceKeys.Data(), Extent{keyCount}, stream);
    REQUIRE(ReadWords(oneIfFilter) == ReadWords(oneAddFilter));
}
