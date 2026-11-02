/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#include "common/static_multimap_test_common.h"

using namespace aclco::test::static_multimap;
namespace {
struct Nonzero {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const { return value != 0; }
};
template <class T>
using Device = aclco::test::DeviceBuffer<T>;
using Extent = aclco::Extent<std::size_t>;
} // namespace

TEMPLATE_TEST_CASE("static_multimap deterministic contents and output order", "[static_multimap][contract]", int32_t,
                   int64_t)
{
    using T = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto map = MakeMultimap<T, T>(7, stream);
    std::vector<aclco::Pair<T, T>> pairs{{3, 9}, {-7, 4}, {3, -8}, {3, -8}, {-7, -3}, {9, 7}, {3, 2}};
    Device<aclco::Pair<T, T>> input(pairs.size());
    std::vector<T> queries{3, -7, 3, 42, EmptyKey<T>()};
    Device<T> keys(queries.size()), found(queries.size());
    Device<T> outputKeys(10), outputValues(10);
    keys.CopyFromHostAsync(queries.data(), queries.size(), stream);
    for (int round = 0; round < 3; ++round) {
        map.Clear(stream);
        std::rotate(pairs.begin(), pairs.begin() + 1, pairs.end());
        input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
        REQUIRE(map.Insert(input.Data(), Extent(pairs.size()), stream) == 0);
        map.Find(keys.Data(), found.Data(), Extent(queries.size()), stream);
        REQUIRE(found.CopyToHost(stream) == std::vector<T>{-8, -3, -8, EmptyValue<T>(), EmptyValue<T>()});
        REQUIRE(map.Count(keys.Data(), Extent(queries.size()), stream) == 10);
        REQUIRE(map.Retrieve(keys.Data(), Extent(queries.size()), outputKeys.Data(), outputValues.Data(), Extent(10),
                             stream) == 10);
        REQUIRE(outputKeys.CopyToHost(stream) == std::vector<T>{3, 3, 3, 3, -7, -7, 3, 3, 3, 3});
        REQUIRE(outputValues.CopyToHost(stream) == std::vector<T>{-8, -8, 2, 9, -3, 4, -8, -8, 2, 9});
        Device<T> allKeys(7), allValues(7);
        REQUIRE(map.RetrieveAll(allKeys.Data(), allValues.Data(), Extent(7), stream) == 7);
        std::vector<T> expectedKeys, expectedValues;
        for (auto pair : pairs) {
            expectedKeys.push_back(pair.first);
            expectedValues.push_back(pair.second);
        }
        REQUIRE(allKeys.CopyToHost(stream) == expectedKeys);
        REQUIRE(allValues.CopyToHost(stream) == expectedValues);
    }
}

TEMPLATE_TEST_CASE("static_multimap selected prefix spans scan levels", "[static_multimap][contract]", int32_t, int64_t)
{
    using T = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    constexpr std::size_t n = 66049;
    auto map = MakeMultimap<T, T>(513, stream);
    std::vector<aclco::Pair<T, T>> pairs;
    std::vector<uint32_t> stencil(n);
    std::size_t selected = 0;
    for (std::size_t i = 0; i < n; ++i) {
        stencil[i] = (i % 3 != 0);
        if (stencil[i]) {
            ++selected;
        }
        pairs.emplace_back(T(i + 1), T(i + 7));
    }
    Device<aclco::Pair<T, T>> input(n);
    Device<uint32_t> mask(n);
    input.CopyFromHostAsync(pairs.data(), n, stream);
    mask.CopyFromHostAsync(stencil.data(), n, stream);
    REQUIRE((map.template InsertIf<uint32_t, Nonzero>(input.Data(), mask.Data(), Extent(n), stream)) == selected - 513);
    Device<T> keys(513), values(513);
    REQUIRE(map.RetrieveAll(keys.Data(), values.Data(), Extent(513), stream) == 513);
    std::vector<T> expected;
    for (std::size_t i = 0; expected.size() < 513; ++i) {
        if (stencil[i]) {
            expected.push_back(T(i + 1));
        }
    }
    REQUIRE(keys.CopyToHost(stream) == expected);
    REQUIRE(map.Size(stream) == 513);
    map.Clear(stream);
    REQUIRE(map.Insert(input.Data(), Extent(n), stream) == n - 513);
    REQUIRE(map.RetrieveAll(keys.Data(), values.Data(), Extent(513), stream) == 513);
    std::iota(expected.begin(), expected.end(), T(1));
    REQUIRE(keys.CopyToHost(stream) == expected);
}

TEMPLATE_TEST_CASE("static_multimap validation preserves state and output", "[static_multimap][contract]", int32_t,
                   int64_t)
{
    using T = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto map = MakeMultimap<T, T>(8, stream);
    std::vector<aclco::Pair<T, T>> pairs{{1, 4}, {2, EmptyValue<T>()}, {EmptyKey<T>(), 7}};
    Device<aclco::Pair<T, T>> input(3);
    input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    REQUIRE_THROWS_AS(map.Insert(input.Data(), Extent(3), stream), std::invalid_argument);
    REQUIRE(map.Size(stream) == 0);
    std::vector<uint32_t> stencil{1, 0, 0};
    Device<uint32_t> mask(3);
    mask.CopyFromHostAsync(stencil.data(), 3, stream);
    REQUIRE((map.template InsertIf<uint32_t, Nonzero>(input.Data(), mask.Data(), Extent(3), stream)) == 0);
    REQUIRE(map.Size(stream) == 1);
    REQUIRE_THROWS_AS((map.template InsertIf<uint32_t, Nonzero>(input.Data(), nullptr, Extent(3), stream)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((map.template InsertIf<uint32_t, Nonzero>(nullptr, mask.Data(), Extent(3), stream)),
                      std::invalid_argument);
    std::vector<T> query{1, 1}, sentinel{T(123), T(123)};
    Device<T> keys(2), outKeys(2), outValues(2);
    keys.CopyFromHostAsync(query.data(), 2, stream);
    outKeys.CopyFromHostAsync(sentinel.data(), 2, stream);
    outValues.CopyFromHostAsync(sentinel.data(), 2, stream);
    REQUIRE_THROWS_AS(map.Retrieve(keys.Data(), Extent(2), outKeys.Data(), outValues.Data(), Extent(1), stream),
                      std::length_error);
    REQUIRE(outKeys.CopyToHost(stream) == sentinel);
    REQUIRE(outValues.CopyToHost(stream) == sentinel);
    REQUIRE_THROWS_AS(map.RetrieveAll(outKeys.Data(), outValues.Data(), Extent(0), stream), std::length_error);
    REQUIRE_THROWS_AS(map.Find(nullptr, outValues.Data(), Extent(1), stream), std::invalid_argument);
    REQUIRE_THROWS_AS(map.Contains(keys.Data(), nullptr, Extent(1), stream), std::invalid_argument);
    REQUIRE_THROWS_AS(map.Count(keys.Data(), Extent(std::numeric_limits<std::size_t>::max()), stream),
                      std::overflow_error);
    REQUIRE(map.Size(stream) == 1);
}

TEST_CASE("static_multimap scan keeps uint64 totals across chunks", "[static_multimap][contract]")
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    for (std::size_t n : {1, 31, 32, 33, 255, 256, 257, 65537}) {
        std::vector<uint64_t> input(n + 1), expected(n + 1);
        for (std::size_t i = 0; i < n; ++i) {
            // Exercise low-word overflow as well as nonzero high words.
            input[i] = (uint64_t(1) << 32) - 9 + i % 17;
            expected[i + 1] = expected[i] + input[i];
        }
        input[n] = std::numeric_limits<uint64_t>::max();
        Device<uint64_t> values(n + 1);
        values.CopyFromHostAsync(input.data(), input.size(), sg.stream);
        std::vector<aclco::detail::multimap::Buffer<uint64_t>> workspace;
        aclco::detail::multimap::Prefix(values.Data(), n, sg.stream, workspace);
        auto actual = values.CopyToHost(sg.stream);
        for (std::size_t i = 0; i <= n; ++i) {
            REQUIRE(actual[i] == expected[i]);
        }
    }
}

TEMPLATE_TEST_CASE("static_multimap moves ownership and supports integer boundaries", "[static_multimap][contract]",
                   int32_t, int64_t)
{
    using T = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    // Non-minimum sentinels allow the complete signed boundaries in payloads.
    aclco::StaticMultimap<T, T> map(Extent(5), T(17), T(19), stream);
    std::vector<aclco::Pair<T, T>> pairs{{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()},
                                         {std::numeric_limits<T>::max(), std::numeric_limits<T>::min()},
                                         {-1, -1}};
    Device<aclco::Pair<T, T>> input(3);
    input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    REQUIRE(map.Insert(input.Data(), Extent(3), stream) == 0);
    auto moved = std::move(map);
    REQUIRE(map.Capacity() == 0);
    REQUIRE_THROWS_AS(map.Clear(stream), std::logic_error);
    auto assigned = MakeMultimap<T, T>(2, stream);
    assigned = std::move(moved);
    REQUIRE(assigned.Size(stream) == 3);
    Device<T> keys(3), values(3);
    REQUIRE(assigned.RetrieveAll(keys.Data(), values.Data(), Extent(3), stream) == 3);
    REQUIRE(keys.CopyToHost(stream) ==
            std::vector<T>{std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), -1});
    REQUIRE(values.CopyToHost(stream) ==
            std::vector<T>{std::numeric_limits<T>::max(), std::numeric_limits<T>::min(), -1});
    std::vector<T> query{std::numeric_limits<T>::max(), -1, std::numeric_limits<T>::min()};
    Device<T> probes(query.size());
    probes.CopyFromHostAsync(query.data(), query.size(), stream);
    REQUIRE(assigned.Retrieve(probes.Data(), Extent(3), keys.Data(), values.Data(), Extent(3), stream) == 3);
    REQUIRE(keys.CopyToHost(stream) == query);
    REQUIRE(values.CopyToHost(stream) ==
            std::vector<T>{std::numeric_limits<T>::min(), -1, std::numeric_limits<T>::max()});
}

TEMPLATE_TEST_CASE("static_multimap dense export preserves accepted insertion order", "[static_multimap][contract]",
                   int32_t, int64_t)
{
    using T = TestType;
    using Pair = aclco::Pair<T, T>;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    constexpr std::size_t n = 1027;
    auto map = MakeMultimap<T, T>(2 * n, stream);
    std::vector<Pair> pairs(n);
    // Repeated signed keys and values distinguish accepted insertion order from
    // key ordering. The public contract requires the complete multiset only.
    for (std::size_t i = 0; i < n; ++i) {
        pairs[i] = Pair{T(int(i / 9) - 70), T(int(i % 9) / 2 - 7)};
    }
    Device<Pair> input(n);
    Device<T> keys(2 * n), values(2 * n);
    std::vector<Pair> expected;
    auto verify = [&] {
        REQUIRE(map.RetrieveAll(keys.Data(), values.Data(), Extent(2 * n), stream) == expected.size());
        auto actualKeys = keys.CopyToHost(stream);
        auto actualValues = values.CopyToHost(stream);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            REQUIRE(actualKeys[i] == expected[i].first);
            REQUIRE(actualValues[i] == expected[i].second);
        }
    };
    input.CopyFromHostAsync(pairs.data(), n, stream);
    REQUIRE(map.Insert(input.Data(), Extent(n), stream) == 0);
    expected = pairs;
    verify();
    // A second batch follows the first without device sorting.
    REQUIRE(map.Insert(input.Data(), Extent(n), stream) == 0);
    expected.insert(expected.end(), pairs.begin(), pairs.end());
    verify();
    map.Clear(stream);
    // An inversion across a warp/chunk boundary must also retain its input rank.
    std::swap(pairs[255], pairs[256]);
    pairs[255].first = pairs[256].first;
    pairs[255].second = T(42);
    pairs[256].second = T(-42);
    input.CopyFromHostAsync(pairs.data(), n, stream);
    REQUIRE(map.Insert(input.Data(), Extent(n), stream) == 0);
    expected = pairs;
    verify();
}

TEST_CASE("static_multimap I64 hash tags never replace key equality", "[static_multimap][contract]")
{
    using T = int64_t;
    // Fmix64 outputs differ in their high word and share 0x12345678 below.
    constexpr T a = -2779099081600048012LL;
    constexpr T b = -4184281966385060097LL;
    constexpr T missing = -1175811474899109790LL;
    REQUIRE(uint32_t(aclco::murmurhash3_fmix64<T>{}(a)) == uint32_t(aclco::murmurhash3_fmix64<T>{}(b)));
    REQUIRE(uint32_t(aclco::murmurhash3_fmix64<T>{}(a)) == uint32_t(aclco::murmurhash3_fmix64<T>{}(missing)));
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto map = MakeMultimap<T, T>(7, stream);
    std::vector<aclco::Pair<T, T>> pairs{{a, 41}, {b, 17}, {a, -7}, {b, 19}};
    Device<aclco::Pair<T, T>> input(pairs.size());
    input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    REQUIRE(map.Insert(input.Data(), Extent(pairs.size()), stream) == 0);
    std::vector<T> queries{a, b, missing};
    Device<T> keys(3), found(3), allKeys(4), allValues(4);
    Device<uint8_t> contains(3);
    keys.CopyFromHostAsync(queries.data(), queries.size(), stream);
    map.Contains(keys.Data(), contains.Data(), Extent(3), stream);
    REQUIRE(contains.CopyToHost(stream) == std::vector<uint8_t>{1, 1, 0});
    map.Find(keys.Data(), found.Data(), Extent(3), stream);
    REQUIRE(found.CopyToHost(stream) == std::vector<T>{-7, 17, EmptyValue<T>()});
    REQUIRE(map.Count(keys.Data(), Extent(3), stream) == 4);
    REQUIRE(map.Retrieve(keys.Data(), Extent(3), allKeys.Data(), allValues.Data(), Extent(4), stream) == 4);
    REQUIRE(allKeys.CopyToHost(stream) == std::vector<T>{a, a, b, b});
    REQUIRE(allValues.CopyToHost(stream) == std::vector<T>{-7, 41, 17, 19});
    Device<T> cachedKeys(6), cachedValues(6);
    REQUIRE(map.Retrieve(keys.Data(), Extent(3), cachedKeys.Data(), cachedValues.Data(), Extent(6), stream) == 4);
    auto cachedKeyResult = cachedKeys.CopyToHost(stream), cachedValueResult = cachedValues.CopyToHost(stream);
    cachedKeyResult.resize(4); cachedValueResult.resize(4);
    REQUIRE(cachedKeyResult == std::vector<T>{a, a, b, b});
    REQUIRE(cachedValueResult == std::vector<T>{-7, 41, 17, 19});

}

TEST_CASE("static_multimap I32 multiplication hash handles long probe chains", "[static_multimap][contract]")
{
    using T = int32_t;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    constexpr std::size_t n = 257;
    auto map = MakeMultimap<T, T>(n, stream);
    std::vector<aclco::Pair<T, T>> pairs;
    std::vector<T> query;
    // 0x144cbc89 is the inverse of 0x9e3779b9 modulo 2^32. All generated
    // hashes fall in the first range-reduction bucket at this capacity.
    static_assert(uint32_t(0x144cbc89u * 0x9e3779b9u) == 1);
    for (uint32_t i = 1; i <= n; ++i) {
        T key = T(i * 0x144cbc89u);
        pairs.emplace_back(key, T(i + 700));
        query.push_back(key);
    }
    query.push_back(T(258u * 0x144cbc89u));
    Device<aclco::Pair<T, T>> input(n);
    Device<T> keys(query.size()), values(query.size());
    input.CopyFromHostAsync(pairs.data(), n, stream);
    keys.CopyFromHostAsync(query.data(), query.size(), stream);
    REQUIRE(map.Insert(input.Data(), Extent(n), stream) == 0);
    REQUIRE(map.Count(keys.Data(), Extent(query.size()), stream) == n);
    map.Find(keys.Data(), values.Data(), Extent(query.size()), stream);
    auto actual = values.CopyToHost(stream);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(actual[i] == pairs[i].second);
    }
    REQUIRE(actual[n] == EmptyValue<T>());
}
