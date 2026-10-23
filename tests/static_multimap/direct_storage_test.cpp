/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#include "common/static_multimap_test_common.h"

namespace aclco::detail::multimap {
// This test-only access object leaves the public constructor unchanged.
struct StaticMultimapTestAccess {
    template <class T>
    static StaticMultimap<T, T> Direct(std::size_t capacity, T emptyKey, T emptyValue, aclrtStream stream)
    {
        return StaticMultimap<T, T>(Extent<std::size_t>(capacity), emptyKey, emptyValue, stream, StorageMode::Direct);
    }
    template <class T>
    static bool UsesDirectSlots(StaticMultimap<T, T> const& map)
    {
        return map.Get().indices.Data() == nullptr;
    }
};
} // namespace aclco::detail::multimap

namespace {
struct Nonzero {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const { return value != 0; }
};
template <class T>
using Device = aclco::test::DeviceBuffer<T>;
using Extent = aclco::Extent<std::size_t>;
using Access = aclco::detail::multimap::StaticMultimapTestAccess;
using namespace aclco::test::static_multimap;
template <class T>
void BuildDirectOracle(std::vector<aclco::Pair<T, T>> const& admitted, std::vector<T>& queries,
                       std::vector<T>& expectedFind, std::vector<uint8_t>& expectedContains,
                       std::vector<uint32_t>& queryStencil, std::vector<T>& expectedKeys,
                       std::vector<T>& expectedValues)
{
    auto queryCount = queries.size();
    for (std::size_t i = 0; i < queryCount; ++i) {
        queries[i] = i % 9 == 8 ? EmptyKey<T>() : T(int(i % 9) - 3);
        queryStencil[i] = (i % 3 != 0);
        std::vector<T> values;
        for (auto const& pair : admitted) {
            if (pair.first == queries[i]) {
                values.push_back(pair.second);
            }
        }
        std::sort(values.begin(), values.end());
        if (!values.empty()) {
            expectedContains[i] = 1;
            expectedFind[i] = values.front();
        }
        for (auto value : values) {
            expectedKeys.push_back(queries[i]);
            expectedValues.push_back(value);
        }
    }
}
template <class T>
Device<T> CheckDirectQueries(aclco::StaticMultimap<T, T>& map, std::vector<aclco::Pair<T, T>> const& admitted,
                             aclrtStream stream)
{
    constexpr std::size_t queryCount = 515; // Two complete scan chunks and a tail.
    std::vector<T> queries(queryCount), expectedFind(queryCount, EmptyValue<T>());
    std::vector<uint8_t> expectedContains(queryCount, 0);
    std::vector<uint32_t> queryStencil(queryCount);
    std::vector<T> expectedKeys, expectedValues;
    BuildDirectOracle(admitted, queries, expectedFind, expectedContains, queryStencil, expectedKeys, expectedValues);
    Device<T> keys(queryCount), found(queryCount);
    Device<uint8_t> contains(queryCount);
    Device<uint32_t> queryMask(queryCount);
    keys.CopyFromHostAsync(queries.data(), queryCount, stream);
    queryMask.CopyFromHostAsync(queryStencil.data(), queryCount, stream);
    map.Contains(keys.Data(), contains.Data(), Extent(queryCount), stream);
    REQUIRE(contains.CopyToHost(stream) == expectedContains);
    map.Find(keys.Data(), found.Data(), Extent(queryCount), stream);
    REQUIRE(found.CopyToHost(stream) == expectedFind);
    REQUIRE(map.Count(keys.Data(), Extent(queryCount), stream) == expectedValues.size());
    Device<T> outputKeys(expectedKeys.size()), outputValues(expectedValues.size());
    REQUIRE(map.Retrieve(keys.Data(), Extent(queryCount), outputKeys.Data(), outputValues.Data(),
                         Extent(expectedKeys.size()), stream) == expectedKeys.size());
    REQUIRE(outputKeys.CopyToHost(stream) == expectedKeys);
    REQUIRE(outputValues.CopyToHost(stream) == expectedValues);
    REQUIRE_THROWS_AS(map.Retrieve(keys.Data(), Extent(queryCount), outputKeys.Data(), outputValues.Data(),
                                   Extent(expectedKeys.size() - 1), stream),
                      std::length_error);

    map.template ContainsIf<uint32_t, Nonzero>(keys.Data(), queryMask.Data(), contains.Data(), Extent(queryCount),
                                               stream);
    map.template FindIf<uint32_t, Nonzero>(keys.Data(), queryMask.Data(), found.Data(), Extent(queryCount), stream);
    for (std::size_t i = 0; i < queryCount; ++i) {
        if (!queryStencil[i]) {
            expectedContains[i] = 0;
            expectedFind[i] = EmptyValue<T>();
        }
    }
    REQUIRE(contains.CopyToHost(stream) == expectedContains);
    REQUIRE(found.CopyToHost(stream) == expectedFind);

    return keys;
}

template <class T>
void CheckDirectExport(aclco::StaticMultimap<T, T>& map, std::vector<aclco::Pair<T, T>> const& admitted,
                       aclrtStream stream)
{
    Device<T> allKeys(admitted.size()), allValues(admitted.size());
    REQUIRE(map.RetrieveAll(allKeys.Data(), allValues.Data(), Extent(admitted.size()), stream) == admitted.size());
    auto less = [](auto const& a, auto const& b) {
        return a.first < b.first ||
               (a.first == b.first && a.second < b.second);
    };

    auto expected = admitted;
    std::sort(expected.begin(), expected.end(), less);

    auto actualKeys = allKeys.CopyToHost(stream);
    auto actualValues = allValues.CopyToHost(stream);

    std::vector<aclco::Pair<T, T>> actual;
    actual.reserve(actualKeys.size());

    for (std::size_t i = 0; i < actualKeys.size(); ++i) {
        actual.push_back({actualKeys[i], actualValues[i]});
    }

    std::sort(actual.begin(), actual.end(), less);

    REQUIRE(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i].first == expected[i].first);
        REQUIRE(actual[i].second == expected[i].second);
    }
}

using GenericPair = aclco::Pair<uint32_t, uint32_t>;

void CheckPackedPairs(std::vector<GenericPair> actual, std::vector<GenericPair> expected)
{
    actual.erase(std::remove_if(actual.begin(), actual.end(),
                                [](const GenericPair& pair) { return pair.first == 0xffffffffu; }), actual.end());
    auto less = [](const GenericPair& a, const GenericPair& b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    std::sort(expected.begin(), expected.end(), less);
    std::sort(actual.begin(), actual.end(), less);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i].first == expected[i].first);
        REQUIRE(actual[i].second == expected[i].second);
    }
}

void CheckConditionalPairPacking(Device<GenericPair>& table, Device<GenericPair>& input,
                                  Device<uint32_t>& failed, const std::vector<GenericPair>& pairs,
                                  aclrtStream stream)
{
    using namespace aclco::detail::multimap;
    const auto capacity = pairs.size();
    const uint32_t empty = 0xffffffffu;
    const std::vector<uint32_t> stencil{1, 0, 1, 0, 1, 0, 1};
    const std::vector<uint64_t> offsets{0};
    Device<uint32_t> selection(capacity);
    Device<uint64_t> ranks(1);
    selection.CopyFromHostAsync(stencil.data(), capacity, stream);
    ranks.CopyFromHostAsync(offsets.data(), 1, stream);
    failed.MemsetZero(stream);
    Clear<uint32_t, uint32_t><<<1, nullptr, stream>>>(table.Data(), nullptr, capacity, empty, empty);
    Insert<uint32_t, uint32_t, uint32_t, Nonzero, true><<<1, nullptr, stream>>>(
        table.Data(), nullptr, capacity, empty, empty, input.Data(), selection.Data(), capacity,
        ranks.Data(), 4, 0, failed.Data());
    REQUIRE(failed.CopyToHost(stream)[0] == 0);
    std::vector<GenericPair> expected;
    for (std::size_t i = 0; i < capacity; ++i) {
        if (stencil[i]) {
            expected.push_back(pairs[i]);
        }
    }
    CheckPackedPairs(table.CopyToHost(stream), expected);
}
} // namespace

TEST_CASE("static_multimap direct kernel packs generic eight-byte pairs",
          "[static_multimap][direct_storage][pair_packing]")
{
    // The public map accepts signed types only; exercise the generic ref via its kernel.
    using T = uint32_t;
    using P = aclco::Pair<T, T>;
    using namespace aclco::detail::multimap;
    static_assert(sizeof(P) == sizeof(uint64_t));
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    auto stream = guard.stream;
    const std::vector<P> pairs{
        {0u, 0x80000000u}, {0x80000000u, 0u}, {0x12345678u, 0xabcdef01u},
        {0x12345678u, 0xfedcba98u}, {0xfffffffeu, 0xfffffffeu},
        {42u, 0x01020304u}, {42u, 0x01020304u}};
    const auto capacity = pairs.size();
    const T emptyKey = 0xffffffffu;
    const T emptyValue = 0xffffffffu;
    Device<P> table(capacity), input(capacity);
    Device<uint32_t> failed(1);
    input.CopyFromHostAsync(pairs.data(), capacity, stream);
    failed.MemsetZero(stream);
    Clear<T, T><<<1, nullptr, stream>>>(table.Data(), nullptr, capacity, emptyKey, emptyValue);
    Insert<T, T, uint32_t, SelectAll, false><<<1, nullptr, stream>>>(
        table.Data(), nullptr, capacity, emptyKey, emptyValue, input.Data(), nullptr, capacity,
        nullptr, capacity, 0, failed.Data());
    REQUIRE(failed.CopyToHost(stream)[0] == 0);
    CheckPackedPairs(table.CopyToHost(stream), pairs);
    Insert<T, T, uint32_t, SelectAll, false><<<1, nullptr, stream>>>(
        table.Data(), nullptr, capacity, emptyKey, emptyValue, input.Data(), nullptr, 1,
        nullptr, 1, 0, failed.Data());
    REQUIRE(failed.CopyToHost(stream)[0] == 1);

    CheckConditionalPairPacking(table, input, failed, pairs, stream);
}

TEMPLATE_TEST_CASE("static_multimap direct storage covers complete public operations",
                   "[static_multimap][direct_storage]", int32_t, int64_t)
{
    using T = TestType;
    auto capacity = GENERATE(1u, 7u, 257u, 1027u);
    auto full = GENERATE(false, true);
    auto conditional = GENERATE(false, true);
    CAPTURE(capacity, full, conditional);
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    auto stream = guard.stream;
    auto map = Access::Direct<T>(capacity, EmptyKey<T>(), EmptyValue<T>(), stream);
    REQUIRE(Access::UsesDirectSlots(map));
    REQUIRE(map.Capacity() == capacity);
    REQUIRE(map.Size(stream) == 0);
    REQUIRE(map.RetrieveAll(nullptr, nullptr, Extent(0), stream) == 0);
    auto n = full ? 2 * capacity + 9 : std::max(1u, capacity / 2);
    std::vector<aclco::Pair<T, T>> pairs, admitted;
    std::vector<uint32_t> stencil(n);
    std::size_t selected = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // Unsorted values and repeated keys require full probe-chain queries.
        pairs.emplace_back(T(int(i % 5) - 2), T(1000 - int(i % 31)));
        stencil[i] = (i % 2 == 0);
        if (!conditional || stencil[i]) {
            ++selected;
            if (admitted.size() < capacity) {
                admitted.push_back(pairs.back());
            }
        }
    }
    Device<aclco::Pair<T, T>> input(n);
    Device<uint32_t> selection(n);
    input.CopyFromHostAsync(pairs.data(), n, stream);
    selection.CopyFromHostAsync(stencil.data(), n, stream);
    auto failures = conditional ?
                        map.template InsertIf<uint32_t, Nonzero>(input.Data(), selection.Data(), Extent(n), stream) :
                        map.Insert(input.Data(), Extent(n), stream);
    REQUIRE(failures == selected - admitted.size());
    REQUIRE(map.Size(stream) == admitted.size());

    auto keys = CheckDirectQueries(map, admitted, stream);
    constexpr std::size_t queryCount = 515;

    CheckDirectExport(map, admitted, stream);
    map.Clear(stream);
    REQUIRE(map.Size(stream) == 0);
    REQUIRE(map.Count(keys.Data(), Extent(queryCount), stream) == 0);
    REQUIRE(map.RetrieveAll(nullptr, nullptr, Extent(0), stream) == 0);
    REQUIRE(map.Insert(input.Data(), Extent(1), stream) == 0);
    REQUIRE(map.Size(stream) == 1);
    auto moved = std::move(map);
    REQUIRE(Access::UsesDirectSlots(moved));
    REQUIRE(moved.Size(stream) == 1);
}

TEMPLATE_TEST_CASE("static_multimap retrieval spans recursive chunk offsets", "[static_multimap][retrieve_scan]",
                   int32_t, int64_t)
{
    using T = TestType;
    auto direct = GENERATE(false, true);
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    auto stream = guard.stream;
    auto map = direct ? Access::Direct<T>(7, EmptyKey<T>(), EmptyValue<T>(), stream) : MakeMultimap<T, T>(7, stream);
    std::vector<aclco::Pair<T, T>> pairs{{3, std::numeric_limits<T>::max()},
                                         {3, std::numeric_limits<T>::min() + 1},
                                         {3, -8},
                                         {-5, std::numeric_limits<T>::min() + 2}};
    Device<aclco::Pair<T, T>> input(pairs.size());
    input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    REQUIRE(map.Insert(input.Data(), Extent(pairs.size()), stream) == 0);
    for (std::size_t n : {31, 32, 33, 255, 256, 257, 65535, 65536, 65537}) {
        CAPTURE(direct, n);
        std::vector<T> queries(n), expectedKeys, expectedValues;
        for (std::size_t i = 0; i < n; ++i) {
            constexpr T choices[] = {3, -5, 99};
            queries[i] = i % 4 == 3 ? EmptyKey<T>() : choices[i % 4];
            std::vector<T> values;
            for (auto const& pair : pairs) {
                if (pair.first == queries[i]) {
                    values.push_back(pair.second);
                }
            }
            std::sort(values.begin(), values.end());
            for (auto value : values) {
                expectedKeys.push_back(queries[i]);
                expectedValues.push_back(value);
            }
        }
        auto total = expectedKeys.size();
        // Guard elements must survive writes at exact output capacity.
        expectedKeys.insert(expectedKeys.end(), 2, T(123));
        expectedValues.insert(expectedValues.end(), 2, T(123));
        std::vector<T> initial(total + 2, T(123));
        Device<T> keys(n), outputKeys(total + 2), outputValues(total + 2);
        keys.CopyFromHostAsync(queries.data(), n, stream);
        outputKeys.CopyFromHostAsync(initial.data(), initial.size(), stream);
        outputValues.CopyFromHostAsync(initial.data(), initial.size(), stream);
        REQUIRE(map.Retrieve(keys.Data(), Extent(n), outputKeys.Data(), outputValues.Data(), Extent(total), stream) ==
                total);
        REQUIRE(outputKeys.CopyToHost(stream) == expectedKeys);
        REQUIRE(outputValues.CopyToHost(stream) == expectedValues);
    }
}
