/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifdef ACLCO_MULTIMAP_SCRATCH_BENCH
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "static_multimap.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace {
bool measuring = false;
uint64_t allocations = 0, frees = 0;
}
extern "C" aclError __real_aclrtMalloc(void**, size_t, aclrtMemMallocPolicy);
extern "C" aclError __wrap_aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy policy)
{
    if (measuring) ++allocations;
    return __real_aclrtMalloc(p, n, policy);
}
extern "C" aclError __real_aclrtFree(void*);
extern "C" aclError __wrap_aclrtFree(void* p)
{
    if (measuring && p) ++frees;
    return __real_aclrtFree(p);
}

struct ScratchMeasurement {
    uint64_t coldAllocations = 0, coldFrees = 0;
    std::vector<double> samples;
};

template <class Call>
ScratchMeasurement MeasureScratch(Call&& call, size_t repeats, bool expectReuse)
{
    ScratchMeasurement result;
    allocations = frees = 0;
    measuring = true;
    call();
    measuring = false;
    result.coldAllocations = allocations;
    result.coldFrees = frees;
    allocations = frees = 0;
    auto& samples = result.samples;
    samples.reserve(repeats);
    measuring = true;
    for (size_t i = 0; i < repeats; ++i) {
        auto start = std::chrono::steady_clock::now();
        call();
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    measuring = false;
    if (expectReuse && (allocations != 0 || frees != 0))
        throw std::runtime_error("scratch steady-state allocation/free is not zero");
    return result;
}

template <class T>
void ReportScratch(size_t n, size_t multiplicity, size_t hitPercent, bool retrieve,
                   size_t repeats, ScratchMeasurement& result)
{
    auto& samples = result.samples;
    double total = std::accumulate(samples.begin(), samples.end(), 0.0);
    std::sort(samples.begin(), samples.end());
    std::cout << sizeof(T) * 8 << ',' << n << ',' << multiplicity << ',' << hitPercent << ','
              << (retrieve ? "retrieve" : "count") << ',' << repeats << ',' << result.coldAllocations << ','
              << result.coldFrees << ',' << allocations << ',' << frees << ',' << total / repeats << ','
              << (samples[(repeats - 1) / 2] + samples[repeats / 2]) / 2 << ','
              << samples[(repeats * 95 + 99) / 100 - 1] << ',' << total << std::endl;
}

template <class T>
void ScratchCase(size_t n, size_t multiplicity, size_t hitPercent, bool retrieve,
                 size_t repeats, bool expectReuse, aclrtStream stream)
{
    using Map = aclco::StaticMultimap<T, T>;
    using Pair = aclco::Pair<T, T>;
    constexpr size_t distinct = 4096;
    T empty = std::numeric_limits<T>::lowest();
    Map map(aclco::Extent<size_t>{distinct * multiplicity * 2}, empty, empty, stream);
    std::vector<Pair> pairs;
    for (size_t k = 1; k <= distinct; ++k)
        for (size_t v = 0; v < multiplicity; ++v)
            pairs.emplace_back(T(k), T(k * 8 + v + 1));
    aclco::test::DeviceBuffer<Pair> input(pairs.size());
    input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    if (map.Insert(input.Data(), aclco::Extent<size_t>{pairs.size()}, stream) != 0)
        throw std::runtime_error("scratch benchmark insertion failed");
    auto hits = n * hitPercent / 100;
    auto expected = hits * multiplicity;
    std::vector<T> queries(n);
    for (size_t i = 0; i < n; ++i)
        queries[i] = T(i < hits ? i % distinct + 1 : distinct + i + 1);
    aclco::test::DeviceBuffer<T> keys(n), outKeys(expected), outValues(expected);
    keys.CopyFromHostAsync(queries.data(), n, stream);
    aclco::test::Sync(stream);
    auto call = [&] {
        auto total = retrieve
            ? map.Retrieve(keys.Data(), aclco::Extent<size_t>{n}, outKeys.Data(), outValues.Data(),
                           aclco::Extent<size_t>{expected}, stream)
            : map.Count(keys.Data(), aclco::Extent<size_t>{n}, stream);
        if (total != expected) throw std::runtime_error("scratch benchmark result count mismatch");
    };
    auto measurement = MeasureScratch(call, repeats, expectReuse);
    if (retrieve && expected) {
        auto returnedKeys = outKeys.CopyToHost(stream), returnedValues = outValues.CopyToHost(stream);
        for (size_t i = 0; i < expected; ++i)
            if (returnedKeys[i] != queries[i / multiplicity] ||
                returnedValues[i] != T(queries[i / multiplicity] * 8 + i % multiplicity + 1))
                throw std::runtime_error("scratch benchmark query ordering or value mismatch");
    }
    ReportScratch<T>(n, multiplicity, hitPercent, retrieve, repeats, measurement);
}

template <class T>
void ScratchCases(size_t repeats, size_t batchFilter, bool expectReuse, aclrtStream stream)
{
    for (size_t n : {size_t{1}, size_t{1024}, size_t{100000}}) {
        if (batchFilter && n != batchFilter) continue;
        for (size_t multiplicity : {size_t{1}, size_t{4}})
            for (size_t hit : {size_t{0}, size_t{50}, size_t{100}}) {
                if (repeats > 1000 && (n != 1024 || multiplicity != 4 || hit != 100)) continue;
                for (bool retrieve : {false, true})
                    ScratchCase<T>(n, multiplicity, hit, retrieve, repeats, expectReuse, stream);
            }
    }
}

#ifdef ACLCO_MULTIMAP_SCRATCH_LIBRARY
extern "C" __attribute__((visibility("default")))
int run_scratch_case(int bits, size_t n, size_t multiplicity, size_t hitPercent,
                     bool retrieve, size_t repeats, bool expectReuse, aclrtStream stream)
{
    try {
        if (bits == 32)
            ScratchCase<int32_t>(n, multiplicity, hitPercent, retrieve, repeats, expectReuse, stream);
        else
            ScratchCase<int64_t>(n, multiplicity, hitPercent, retrieve, repeats, expectReuse, stream);
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
#else
int main(int argc, char** argv)
{
    size_t repeats = argc > 1 ? std::stoull(argv[1]) : 1000;
    size_t filter = argc > 2 ? std::stoull(argv[2]) : 0;
    bool expectReuse = argc > 3 && std::string(argv[3]) == "--expect-reuse";
    if (!repeats || repeats > 100000 || (filter && filter != 1 && filter != 1024 && filter != 100000))
        throw std::invalid_argument("usage: multimap_query_scratch [1..100000 repeats] [0|1|1024|100000] [--expect-reuse]");
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard stream;
    std::cout << "bits,batch,multiplicity,hit_pct,operation,repeats,cold_allocs,cold_frees,steady_allocs,steady_frees,mean_us,median_us,p95_us,total_us\n";
    ScratchCases<int32_t>(repeats, filter, expectReuse, stream.stream);
    ScratchCases<int64_t>(repeats, filter, expectReuse, stream.stream);
}
#endif
#else
#include "static_multimap_perf_common.h"

namespace aclco::test {
namespace {

template <typename Key, typename Value>
using Context = QueryContext<Key, Value>;

template <typename Key, typename Value>
Context<Key, Value>& GetContext()
{
    return GetContextInstance<Context<Key, Value>>();
}

template <typename Key, typename Value>
void SetupCount(std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity, double matchingRate)
{
    auto& context = GetContext<Key, Value>();
    SetupPairs<Key, Value>(context, numInputs, occupancy, multiplicity);
    context.map->Insert(context.pairs.Data(), aclco::Extent<std::size_t>(context.numInputs),
                        context.streamGuard.stream);
    SetupQueries<Key>(context, matchingRate);

    Sync(context.streamGuard.stream);
}

template <typename Key, typename Value>
TestResult TestCount()
{
    auto& context = GetContext<Key, Value>();
    return Measure([&] {
        (void)context.map->Count(context.queries.Data(), aclco::Extent<std::size_t>(context.numInputs),
                                 context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultimapCountI32, (TestCount<std::int32_t, std::int32_t>),
                          (SetupCount<std::int32_t, std::int32_t>), std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_TEST(staticMultimapCountI64, (TestCount<std::int64_t, std::int64_t>),
                          (SetupCount<std::int64_t, std::int64_t>), std::uint64_t, double, std::uint64_t, double);

REGISTER_PERFORMANCE_ARGS(staticMultimapCountI32,
                          "StaticMultimap count (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_ARGS(staticMultimapCountI64,
                          "StaticMultimap count (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
} // namespace aclco::test
#endif
