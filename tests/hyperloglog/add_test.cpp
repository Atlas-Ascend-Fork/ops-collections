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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/acl_env.h"
#include "common/device_buffer.h"
#include "common/hll_test_common.h"
#include "hyperloglog.h"

using namespace aclco::test::hll_cases;

TEMPLATE_TEST_CASE_SIG("hyperloglog add correctness", "[hyperloglog][add]", ((typename Key, int Dummy), Key, Dummy),
                       (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    auto key_num = GENERATE(0u, 1u, 2u, 3u, 7u, 31u, 128u, 1024u, 4096u, 8195u, 32768u, 131072u);
    CAPTURE(sketch_kb, key_num);

    SECTION("unique and duplicate distributions")
    {
        auto distinct = GENERATE_COPY(std::max<std::size_t>(key_num, 1u), std::max<std::size_t>(key_num / 2u, 1u),
                                      std::max<std::size_t>(key_num / 8u, 1u));
        auto keys = MakeKeys<Key>(key_num, distinct, 1u);
        auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
        auto estimate = AddAndEstimate(hll, keys, stream);
        REQUIRE(EstimateWithinAcceptance(estimate, ExactCardinality(keys), sketch_kb));
    }

    SECTION("batched add equals one-shot estimate")
    {
        auto keys = MakeKeys<Key>(key_num, std::max<std::size_t>(key_num / 2u, 1u), 17u);
        auto one_shot = MakeBySketchSize<Key>(sketch_kb, stream);
        auto batched = MakeBySketchSize<Key>(sketch_kb, stream);
        auto expected = AddAndEstimate(one_shot, keys, stream);

        auto split = keys.size() / 2u;
        std::vector<Key> left(keys.begin(), keys.begin() + split);
        std::vector<Key> right(keys.begin() + split, keys.end());
        AddAndEstimate(batched, left, stream);
        auto actual = AddAndEstimate(batched, right, stream);
        REQUIRE(actual == expected);
    }

    SECTION("null pointer with nonzero count is rejected")
    {
        auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
        REQUIRE_THROWS(hll.Add(nullptr, aclco::Extent<std::size_t>(1u), stream));
    }
}

TEMPLATE_TEST_CASE_SIG("hyperloglog add large generalization", "[hyperloglog][add][large]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    constexpr std::size_t key_num = 100000000ULL;
    auto keys = MakeKeys<Key>(key_num, key_num, 2026u);
    auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
    auto estimate = AddAndEstimate(hll, keys, stream);
    REQUIRE(EstimateWithinAcceptance(estimate, key_num, sketch_kb));

    auto batched = MakeBySketchSize<Key>(sketch_kb, stream);
    aclco::test::DeviceBuffer<Key> input(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), stream);
    auto split = keys.size() / 2u;
    batched.Add(input.Data(), aclco::Extent<std::size_t>(split), stream);
    batched.Add(input.Data() + split, aclco::Extent<std::size_t>(keys.size() - split), stream);
    REQUIRE(batched.Estimate(stream) == estimate);
}

/* The cases below cross the largest element block a single Add call may take.
 * They are tagged [.] because a run reserves more than 8 GiB of device memory. */
namespace {
using aclco::test::CheckAcl;
using aclco::test::DeviceBuffer;
using Extent = aclco::Extent<std::size_t>;

// Largest element block handed to a single Add call in this regression.
constexpr std::size_t kHalf = std::size_t{1} << 30;
constexpr std::size_t kLimit = 0x7fffffffULL;

void RequireLarge(bool condition, char const* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Key>
std::vector<Key> SeedKeys()
{
    return {static_cast<Key>(1234567), static_cast<Key>(7654321)};
}

template <class Key>
std::vector<Key> AfterKeys()
{
    return {static_cast<Key>(24681357), static_cast<Key>(97531864)};
}

template <class Key>
std::vector<std::uint8_t> Registers(aclco::HyperLogLog<Key> const& h, unsigned p)
{
    std::vector<std::uint8_t> result(std::size_t{1} << p);
    CheckAcl(aclrtMemcpy(result.data(), result.size(), h.Data(), result.size(), ACL_MEMCPY_DEVICE_TO_HOST),
             "copy all register bytes");
    return result;
}

/* input is borrowed and sized for the largest case; only its first n elements
 * are used. The three aux buffers are uploaded once per case. */
template <class Key>
class LargeInputs {
public:
    LargeInputs(DeviceBuffer<Key>& source, std::size_t count, std::vector<Key> markers, aclrtStream s)
        : input(source), n(count), compact(std::move(markers)), stream(s)
    {
        auto const seed = SeedKeys<Key>();
        auto const after = AfterKeys<Key>();
        small.Resize(compact.size());
        pre.Resize(seed.size());
        post.Resize(after.size());
        small.CopyFromHostAsync(compact.data(), compact.size(), stream);
        pre.CopyFromHostAsync(seed.data(), seed.size(), stream);
        post.CopyFromHostAsync(after.data(), after.size(), stream);
    }

    DeviceBuffer<Key>& input;
    std::size_t n;
    std::vector<Key> compact;
    aclrtStream stream;
    DeviceBuffer<Key> small;
    DeviceBuffer<Key> pre;
    DeviceBuffer<Key> post;
};

/* Most inputs are zero, but each chunk boundary and the very last element carry
 * distinguishable markers. A repeated block alone cannot catch a truncated
 * prefix: the small slow-path reference contains each marker once. */
template <class Key>
std::vector<Key> PrepareInput(DeviceBuffer<Key>& input, std::size_t n, aclrtStream stream)
{
    CheckAcl(aclrtMemsetAsync(input.Data(), input.Bytes(), 0, input.Bytes(), stream), "initialize large input");
    CheckAcl(aclrtSynchronizeStream(stream), "wait for input initialization");
    std::vector<std::size_t> positions{0, kHalf - 1, kHalf, n - 2, n - 1};
    if (n > 2 * kHalf)
        positions.push_back(2 * kHalf);
    if (n > 3 * kHalf)
        positions.push_back(3 * kHalf);
    if (n > 4 * kHalf)
        positions.push_back(4 * kHalf);
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    std::vector<Key> compact{Key{0}};
    for (std::size_t i = 0; i < positions.size(); ++i) {
        Key marker = static_cast<Key>(i + 1);
        compact.push_back(marker);
        CheckAcl(aclrtMemcpy(input.Data() + positions[i], sizeof(Key), &marker, sizeof(Key), ACL_MEMCPY_HOST_TO_DEVICE),
                 "write boundary marker");
    }
    return compact;
}

/* Every chosen marker must be observable at this precision, otherwise the
 * register oracle would accept an Add that silently dropped a chunk or tail. */
template <class Key>
void RequireMarkersDiscriminate(LargeInputs<Key>& in, aclco::HyperLogLog<Key>& expected,
                                std::vector<std::uint8_t> const& markerSketch, unsigned p)
{
    for (std::size_t i = 1; i < in.compact.size(); ++i) {
        auto missing = in.compact;
        missing.erase(missing.begin() + i);
        DeviceBuffer<Key> subset(missing.size());
        subset.CopyFromHostAsync(missing.data(), missing.size(), in.stream);
        expected.Clear(in.stream);
        expected.Add(subset.Data(), Extent{missing.size()}, in.stream);
        RequireLarge(Registers(expected, p) != markerSketch, "marker not discriminating; revise test keys");
    }
}

template <class Key>
void AddLarge(aclco::HyperLogLog<Key>& target, LargeInputs<Key>& in, char const* mode)
{
    if (std::string(mode) == "one") {
        target.Add(in.input.Data(), Extent{in.n}, in.stream);
        return;
    }
    for (std::size_t offset = 0; offset < in.n;) {
        auto const batch = std::min(kHalf, in.n - offset);
        target.Add(in.input.Data() + offset, Extent{batch}, in.stream);
        offset += batch;
    }
}

struct Measurement {
    std::size_t keyBytes;
    unsigned p;
    std::size_t n;
    char const* mode;
    unsigned round;
    double milliseconds;
    unsigned long long estimate;
};

void Report(Measurement const& m)
{
    std::printf("HLL_LARGE key_bytes=%zu p=%u n=%zu mode=%s round=%u ms=%.6f estimate=%llu registers=exact "
                "continuation=exact\n",
                m.keyBytes, m.p, m.n, m.mode, m.round, m.milliseconds, m.estimate);
}

struct RoundTag {
    unsigned p;
    char const* mode;
    unsigned round;
};

template <class Key>
void RunRound(LargeInputs<Key>& in, aclco::HyperLogLog<Key>& actual, aclco::HyperLogLog<Key>& expected,
              std::vector<std::uint8_t> const& expectedLarge, RoundTag const& tag)
{
    actual.Clear(in.stream);
    actual.Add(in.pre.Data(), Extent{in.pre.Size()}, in.stream);
    auto const start = std::chrono::steady_clock::now();
    AddLarge(actual, in, tag.mode);
    auto const stop = std::chrono::steady_clock::now();
    RequireLarge(Registers(actual, tag.p) == expectedLarge, "large Add exact-register mismatch");
    auto const estimate = actual.Estimate(in.stream);
    RequireLarge(estimate == expected.Estimate(in.stream), "large Add estimate mismatch");
    actual.Add(in.post.Data(), Extent{in.post.Size()}, in.stream);
    // Separate reference keeps the timed input oracle unchanged.
    auto continued = aclco::HyperLogLog<Key>::CreateWithPrecision(tag.p, in.stream);
    continued.Merge(expected, in.stream);
    continued.Add(in.post.Data(), Extent{in.post.Size()}, in.stream);
    auto const afterSketch = Registers(continued, tag.p);
    RequireLarge(afterSketch != expectedLarge, "post-Add oracle not discriminating");
    RequireLarge(Registers(actual, tag.p) == afterSketch, "post-large continuation mismatch");
    Report(Measurement{sizeof(Key), tag.p, in.n, tag.mode, tag.round,
                       std::chrono::duration<double, std::milli>(stop - start).count(),
                       static_cast<unsigned long long>(estimate)});
}

template <class Key>
void RunPrecision(LargeInputs<Key>& in, unsigned p)
{
    auto actual = aclco::HyperLogLog<Key>::CreateWithPrecision(p, in.stream);
    auto expected = aclco::HyperLogLog<Key>::CreateWithPrecision(p, in.stream);
    expected.Add(in.small.Data(), Extent{in.small.Size()}, in.stream);
    auto const markerSketch = Registers(expected, p);
    RequireMarkersDiscriminate(in, expected, markerSketch, p);
    expected.Clear(in.stream);
    expected.Add(in.pre.Data(), Extent{in.pre.Size()}, in.stream);
    auto const seedSketch = Registers(expected, p);
    expected.Add(in.small.Data(), Extent{in.small.Size()}, in.stream);
    auto const expectedLarge = Registers(expected, p);
    RequireLarge(expectedLarge != markerSketch && expectedLarge != seedSketch,
                 "seed/input continuation oracle is not discriminating");
    for (char const* mode : {"one", "public_chunks"}) {
        for (unsigned round = 0; round < 3; ++round) {
            RunRound(in, actual, expected, expectedLarge, RoundTag{p, mode, round});
        }
    }
}

template <class Key>
void RunLarge(aclrtStream stream)
{
    bool const multi = std::getenv("HLL_TEST_MULTI_CHUNK") != nullptr;
    std::size_t const maximum = multi ? 4 * kHalf + 257 : 2 * kHalf + 257;
    std::size_t freeBytes = 0, totalBytes = 0;
    CheckAcl(aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes), "query free HBM");
    RequireLarge(freeBytes > maximum * sizeof(Key) + (std::size_t{256} << 20), "insufficient HBM for large Add test");
    DeviceBuffer<Key> input(maximum);
    std::vector<std::size_t> counts{kLimit, kLimit + 1, 2 * kHalf + 257};
    if (multi)
        counts.push_back(maximum);
    for (auto n : counts) {
        LargeInputs<Key> in(input, n, PrepareInput(input, n, stream), stream);
        for (unsigned p = 13; p <= 18; ++p) {
            RunPrecision(in, p);
        }
    }
}

void RunLargeTests()
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard stream;
    RunLarge<std::int32_t>(stream.stream);
    RunLarge<std::int64_t>(stream.stream);
}
} // namespace

TEST_CASE("hyperloglog large Add preserves every chunk and continuation", "[.][hyperloglog][add][large-boundary]")
{
    RunLargeTests();
}
