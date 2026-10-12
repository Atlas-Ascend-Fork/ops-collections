/*
 * Copyright (c) liaopinyi714. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

#include "extent.h"
#include "hash_functions.h"
#include "probing_scheme.h"
#include "storage.h"
#include "utility/math_utils.h"

TEST_CASE("IntDivCeil handles nonnegative values at integer limits", "[math-utils][int-div-ceil]")
{
    SECTION("Unsigned exact division")
    {
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::uint32_t>::max(), 5) == 858993459u);
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::uint64_t>::max(), std::uint64_t{5}) == 3689348814741910323ull);
    }
    SECTION("Unsigned division with a remainder")
    {
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::uint32_t>::max(), 2) == 2147483648u);
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::uint64_t>::max(), std::uint64_t{2}) == 9223372036854775808ull);
    }
    SECTION("Signed positive division")
    {
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::int32_t>::max(), 2) == 1073741824);
        CHECK(aclco::IntDivCeil(std::numeric_limits<std::int64_t>::max(), std::int64_t{2}) == 4611686018427387904ll);
    }
}

TEST_CASE("IntDivCeil keeps ordinary and zero-input results", "[math-utils][int-div-ceil]")
{
    CHECK(aclco::IntDivCeil(0u, 5) == 0u);
    CHECK(aclco::IntDivCeil(0, 5) == 0);
    CHECK(aclco::IntDivCeil(5u, 5) == 1u);
    CHECK(aclco::IntDivCeil(6u, 5) == 2u);
    CHECK(aclco::IntDivCeil(42u, 1) == 42u);
    CHECK(aclco::IntDivCeil(42u, 0) == 0u);
    constexpr auto rounded = aclco::IntDivCeil(std::uint64_t{24}, std::uint64_t{5});
    CHECK(rounded == 5);
}

TEST_CASE("MakeValidExtent retains representable uint32 capacities", "[math-utils][extent-ceil]")
{
    using Size = std::uint32_t;
    using Storage = aclco::Storage<5>;
    using Probing = aclco::LinearProbing<aclco::murmurhash3_32<Size>>;
    constexpr Size largest = std::numeric_limits<Size>::max();

    SECTION("Dynamic exact and inexact capacities round to the same valid maximum")
    {
        for (Size distance = 0; distance < 5; ++distance) {
            auto const result = aclco::MakeValidExtent<Probing, Storage>(aclco::Extent<Size>{largest - distance});
            CHECK(static_cast<Size>(result) == largest);
        }
    }
    SECTION("Static capacity remains constant-evaluable")
    {
        constexpr auto result = aclco::MakeValidExtent<Probing, Storage>(aclco::Extent<Size, largest>{});
        CHECK(static_cast<Size>(result) == largest);
    }
    SECTION("Ordinary and zero extent controls")
    {
        CHECK(static_cast<Size>(aclco::MakeValidExtent<Probing, Storage>(aclco::Extent<Size>{0})) == 10);
        CHECK(static_cast<Size>(aclco::MakeValidExtent<Probing, Storage>(aclco::Extent<Size>{1})) == 5);
        CHECK(static_cast<Size>(aclco::MakeValidExtent<Probing, Storage>(aclco::Extent<Size>{6})) == 10);
    }
}
