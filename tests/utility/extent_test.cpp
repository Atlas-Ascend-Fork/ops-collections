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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "extent.h"
#include "storage.h"
#include "probing_scheme.h"
#include "hash_functions.h"

TEST_CASE("MakeValidExtent rounds up to bucket stride and handles zero", "[extent]")
{
    using Size = std::size_t;
    using Storage = aclco::Storage<5>;
    using Probing = aclco::LinearProbing<aclco::murmurhash3_32<std::uint32_t>>;

    {
        auto out0 = aclco::MakeValidExtent<Probing, Storage, Size, aclco::dynamicExtent>(aclco::Extent<Size>{0});
        REQUIRE(static_cast<Size>(out0) == 10);

        auto out1 = aclco::MakeValidExtent<Probing, Storage, Size, aclco::dynamicExtent>(aclco::Extent<Size>{1});
        REQUIRE(static_cast<Size>(out1) == 5);

        auto out5 = aclco::MakeValidExtent<Probing, Storage, Size, aclco::dynamicExtent>(aclco::Extent<Size>{5});
        REQUIRE(static_cast<Size>(out5) == 5);

        auto out6 = aclco::MakeValidExtent<Probing, Storage, Size, aclco::dynamicExtent>(aclco::Extent<Size>{6});
        REQUIRE(static_cast<Size>(out6) == 10);
    }

    {
        using E7 = aclco::Extent<Size, 7>;
        auto out = aclco::MakeValidExtent<Probing, Storage, Size, 7>(E7{});
        REQUIRE(static_cast<Size>(out) == 10);
    }
}

TEST_CASE("MakeValidExtent using DoubleHashing", "[extent]")
{
    using Size = std::size_t;
    using Storage = aclco::Storage<5>;
    using Probing = aclco::DoubleHashing<aclco::murmurhash3_32<std::uint32_t>>;

    SECTION("Dynamic extent with DoubleHashing")
    {
        aclco::Extent<Size, aclco::dynamicExtent> dynamicExt(10);
        auto result = aclco::MakeValidExtent<Probing, Storage>(dynamicExt);

        REQUIRE(result >= 10);
        REQUIRE(result % Storage::bucketSize == 0);
    }

    SECTION("Static extent with DoubleHashing")
    {
        constexpr std::size_t staticSize = 15;
        aclco::Extent<Size, staticSize> staticExt;
        auto result = aclco::MakeValidExtent<Probing, Storage>(staticExt);
        REQUIRE(result >= staticSize);
        REQUIRE(result % Storage::bucketSize == 0);
    }

    SECTION("Zero extent")
    {
        aclco::Extent<Size, aclco::dynamicExtent> zeroExt(0);
        auto result = aclco::MakeValidExtent<Probing, Storage>(zeroExt);

        REQUIRE(result >= 1 * Storage::bucketSize);
        REQUIRE(result % Storage::bucketSize == 0);
    }
}

TEST_CASE("Static DoubleHashing extents use only prime bucket rounding", "[extent]")
{
    using Size = std::size_t;
    using Probing = aclco::DoubleHashing<aclco::murmurhash3_32<std::uint32_t>>;

    constexpr auto out20 = aclco::MakeValidExtent<Probing, aclco::Storage<5>>(aclco::Extent<Size, 20>{});
    static_assert(std::is_same_v<decltype(out20), const aclco::Extent<Size, 25>>);
    REQUIRE(static_cast<Size>(out20) == 25);

    constexpr auto out8 = aclco::MakeValidExtent<Probing, aclco::Storage<1>>(aclco::Extent<Size, 8>{});
    static_assert(std::is_same_v<decltype(out8), const aclco::Extent<Size, 11>>);
    REQUIRE(static_cast<Size>(out8) == 11);

    constexpr auto out21 = aclco::MakeValidExtent<Probing, aclco::Storage<3>>(aclco::Extent<Size, 21>{});
    static_assert(std::is_same_v<decltype(out21), const aclco::Extent<Size, 21>>);
    REQUIRE(static_cast<Size>(out21) == 21);

    auto dynamic20 = aclco::MakeValidExtent<Probing, aclco::Storage<5>>(aclco::Extent<Size>{20});
    REQUIRE(static_cast<Size>(dynamic20) == 25);

    using Linear = aclco::LinearProbing<aclco::murmurhash3_32<std::uint32_t>>;
    constexpr auto linear20 = aclco::MakeValidExtent<Linear, aclco::Storage<5>>(aclco::Extent<Size, 20>{});
    static_assert(std::is_same_v<decltype(linear20), const aclco::Extent<Size, 20>>);
    REQUIRE(static_cast<Size>(linear20) == 20);
}
