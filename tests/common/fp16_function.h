/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <ostream>
#include <type_traits>

#include "include/utility/math_utils.h"
#include "include/pair.h"
#include "include/utility/traits.h"

namespace aclco {

COLLECTION_HOST_DEVICE float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>((h >> 15) & 0x1);
    uint32_t exponent = static_cast<uint32_t>((h >> 10) & 0x1F);
    uint32_t mantissa = static_cast<uint32_t>(h & 0x3FF);

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t f = sign << 31;
            return *reinterpret_cast<float*>(&f);
        } else {
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x400U;
        }
    } else if (exponent == 31) {
        uint32_t f = (sign << 31) | (0xFFU << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&f);
    }

    exponent += 127 - 15;
    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&f);
}

COLLECTION_HOST_DEVICE uint16_t FloatToFp16(float f)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t fExp = (bits >> 23) & 0xFF;
    uint32_t fMant = bits & 0x7FFFFFU;

    if (fExp == 0 && fMant == 0) {
        return static_cast<uint16_t>(sign << 15);
    }

    if (fExp == 255) {
        if (fMant == 0) {
            return static_cast<uint16_t>((sign << 15) | 0x7C00U);
        } else {
            return static_cast<uint16_t>((sign << 15) | 0x7C00U | (fMant >> 13));
        }
    }

    int32_t hExp = static_cast<int32_t>(fExp) - 127 + 15;

    if (hExp <= 0) {
        if (hExp < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        fMant |= 0x800000U;
        uint32_t shift = static_cast<uint32_t>(14 - hExp);
        uint32_t hMant = fMant >> shift;
        uint32_t roundBit = (fMant >> (shift - 1)) & 0x1;
        uint32_t sticky = (fMant & ((1U << (shift - 1)) - 1)) != 0 ? 1 : 0;
        if (roundBit && (sticky || (hMant & 0x1))) {
            hMant++;
        }
        return static_cast<uint16_t>((sign << 15) | hMant);
    }

    if (hExp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00U);
    }

    uint32_t hMant = fMant >> 13;
    uint32_t roundBit = (fMant >> 12) & 0x1;
    uint32_t sticky = (fMant & 0xFFFU) != 0 ? 1 : 0;
    if (roundBit && (sticky || (hMant & 0x1))) {
        hMant++;
        if (hMant > 0x3FFU) {
            hMant = 0;
            hExp++;
            if (hExp >= 31) {
                return static_cast<uint16_t>((sign << 15) | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(hExp) << 10) | hMant);
}

struct fp16 {
    uint16_t val;

    fp16() : val(0x0u) {}

    constexpr fp16(uint16_t uiVal) : val(uiVal) {}

    COLLECTION_HOST_DEVICE bool operator==(fp16 const& rhs) const noexcept { return val == rhs.val; }
    COLLECTION_HOST_DEVICE bool operator!=(fp16 const& rhs) const noexcept { return val != rhs.val; }
    COLLECTION_HOST_DEVICE bool operator<(fp16 const& rhs) const noexcept { return val < rhs.val; }
    COLLECTION_HOST_DEVICE bool operator<=(fp16 const& rhs) const noexcept { return val <= rhs.val; }
    COLLECTION_HOST_DEVICE bool operator>(fp16 const& rhs) const noexcept { return val > rhs.val; }
    COLLECTION_HOST_DEVICE bool operator>=(fp16 const& rhs) const noexcept { return val >= rhs.val; }

    COLLECTION_HOST_DEVICE fp16& operator++() noexcept { ++val; return *this; }
    COLLECTION_HOST_DEVICE fp16 operator++(int) noexcept { fp16 tmp = *this; ++val; return tmp; }
    COLLECTION_HOST_DEVICE fp16& operator--() noexcept { --val; return *this; }
    COLLECTION_HOST_DEVICE fp16 operator--(int) noexcept { fp16 tmp = *this; --val; return tmp; }

    COLLECTION_HOST_DEVICE fp16& operator=(uint16_t rhs) noexcept { val = rhs; return *this; }

    explicit COLLECTION_HOST_DEVICE operator float() const { return Fp16ToFloat(val); }
    explicit COLLECTION_HOST_DEVICE operator double() const { return static_cast<double>(Fp16ToFloat(val)); }
    explicit COLLECTION_HOST_DEVICE operator int8_t() const { return static_cast<int8_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator uint8_t() const { return static_cast<uint8_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator int16_t() const { return static_cast<int16_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator uint16_t() const { return val; }
    explicit COLLECTION_HOST_DEVICE operator int32_t() const { return static_cast<int32_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator uint32_t() const { return static_cast<uint32_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator int64_t() const { return static_cast<int64_t>(val); }
    explicit COLLECTION_HOST_DEVICE operator uint64_t() const { return static_cast<uint64_t>(val); }
    COLLECTION_HOST_DEVICE operator bool() const { return val != 0; }
};

typedef fp16 fp16_t;

COLLECTION_HOST_DEVICE  uint16_t operator%(fp16_t lhs, int rhs) noexcept
{
    return lhs.val % static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  uint16_t operator%(fp16_t lhs, uint16_t rhs) noexcept
{
    return lhs.val % rhs;
}

COLLECTION_HOST_DEVICE  uint16_t operator%(fp16_t lhs, fp16_t rhs) noexcept
{
    return lhs.val % rhs.val;
}

COLLECTION_HOST_DEVICE  bool operator==(fp16_t lhs, int rhs) noexcept
{
    return lhs.val == static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  bool operator!=(fp16_t lhs, int rhs) noexcept
{
    return lhs.val != static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  bool operator<(fp16_t lhs, int rhs) noexcept
{
    return lhs.val < static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  bool operator<=(fp16_t lhs, int rhs) noexcept
{
    return lhs.val <= static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  bool operator>(fp16_t lhs, int rhs) noexcept
{
    return lhs.val > static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  bool operator>=(fp16_t lhs, int rhs) noexcept
{
    return lhs.val >= static_cast<uint16_t>(rhs);
}

COLLECTION_HOST_DEVICE  float operator*(fp16_t lhs, int rhs) noexcept
{
    return Fp16ToFloat(lhs.val) * static_cast<float>(rhs);
}

COLLECTION_HOST_DEVICE  float operator/(fp16_t lhs, int rhs) noexcept
{
    return Fp16ToFloat(lhs.val) / static_cast<float>(rhs);
}

COLLECTION_HOST_DEVICE  float operator+(fp16_t lhs, int rhs) noexcept
{
    return Fp16ToFloat(lhs.val) + static_cast<float>(rhs);
}

COLLECTION_HOST_DEVICE  float operator-(fp16_t lhs, int rhs) noexcept
{
    return Fp16ToFloat(lhs.val) - static_cast<float>(rhs);
}

COLLECTION_HOST_DEVICE  float operator*(fp16_t lhs, float rhs) noexcept
{
    return Fp16ToFloat(lhs.val) * rhs;
}

COLLECTION_HOST_DEVICE  float operator/(fp16_t lhs, float rhs) noexcept
{
    return Fp16ToFloat(lhs.val) / rhs;
}

COLLECTION_HOST_DEVICE  float operator+(fp16_t lhs, float rhs) noexcept
{
    return Fp16ToFloat(lhs.val) + rhs;
}

COLLECTION_HOST_DEVICE  float operator-(fp16_t lhs, float rhs) noexcept
{
    return Fp16ToFloat(lhs.val) - rhs;
}

COLLECTION_HOST_DEVICE void SetFp16Value(fp16_t& out, uint16_t val)
{
    out.val = val;
}

COLLECTION_HOST_DEVICE void SetFp16FromFloat(fp16_t& out, float f)
{
    out.val = FloatToFp16(f);
}

template <typename T>
COLLECTION_HOST_DEVICE void StaticCast(T& out, uint64_t val)
{
    if constexpr (std::is_same_v<T, fp16_t>) {
        SetFp16Value(out, static_cast<uint16_t>(val));
    } else {
        out = static_cast<T>(val);
    }
}

 std::ostream& operator<<(std::ostream& os, fp16_t const& v)
{
    return os << Fp16ToFloat(v.val);
}

template <>
struct EqualTo<fp16_t>
{
    COLLECTION_HOST_DEVICE bool operator()(fp16_t const& lhs, fp16_t const& rhs) const noexcept
    {
        return lhs.val == rhs.val;
    }

    COLLECTION_HOST_DEVICE bool operator()(uint16_t const& lhs, uint16_t const& rhs) const noexcept
    {
        return lhs == rhs;
    }
};
}

namespace std {

template <>
struct hash<aclco::fp16_t>
{
    size_t operator()(aclco::fp16_t const& v) const noexcept
    {
        return hash<uint16_t>{}(v.val);
    }
};

}