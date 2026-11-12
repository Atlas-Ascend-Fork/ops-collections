/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "detail/hash_functions/utils.h"
#include "macros.h"

namespace aclco::detail {

template <typename Key>
struct MurmurHash3Fmix32 {
    using ArgumentType = Key;
    using ResultType   = std::uint32_t;

    COLLECTION_HOST_DEVICE constexpr MurmurHash3Fmix32(std::uint32_t seed = 0) : seed_{seed} {}
    COLLECTION_HOST_DEVICE constexpr bool operator==(MurmurHash3Fmix32 const& other) const noexcept
    {
        return seed_ == other.seed_;
    }
    COLLECTION_HOST_DEVICE constexpr ResultType operator()(Key const& key) const noexcept
    {
        return Fmix32(key, seed_);
    }
private:
    std::uint32_t seed_;
};

template <typename Key>
struct MurmurHash3Fmix64 {
    using ArgumentType = Key;
    using ResultType   = std::uint64_t;

    COLLECTION_HOST_DEVICE constexpr MurmurHash3Fmix64(std::uint64_t seed = 0) : seed_(seed) {}
    COLLECTION_HOST_DEVICE constexpr bool operator==(MurmurHash3Fmix64 const& other) const noexcept
    {
        return seed_ == other.seed_;
    }
    COLLECTION_HOST_DEVICE constexpr ResultType operator()(Key const& key) const noexcept
    {
        return Fmix64(key, seed_);
    }
private:
    std::uint64_t seed_;
};

template <typename Key>
struct MurmurHash3_32 {
    using ArgumentType = Key;
    using ResultType   = std::uint32_t;
    COLLECTION_HOST_DEVICE constexpr MurmurHash3_32(std::uint32_t seed = 0) : seed_{seed} {}
    COLLECTION_HOST_DEVICE constexpr bool operator==(MurmurHash3_32 const& other) const noexcept
    {
        return seed_ == other.seed_;
    }
    COLLECTION_HOST_DEVICE constexpr ResultType operator()(Key const& key) const noexcept
    {
        return ComputeHash(reinterpret_cast<std::byte const*>(&key), sizeof(key));
    }

    template <typename Extent>
    COLLECTION_HOST_DEVICE constexpr ResultType ComputeHash(std::byte const* bytes,
                                                             Extent size) const noexcept
    {
        auto const nblocks = size / 4;
        std::uint32_t h1 = seed_;
        constexpr std::uint32_t c1 = 0xcc9e2d51;
        constexpr std::uint32_t c2 = 0x1b873593;

        for (size_t i = 0; size >= 4 && i < nblocks; ++i) {
            std::uint32_t k1 = LoadChunk<std::uint32_t, std::byte, Extent>(bytes, i, size);
            k1 *= c1;
            k1 = Rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = Rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }
        std::uint32_t k1 = 0;
        switch (size & 3) {
            case 3:
                k1 ^= (std::uint32_t)(bytes[nblocks * 4 + 2]) << 16;
            case 2:
                k1 ^= (std::uint32_t)(bytes[nblocks * 4 + 1]) << 8;
            case 1:
                k1 ^= (std::uint32_t)(bytes[nblocks * 4 + 0]);
                k1 *= c1;
                k1 = Rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
            case 0: break;
        };
        h1 ^= size;
        h1 = Fmix32(h1);
        return h1;
    }
private:
    std::uint32_t seed_;
};
} // namespace aclco::detail