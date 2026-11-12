/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * XXHash_64 is adapted from NVIDIA cuCollections and xxHash.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * xxHash - Extremely Fast Hash algorithm
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "detail/hash_functions/utils.h"
#include "extent.h"
#include "macros.h"

namespace aclco::detail {
template <typename Key>
struct XXHash_32 {
private:
    static constexpr std::uint32_t prime1 = 0x9e3779b1u;
    static constexpr std::uint32_t prime2 = 0x85ebca77u;
    static constexpr std::uint32_t prime3 = 0xc2b2ae3du;
    static constexpr std::uint32_t prime4 = 0x27d4eb2fu;
    static constexpr std::uint32_t prime5 = 0x165667b1u;

public:
    using ArgumentType = Key;         ///< The type of the values taken as argument
    using ResultType = std::uint32_t; ///< The type of the hash values produced

    /**
     * @brief Constructs a XXH32 hash function with the given `seed`.
     *
     * @param seed A custom number to randomize the resulting hash value
     */
    COLLECTION_HOST_DEVICE constexpr XXHash_32(std::uint32_t seed = 0) : seed_{seed} {}

    COLLECTION_HOST_DEVICE constexpr bool operator==(XXHash_32 const& other) const noexcept
    {
        return seed_ == other.seed_;
    }

    /**
     * @brief Returns a hash value for its argument, as a value of type `ResultType`.
     *
     * @param key The input argument to hash
     * @return The resulting hash value for `key`
     */
    constexpr ResultType COLLECTION_HOST_DEVICE operator()(Key const& key) const noexcept
    {
        if constexpr (sizeof(Key) <= 16) {
            Key const key_copy = key;
            return ComputeHash(reinterpret_cast<std::byte const*>(&key_copy),
                               aclco::Extent<std::size_t, sizeof(Key)>{});
        } else {
            return ComputeHash(reinterpret_cast<std::byte const*>(&key), aclco::Extent<std::size_t, sizeof(Key)>{});
        }
    }

    /**
     * @brief Returns a hash value for its argument, as a value of type `ResultType`.
     *
     * @tparam Extent The Extent type
     *
     * @param bytes The input argument to hash
     * @param size The Extent of the data in bytes
     * @return The resulting hash value
     */
    template <typename Extent>
    constexpr ResultType COLLECTION_HOST_DEVICE ComputeHash(std::byte const* bytes, Extent size) const noexcept
    {
        size_t offset = 0;
        std::uint32_t h32{};

        // data can be processed in 16-byte chunks
        if (size >= 16) {
            auto const limit = size - 16;
            std::uint32_t v1 = seed_ + prime1 + prime2;
            std::uint32_t v2 = seed_ + prime2;
            std::uint32_t v3 = seed_;
            std::uint32_t v4 = seed_ - prime1;

            do {
                // pipeline 4*4byte computations
                auto const pipeline_offset = offset / 4;
                v1 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 0, size) * prime2;
                v1 = Rotl32(v1, 13);
                v1 *= prime1;
                v2 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 1, size) * prime2;
                v2 = Rotl32(v2, 13);
                v2 *= prime1;
                v3 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 2, size) * prime2;
                v3 = Rotl32(v3, 13);
                v3 *= prime1;
                v4 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 3, size) * prime2;
                v4 = Rotl32(v4, 13);
                v4 *= prime1;
                offset += 16;
            } while (offset <= limit);

            h32 = Rotl32(v1, 1) + Rotl32(v2, 7) + Rotl32(v3, 12) + Rotl32(v4, 18);
        } else {
            h32 = seed_ + prime5;
        }

        h32 += static_cast<std::uint32_t>(size);

        // remaining data can be processed in 4-byte chunks
        if ((size % 16) >= 4) {
            for (; offset <= size - 4; offset += 4) {
                h32 += LoadChunk<std::uint32_t>(bytes, offset / 4, size) * prime3;
                h32 = Rotl32(h32, 17) * prime4;
            }
        }

        // the following loop is only needed if the size of the key is not a multiple of the block size
        if (size % 4) {
            while (offset < size) {
                h32 += (static_cast<std::uint32_t>(bytes[offset]) & 255) * prime5;
                h32 = Rotl32(h32, 11) * prime1;
                ++offset;
            }
        }

        return Finalize(h32);
    }

private:
    // avalanche helper
    constexpr COLLECTION_HOST_DEVICE std::uint32_t Finalize(std::uint32_t h) const noexcept
    {
        h ^= h >> 15;
        h *= prime2;
        h ^= h >> 13;
        h *= prime3;
        h ^= h >> 16;
        return h;
    }

    std::uint32_t seed_;
};

/**
 * @brief XXH64 hash functor used by BloomFilter.
 *
 * The implementation follows the canonical XXH64 algorithm and intentionally
 * hashes the object representation of Key. Therefore floating-point keys keep
 * their bitwise identity (for example +0.0f and -0.0f hash differently).
 */
template <typename Key>
struct XXHash_64 {
private:
    static constexpr std::uint64_t prime1 = 11400714785074694791ull;
    static constexpr std::uint64_t prime2 = 14029467366897019727ull;
    static constexpr std::uint64_t prime3 = 1609587929392839161ull;
    static constexpr std::uint64_t prime4 = 9650029242287828579ull;
    static constexpr std::uint64_t prime5 = 2870177450012600261ull;

public:
    using ArgumentType = Key;
    using ResultType = std::uint64_t;

    COLLECTION_HOST_DEVICE constexpr XXHash_64(std::uint64_t seed = 0) noexcept : seed_{seed} {}

    COLLECTION_HOST_DEVICE constexpr ResultType operator()(Key const& key) const noexcept
    {
        if constexpr (std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::uint32_t>) {
            // CANN targets and supported hosts are little-endian. Specialize the
            // canonical four-byte XXH64 tail to keep routed Add hashing entirely in
            // registers instead of materializing and byte-addressing a key copy.
            std::uint32_t const keyBits = static_cast<std::uint32_t>(key);
            std::uint64_t hash = seed_ + prime5 + sizeof(Key);
            hash ^= static_cast<std::uint64_t>(keyBits) * prime1;
            hash = Rotl64(hash, 23) * prime2 + prime3;
            return Finalize(hash);
        } else if constexpr (std::is_same_v<Key, std::int64_t> || std::is_same_v<Key, std::uint64_t>) {
            // Canonical eight-byte XXH64 tail, likewise expressed directly on the
            // integral value to remove generic LoadChunk address arithmetic.
            std::uint64_t hash = seed_ + prime5 + sizeof(Key);
            std::uint64_t lane = static_cast<std::uint64_t>(key) * prime2;
            lane = Rotl64(lane, 31) * prime1;
            hash ^= lane;
            hash = Rotl64(hash, 27) * prime1 + prime4;
            return Finalize(hash);
        } else if constexpr (sizeof(Key) <= 16) {
            Key const keyCopy = key;
            return ComputeHash(reinterpret_cast<std::byte const*>(&keyCopy), aclco::Extent<std::size_t, sizeof(Key)>{});
        } else {
            return ComputeHash(reinterpret_cast<std::byte const*>(&key), aclco::Extent<std::size_t, sizeof(Key)>{});
        }
    }

    template <typename Extent>
    COLLECTION_HOST_DEVICE constexpr ResultType ComputeHash(std::byte const* bytes, Extent size) const noexcept
    {
        std::size_t offset = 0;
        std::uint64_t h64 = 0;

        if (size >= 32) {
            auto const limit = size - 32;
            std::uint64_t v1 = seed_ + prime1 + prime2;
            std::uint64_t v2 = seed_ + prime2;
            std::uint64_t v3 = seed_;
            std::uint64_t v4 = seed_ - prime1;

            do {
                auto const pipelineOffset = offset / 8;
                v1 += LoadChunk<std::uint64_t>(bytes, pipelineOffset + 0, size) * prime2;
                v1 = Rotl64(v1, 31);
                v1 *= prime1;
                v2 += LoadChunk<std::uint64_t>(bytes, pipelineOffset + 1, size) * prime2;
                v2 = Rotl64(v2, 31);
                v2 *= prime1;
                v3 += LoadChunk<std::uint64_t>(bytes, pipelineOffset + 2, size) * prime2;
                v3 = Rotl64(v3, 31);
                v3 *= prime1;
                v4 += LoadChunk<std::uint64_t>(bytes, pipelineOffset + 3, size) * prime2;
                v4 = Rotl64(v4, 31);
                v4 *= prime1;
                offset += 32;
            } while (offset <= limit);

            h64 = Rotl64(v1, 1) + Rotl64(v2, 7) + Rotl64(v3, 12) + Rotl64(v4, 18);
            h64 = MergeRound(h64, v1);
            h64 = MergeRound(h64, v2);
            h64 = MergeRound(h64, v3);
            h64 = MergeRound(h64, v4);
        } else {
            h64 = seed_ + prime5;
        }

        h64 += static_cast<std::uint64_t>(size);

        if ((size % 32) >= 8) {
            for (; offset <= size - 8; offset += 8) {
                std::uint64_t k1 = LoadChunk<std::uint64_t>(bytes, offset / 8, size) * prime2;
                k1 = Rotl64(k1, 31) * prime1;
                h64 ^= k1;
                h64 = Rotl64(h64, 27) * prime1 + prime4;
            }
        }

        if ((size % 8) >= 4) {
            for (; offset <= size - 4; offset += 4) {
                h64 ^= (static_cast<std::uint64_t>(LoadChunk<std::uint32_t>(bytes, offset / 4, size)) & 0xffffffffull) *
                       prime1;
                h64 = Rotl64(h64, 23) * prime2 + prime3;
            }
        }

        while (offset < static_cast<std::size_t>(size)) {
            h64 ^= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(bytes[offset])) & 0xffull) * prime5;
            h64 = Rotl64(h64, 11) * prime1;
            ++offset;
        }
        return Finalize(h64);
    }

    COLLECTION_HOST_DEVICE constexpr std::uint64_t Seed() const noexcept { return seed_; }

    COLLECTION_HOST_DEVICE constexpr bool operator==(XXHash_64 const& other) const noexcept
    {
        return seed_ == other.seed_;
    }

    COLLECTION_HOST_DEVICE constexpr bool operator!=(XXHash_64 const& other) const noexcept
    {
        return !(*this == other);
    }

private:
    COLLECTION_HOST_DEVICE static constexpr std::uint64_t MergeRound(std::uint64_t hash, std::uint64_t value) noexcept
    {
        value *= prime2;
        value = Rotl64(value, 31);
        value *= prime1;
        hash ^= value;
        return hash * prime1 + prime4;
    }

    COLLECTION_HOST_DEVICE static constexpr std::uint64_t Finalize(std::uint64_t hash) noexcept
    {
        hash ^= hash >> 33;
        hash *= prime2;
        hash ^= hash >> 29;
        hash *= prime3;
        hash ^= hash >> 32;
        return hash;
    }

    std::uint64_t seed_;
};
} // namespace aclco::detail
