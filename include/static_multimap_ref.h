/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "hash_functions.h"
#include "kernel_operator.h"
#include "macros.h"
#include "pair.h"

namespace aclco {

/**
 * @brief Device-side probing view: a contiguous window first, then a per-key stride.
 * Each successful insertion owns a different slot, including identical pairs.
 * Dense storage uses one tagged-index CAS; inserting threads only inspect the
 * index table, never another thread's unpublished dense payload. Direct slots
 * are retained for capacities exceeding the 32-bit dense-index range.
 */
template <class Key, class Value>
struct StaticMultimapRef {
    static constexpr uint32_t PROBE_WINDOW = 32; ///< Contiguous slots visited before stride probing takes over.
    __gm__ Pair<Key, Value>* slots; ///< Dense payloads or direct pairs, depending on indices.
    std::size_t capacity;
    Key emptyKey;     ///< Reserved key that identifies an empty direct slot.
    Value emptyValue; ///< Reserved value returned for missing queries.
    // A slot contains a 32-bit key/tag and a 32-bit dense index plus one; zero is empty.
    // Large capacities retain direct slots instead of truncating an index.
    __gm__ uint64_t* indices = nullptr;
    uint32_t rangeMask; ///< Mask of the smallest power-of-two domain covering capacity.

    COLLECTION_SIMT_DEVICE StaticMultimapRef(__gm__ Pair<Key, Value>* table, std::size_t n, Key reservedKey,
                                             Value reservedValue, __gm__ uint64_t* indexTable)
        : slots(table), capacity(n), emptyKey(reservedKey), emptyValue(reservedValue), indices(indexTable)
    {
        rangeMask = uint32_t(capacity - 1);
        rangeMask |= rangeMask >> 1;
        rangeMask |= rangeMask >> 2;
        rangeMask |= rangeMask >> 4;
        rangeMask |= rangeMask >> 8;
        rangeMask |= rangeMask >> 16;
    }

    /**
     * @brief Read the pair at a hash slot.
     * @return Stored pair, or the empty sentinel pair.
     */
    COLLECTION_SIMT_DEVICE Pair<Key, Value> PairAt(std::size_t slot) const
    {
        if (!indices) {
            return slots[slot];
        }
        auto index = uint32_t(indices[slot]);
        if (index) {
            return slots[index - 1];
        }
        return Pair<Key, Value>{emptyKey, emptyValue};
    }

    /**
     * @brief Publish a dense index through a bounded CAS probe.
     * @return Whether an empty index slot was claimed.
     * @note Requires a nonnull index table. Payload must be complete before queries execute.
     */
    COLLECTION_SIMT_DEVICE bool InsertIndex(Key key, std::size_t index) const
    {
        uint32_t start = uint32_t(Start(key));
        uint32_t limit = uint32_t(capacity);
        auto entry = (uint64_t(Tag(key)) << 32) | uint32_t(index + 1);
        // Preload the first slot to avoid a failed atomic write inside the probe loop.
        if (indices[start] == 0 && AscendC::Simt::AtomicCas(indices + start, uint64_t(0), entry) == 0) {
            return true;
        }
        uint32_t slot = start;
        uint32_t i = 0;
        uint32_t step = 0;
        while (i < limit) {
            if (indices[slot] == 0 && AscendC::Simt::AtomicCas(indices + slot, uint64_t(0), entry) == 0) {
                return true;
            }
            ++i;
            if (i < PROBE_WINDOW) {
                if (++slot == limit) {
                    slot = 0;
                }
            } else {
                if (i == PROBE_WINDOW) {
                    if (limit <= PROBE_WINDOW) {
                        break; // The window already covers the whole table.
                    }
                    step = ProbeStep(key, limit);
                }
                // Window landings were already visited during the window phase;
                // advance past them without spending visit budget so the stride
                // cycle still reaches every slot outside the window.
                uint64_t offset = 0;
                do {
                    slot += step;
                    if (slot >= limit) {
                        slot -= limit;
                    }
                    offset = (slot >= start) ? uint64_t(slot) - start : uint64_t(slot) + limit - start;
                } while (offset < PROBE_WINDOW);
            }
        }
        return false;
    }

    /**
     * @brief Compute the key tag stored with a dense index.
     * @return Full I32 key bits or I64 hash summary.
     */
    COLLECTION_SIMT_DEVICE uint32_t Tag(Key key) const
    {
        if constexpr (sizeof(Key) == 4) {
            return uint32_t(key);
        } else {
            return InCapacityRange(key) ? uint32_t(key) : uint32_t(murmurhash3_fmix64<Key>{}(key));
        }
    }
    COLLECTION_SIMT_DEVICE bool Matches(uint64_t entry, Key key) const
    {
        if (uint32_t(entry >> 32) != Tag(key)) {
            return false;
        }
        if constexpr (sizeof(Key) == 4) {
            return true;
        } else {
            return slots[uint32_t(entry) - 1].first == key;
        }
    }

    /**
     * @brief Compute the first probe slot.
     * @return Slot in the exact capacity range.
     */
    COLLECTION_SIMT_DEVICE std::size_t Start(Key key) const
    {
        if (indices) {
            if (InCapacityRange(key)) {
                return PermuteInCapacity(uint32_t(key));
            }
            uint32_t hash;
            // Multiplicative hashing is a permutation of the 32-bit key space.
            if constexpr (sizeof(Key) == 4) {
                hash = uint32_t(key) * 0x9e3779b9u;
            } else {
                hash = uint32_t(murmurhash3_fmix64<Key>{}(key));
            }
            // Multiplicative range reduction avoids integer division and maps
            // every hash into the exact capacity without padding the table.
            return (uint64_t(hash) * uint32_t(capacity)) >> 32;
        }
        if constexpr (sizeof(Key) == 4) {
            return murmurhash3_fmix32<Key>{}(key) % capacity;
        } else {
            return murmurhash3_fmix64<Key>{}(key) % capacity;
        }
    }

    COLLECTION_SIMT_DEVICE bool InCapacityRange(Key key) const
    {
        return indices && key >= Key{0} && uint64_t(key) < capacity;
    }

    /**
     * @brief Diffuse a compact integer key through a bijection of the capacity range.
     * @return A unique slot smaller than capacity.
     */
    COLLECTION_SIMT_DEVICE uint32_t PermuteInCapacity(uint32_t key) const
    {
        auto slot = key;
        do {
            slot = (slot * 33u + 0x85ebca6bu) & rangeMask;
        } while (slot >= capacity);
        return slot;
    }

    /**
     * @brief Compute an odd probe stride coprime with the capacity.
     * @return A stride in [1, capacity).
     * @note Only reached after the contiguous window overflows, so queries and
     * inserts of low-duplicate keys never pay for it.
     */
    COLLECTION_SIMT_DEVICE uint32_t ProbeStep(Key key, uint32_t limit) const
    {
        uint32_t hash;
        if constexpr (sizeof(Key) == 4) {
            hash = murmurhash3_fmix32<Key>{}(key);
        } else {
            hash = uint32_t(murmurhash3_fmix64<Key>{}(key));
        }
        // Multiplicative reduction, no integer division on the device hot path.
        uint32_t step = uint32_t((uint64_t(hash) * limit) >> 32);
        if (step == 0) {
            step = 1;
        }
        while (GcdU32(step, limit) != 1) {
            if (++step == limit) {
                step = 1;
            }
        }
        return step;
    }

    // Binary GCD: shifts and subtractions only, no division.
    COLLECTION_SIMT_DEVICE static uint32_t GcdU32(uint32_t x, uint32_t y)
    {
        if (x == 0) {
            return y;
        }
        if (y == 0) {
            return x;
        }
        uint32_t shift = 0;
        while (((x | y) & 1u) == 0) {
            x >>= 1;
            y >>= 1;
            ++shift;
        }
        while ((x & 1u) == 0) {
            x >>= 1;
        }
        do {
            while ((y & 1u) == 0) {
                y >>= 1;
            }
            if (x > y) {
                uint32_t t = x;
                x = y;
                y = t;
            }
            y -= x;
        } while (y != 0);
        return x << shift;
    }

    /**
     * @brief Insert one pair into direct slots.
     * @return Whether an empty slot was claimed.
     * @note Requires indices == nullptr and no concurrent queries.
     */
    COLLECTION_SIMT_DEVICE bool Insert(Key key, Value value) const
    {
        uint64_t expectedBits = 0;
        uint64_t desiredBits = 0;
        if constexpr (sizeof(Pair<Key, Value>) == sizeof(uint64_t)) {
            if constexpr (std::is_same_v<Key, int32_t> && std::is_same_v<Value, int32_t>) {
                // The direct I32 pair occupies adjacent low/high 32-bit words.
                using PairType = Pair<Key, Value>;
                static_assert(offsetof(PairType, second) == sizeof(Key));
                expectedBits = uint64_t(uint32_t(emptyKey)) | (uint64_t(uint32_t(emptyValue)) << 32);
                desiredBits = uint64_t(uint32_t(key)) | (uint64_t(uint32_t(value)) << 32);
            } else {
                Pair<Key, Value> expected{emptyKey, emptyValue};
                Pair<Key, Value> desired{key, value};
                // Copy object bytes without imposing uint64_t alignment or aliasing.
                const auto* expectedBytes = reinterpret_cast<const unsigned char*>(&expected);
                const auto* desiredBytes = reinterpret_cast<const unsigned char*>(&desired);
                auto* expectedBitsBytes = reinterpret_cast<unsigned char*>(&expectedBits);
                auto* desiredBitsBytes = reinterpret_cast<unsigned char*>(&desiredBits);
                for (std::size_t i = 0; i < sizeof(uint64_t); ++i) {
                    expectedBitsBytes[i] = expectedBytes[i];
                    desiredBitsBytes[i] = desiredBytes[i];
                }
            }
        }
        auto slot = Start(key);
        for (std::size_t step = 0; step < capacity; ++step) {
            // Slots only transition from empty to occupied during insertion.
            // A stale empty read can only cause an extra CAS attempt.
            if (slots[slot].first == emptyKey) {
                if constexpr (sizeof(Pair<Key, Value>) == sizeof(uint64_t)) {
                    if (AscendC::Simt::AtomicCas(reinterpret_cast<__gm__ uint64_t*>(slots + slot), expectedBits,
                                                 desiredBits) == expectedBits) {
                        return true;
                    }
                } else if (AscendC::Simt::AtomicCas(&slots[slot].first, emptyKey, key) == emptyKey) {
                    slots[slot].second = value;
                    return true;
                }
            }
            if (++slot == capacity) {
                slot = 0;
            }
        }
        return false;
    }

    enum class QueryMode { Contains, Find, Count };
    template <QueryMode Mode>
    using QueryResult = std::conditional_t<Mode == QueryMode::Contains, bool,
                                           std::conditional_t<Mode == QueryMode::Find, Value, std::uint64_t>>;

    template <bool Indexed, QueryMode Mode>
    COLLECTION_SIMT_DEVICE QueryResult<Mode> QueryImpl(Key key, std::size_t* first = nullptr) const
    {
        QueryResult<Mode> result{};
        bool found = false;
        if constexpr (Mode == QueryMode::Find) {
            result = emptyValue;
        }
        if (key == emptyKey) {
            return result;
        }
        if constexpr (Indexed) {
            uint32_t start = uint32_t(Start(key));
            uint32_t limit = uint32_t(capacity);
            uint32_t slot = start;
            uint32_t i = 0;
            uint32_t step = 0;
            while (i < limit) {
                auto entry = indices[slot];
                if (entry == 0) {
                    break;
                }
                if (Matches(entry, key)) {
                    if constexpr (Mode == QueryMode::Contains) {
                        return true;
                    } else if constexpr (Mode == QueryMode::Find) {
                        Value value = slots[uint32_t(entry) - 1].second;
                        if (!found || value < result) {
                            result = value;
                            found = true;
                        }
                    } else {
                        if (first && result == 0) {
                            *first = slot;
                        }
                        ++result;
                    }
                }
                ++i;
                if (i < PROBE_WINDOW) {
                    if (++slot == limit) {
                        slot = 0;
                    }
                } else {
                    if (i == PROBE_WINDOW) {
                        if (limit <= PROBE_WINDOW) {
                            break; // The window already covers the whole table.
                        }
                        step = ProbeStep(key, limit);
                    }
                    // Window landings were already visited during the window
                    // phase; advance past them without spending visit budget so
                    // the stride cycle still reaches every slot outside it.
                    uint64_t offset = 0;
                    do {
                        slot += step;
                        if (slot >= limit) {
                            slot -= limit;
                        }
                        offset = (slot >= start) ? uint64_t(slot) - start : uint64_t(slot) + limit - start;
                    } while (offset < PROBE_WINDOW);
                }
            }
        } else {
            std::size_t slot = std::size_t(Start(key));
            for (std::size_t step = 0; step < capacity; ++step) {
                if (slots[slot].first == emptyKey) {
                    break;
                }
                if (slots[slot].first == key) {
                    if constexpr (Mode == QueryMode::Contains) {
                        return true;
                    } else if constexpr (Mode == QueryMode::Find) {
                        Value value = slots[slot].second;
                        if (!found || value < result) {
                            result = value;
                            found = true;
                        }
                    } else {
                        if (first && result == 0) {
                            *first = slot;
                        }
                        ++result;
                    }
                }
                if (++slot == capacity) {
                    slot = 0;
                }
            }
        }
        return result;
    }

    template <bool Indexed>
    COLLECTION_SIMT_DEVICE bool ContainsImpl(Key key) const
    {
        return QueryImpl<Indexed, QueryMode::Contains>(key);
    }

    /**
     * @brief Scan every matching slot and return the minimum Value.
     * @return Minimum matching value, or emptyValue.
     */
    template <bool Indexed>
    COLLECTION_SIMT_DEVICE Value FindImpl(Key key) const
    {
        return QueryImpl<Indexed, QueryMode::Find>(key);
    }

    /**
     * @brief Count all matching slots using the selected representation.
     * @return Number of matching pairs.
     */
    template <bool Indexed>
    COLLECTION_SIMT_DEVICE std::uint64_t CountImpl(Key key, std::size_t* first) const
    {
        return QueryImpl<Indexed, QueryMode::Count>(key, first);
    }
    template <bool Indexed, class Sink>
    COLLECTION_SIMT_DEVICE uint64_t CountCachedImpl(Key key, std::size_t* first, Sink sink) const
    {
        uint64_t count = 0;
        if (key == emptyKey) return count;
        if constexpr (Indexed) {
            uint32_t start = uint32_t(Start(key));
            uint32_t limit = uint32_t(capacity);
            uint32_t slot = start;
            uint32_t i = 0;
            uint32_t step = 0;
            while (i < limit) {
                auto entry = indices[slot];
                if (entry == 0) break;
                if (Matches(entry, key)) {
                    if (first && count == 0) *first = slot;
                    if (sink.Accept(count)) {
                        sink.Store(count, slots[uint32_t(entry) - 1].second);
                    }
                    ++count;
                }
                ++i;
                if (i < PROBE_WINDOW) {
                    if (++slot == limit) {
                        slot = 0;
                    }
                } else {
                    if (i == PROBE_WINDOW) {
                        if (limit <= PROBE_WINDOW) {
                            break; // The window already covers the whole table.
                        }
                        step = ProbeStep(key, limit);
                    }
                    // Window landings were already visited during the window
                    // phase; advance past them without spending visit budget so
                    // the stride cycle still reaches every slot outside it.
                    uint64_t offset = 0;
                    do {
                        slot += step;
                        if (slot >= limit) {
                            slot -= limit;
                        }
                        offset = (slot >= start) ? uint64_t(slot) - start : uint64_t(slot) + limit - start;
                    } while (offset < PROBE_WINDOW);
                }
            }
        } else {
            std::size_t slot = std::size_t(Start(key));
            for (std::size_t step = 0; step < capacity; ++step) {
                if (slots[slot].first == emptyKey) break;
                if (slots[slot].first == key) {
                    if (first && count == 0) *first = slot;
                    if (sink.Accept(count)) {
                        sink.Store(count, slots[slot].second);
                    }
                    ++count;
                }
                if (++slot == capacity) slot = 0;
            }
        }
        return count;
    }
    template <class Sink>
    COLLECTION_SIMT_DEVICE uint64_t CountCached(Key key, std::size_t* first, Sink sink) const
    {
        return indices ? CountCachedImpl<true>(key, first, sink) : CountCachedImpl<false>(key, first, sink);
    }
    COLLECTION_SIMT_DEVICE bool Contains(Key key) const
    {
        return indices ? ContainsImpl<true>(key) : ContainsImpl<false>(key);
    }
    /**
     * @brief Find the smallest matching value.
     * @return Minimum matching Value, or emptyValue when absent.
     */
    COLLECTION_SIMT_DEVICE Value Find(Key key) const { return indices ? FindImpl<true>(key) : FindImpl<false>(key); }
    /**
     * @brief Count pairs matching one key.
     * @return Number of matches; first is unchanged when zero.
     */
    COLLECTION_SIMT_DEVICE std::uint64_t Count(Key key, std::size_t* first = nullptr) const
    {
        return indices ? CountImpl<true>(key, first) : CountImpl<false>(key, first);
    }
};
} // namespace aclco
