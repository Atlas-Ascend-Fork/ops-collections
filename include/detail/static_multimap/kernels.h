/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once
#include "static_multimap_ref.h"
#include "detail/static_multimap/algorithms.h"

namespace aclco::detail::multimap {
constexpr std::size_t CHUNK = 256;
constexpr uint32_t THREADS = 1024;
constexpr uint32_t INSERT_THREADS = 1024;
// Above this batch size the warp-parallel single-pass per-chunk insert is
// cheaper than materializing every rank and scanning the input a second time.
constexpr std::size_t rankedInsertMaximumKeys = 1ULL << 20;

COLLECTION_SIMT_DEVICE std::size_t ThreadIndex()
{
    return std::size_t(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
}
COLLECTION_SIMT_DEVICE std::size_t ThreadStride()
{
    return std::size_t(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
}
struct SelectAll {
    template <class T>
    COLLECTION_SIMT_DEVICE bool operator()(T) const
    {
        return true;
    }
};

template <class Key, class Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ClearSimt(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices,
                                                               std::size_t n, Key emptyKey, Value emptyValue)
{
    for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
        if (indices) {
            indices[i] = 0;
        } else {
            table[i].first = emptyKey;
            table[i].second = emptyValue;
        }
    }
}

template <class Key, class Value, class Stencil, class Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ValidateSimt(__gm__ Pair<Key, Value>* input,
                                                                  __gm__ Stencil* stencil, std::size_t n, Key emptyKey,
                                                                  Value emptyValue, __gm__ uint64_t* counts,
                                                                  __gm__ uint32_t* invalid)
{
    constexpr uint32_t WARP = 32;
    auto lane = AscendC::Simt::GetThreadIdx() % WARP;
    if constexpr (!Conditional) {
        bool bad = false;
        for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
            auto pair = input[i];
            bad |= pair.first == emptyKey || pair.second == emptyValue;
        }
        bad = AscendC::Simt::WarpAnySync(bad);
        if (bad && lane == 0) {
            AscendC::Simt::AtomicCas(invalid, uint32_t(0), uint32_t(1));
        }
        return;
    }
    auto chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex() / WARP; c < chunks; c += ThreadStride() / WARP) {
        uint32_t count = 0;
        bool bad = false;
        auto end = (c + 1) * CHUNK;
        if (end > n) {
            end = n;
        }
        for (auto i = c * CHUNK + lane; i < end; i += WARP) {
            if (!Predicate{}(stencil[i])) {
                continue;
            }
            ++count;
            bad |= input[i].first == emptyKey || input[i].second == emptyValue;
        }
        count = AscendC::Simt::WarpReduceAddSync(count);
        bad = AscendC::Simt::WarpAnySync(bad);
        if (lane == 0) {
            counts[c] = count;
            if (bad) {
                AscendC::Simt::AtomicCas(invalid, uint32_t(0), uint32_t(1));
            }
        }
    }
}

// One warp scans a chunk with coalesced accesses. Recursive scans of sums and
// a separate offset kernel provide a global prefix without spin-waiting.
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ScanSimt(__gm__ uint64_t* values, __gm__ uint64_t* sums,
                                                              std::size_t n)
{
    constexpr uint32_t WARP = 32;
    auto lane = AscendC::Simt::GetThreadIdx() % WARP;
    auto chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex() / WARP; c < chunks; c += ThreadStride() / WARP) {
        uint64_t carry = 0;
        for (uint32_t part = 0; part < CHUNK; part += WARP) {
            auto i = c * CHUNK + part + lane;
            uint64_t value = i < n ? values[i] : 0;
            uint64_t sum = value;
            sum = InclusiveWarpSum(sum, lane);
            if (i < n) {
                values[i] = carry + sum - value;
            }
            carry += AscendC::Simt::WarpShflSync(sum, WARP - 1, WARP);
        }
        if (lane == 0) {
            sums[c] = carry;
        }
    }
}
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void OffsetSimt(__gm__ uint64_t* values, __gm__ uint64_t* sums,
                                                                std::size_t n, std::size_t chunks)
{
    if (chunks > 1) {
        for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
            values[i] += sums[i / CHUNK];
        }
    }
    if (ThreadIndex() == 0) {
        values[n] = chunks == 1 ? sums[0] : sums[chunks];
    }
}

template <class Key, class Value>
COLLECTION_SIMT_DEVICE bool InsertSelected(StaticMultimapRef<Key, Value> const& ref,
                                           __gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices,
                                           __gm__ Pair<Key, Value>* input, std::size_t inputIndex,
                                           std::size_t position)
{
    auto pair = input[inputIndex];
    if (indices) {
        table[position].first = pair.first;
        table[position].second = pair.second;
    }
    return indices ? ref.InsertIndex(pair.first, position) : ref.Insert(pair.first, pair.second);
}

template <class Key, class Value, class Stencil, class Predicate, bool Conditional>
COLLECTION_SIMT_VF LAUNCH_BOUND(INSERT_THREADS) inline void InsertSimt(
    __gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity, Key emptyKey, Value emptyValue,
    __gm__ Pair<Key, Value>* input, __gm__ Stencil* stencil, std::size_t n, __gm__ uint64_t* offsets,
    __gm__ uint64_t* ranks, std::size_t accepted, std::size_t size, __gm__ uint32_t* failed)
{
    StaticMultimapRef<Key, Value> ref{table, capacity, emptyKey, emptyValue, indices};
    if constexpr (!Conditional) {
        (void)ranks;
        for (auto i = ThreadIndex(); i < accepted; i += ThreadStride()) {
            if (!InsertSelected(ref, table, indices, input, i, size + i)) {
                AscendC::Simt::AtomicCas(failed, uint32_t(0), uint32_t(1));
            }
        }
    } else {
        for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
            if (!Predicate{}(stencil[i])) {
                continue;
            }
            auto const rank = ranks[i];
            if (rank >= accepted) {
                continue;
            }
            if (!InsertSelected(ref, table, indices, input, i, size + rank)) {
                AscendC::Simt::AtomicCas(failed, uint32_t(0), uint32_t(1));
            }
        }
    }
}

// Mode 0: Contains, 1: minimum Value.
template <class Key, class Value, class Stencil, class Predicate, bool Conditional, int Mode>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void QuerySimt(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices,
                                                               std::size_t capacity, Key emptyKey, Value emptyValue,
                                                               __gm__ Key* keys, __gm__ Stencil* stencil,
                                                               __gm__ uint8_t* output, std::size_t n)
{
    StaticMultimapRef<Key, Value> ref{table, capacity, emptyKey, emptyValue, indices};
    for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
        bool selected = true;
        if constexpr (Conditional) {
            selected = Predicate{}(stencil[i]);
        }
        if constexpr (Mode == 0) {
            output[i] = selected && ref.Contains(keys[i]);
        }
        if constexpr (Mode == 1) {
            reinterpret_cast<__gm__ Value*>(output)[i] = selected ? ref.Find(keys[i]) : emptyValue;
        }
    }
}

// The count is the discriminator: exactly one activates value; otherwise slot.
// Reading the same active member avoids size_t/Value aliasing and byte-order assumptions.
template <class Value>
union RetrievalMatch {
    std::size_t slot;
    Value value;
};

template <class Key, class Value, bool SaveMatches>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void QueryCountsSimt(__gm__ Pair<Key, Value>* table,
                                                                     __gm__ uint64_t* indices, std::size_t capacity,
                                                                     Key emptyKey, Value emptyValue, __gm__ Key* keys,
                                                                     std::size_t n, __gm__ uint64_t* counts,
                                                                     __gm__ uint64_t* chunkCounts,
                                                                     __gm__ RetrievalMatch<Value>* matches)
{
    constexpr uint32_t WARP = 32;
    auto lane = AscendC::Simt::GetThreadIdx() % WARP;
    StaticMultimapRef<Key, Value> ref{table, capacity, emptyKey, emptyValue, indices};
    auto chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex() / WARP; c < chunks; c += ThreadStride() / WARP) {
        uint64_t sum = 0;
        auto end = (c + 1) * CHUNK;
        if (end > n) {
            end = n;
        }
        for (auto i = c * CHUNK + lane; i < end; i += WARP) {
            if constexpr (SaveMatches) {
                std::size_t first = capacity;
                auto count = ref.Count(keys[i], &first);
                counts[i] = count;
                if (count == 1) {
                    matches[i].value = ref.PairAt(first).second;
                } else {
                    matches[i].slot = first;
                }
                sum += count;
            } else {
                sum += ref.Count(keys[i]);
            }
        }
        for (uint32_t delta = WARP / 2; delta > 0; delta /= 2) {
            sum += AscendC::Simt::WarpShflDownSync(sum, delta, WARP);
        }
        if (lane == 0) {
            chunkCounts[c] = sum;
        }
    }
}

template <class Key, class Value>
COLLECTION_SIMT_DEVICE void WriteMatches(StaticMultimapRef<Key, Value> const& ref, Key key, std::size_t slot,
                                         uint64_t begin, uint64_t count, __gm__ Key* outputKeys,
                                         __gm__ Value* outputValues)
{
    // Count observed the same immutable table; start at its first match
    // and stop after writing the known number of pairs.
    auto out = begin;
    for (std::size_t step = 0; step < ref.capacity; ++step) {
        auto pair = ref.PairAt(slot);
        Key current = pair.first;
        if (current == ref.emptyKey) {
            break;
        }
        if (current == key) {
            outputKeys[out] = key;
            outputValues[out++] = pair.second;
            if (out == begin + count) {
                break;
            }
        }
        if (++slot == ref.capacity) {
            slot = 0;
        }
    }
    // Each query owns a disjoint segment. Heapsort provides deterministic
    // ascending values with O(m log m) work and no per-query scratch array.
    HeapSort(outputValues + begin, count);
}

template <class Key, class Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void RetrieveSimt(
    __gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity, Key emptyKey, Value emptyValue,
    __gm__ Key* keys, std::size_t n, __gm__ uint64_t* counts, __gm__ uint64_t* chunkOffsets,
    __gm__ RetrievalMatch<Value>* matches, __gm__ Key* outputKeys, __gm__ Value* outputValues)
{
    constexpr uint32_t WARP = 32;
    auto lane = AscendC::Simt::GetThreadIdx() % WARP;
    StaticMultimapRef<Key, Value> ref{table, capacity, emptyKey, emptyValue, indices};
    auto chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex() / WARP; c < chunks; c += ThreadStride() / WARP) {
        uint64_t carry = chunkOffsets[c];
        // All lanes participate, including the final partially populated warp.
        for (uint32_t part = 0; part < CHUNK; part += WARP) {
            auto i = c * CHUNK + part + lane;
            uint64_t count = i < n ? counts[i] : 0;
            uint64_t sum = count;
            sum = InclusiveWarpSum(sum, lane);
            auto begin = carry + sum - count;
            carry += AscendC::Simt::WarpShflSync(sum, WARP - 1, WARP);
            if (!count) {
                continue;
            }
            Key key = keys[i];
            if (count == 1) {
                outputKeys[begin] = key;
                outputValues[begin] = matches[i].value;
                continue;
            }
            WriteMatches(ref, key, matches[i].slot, begin, count, outputKeys, outputValues);
        }
    }
}

template <class Key, class Value, bool Write>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void CompactSimt(__gm__ Pair<Key, Value>* table,
                                                                 __gm__ uint64_t* indices, std::size_t capacity,
                                                                 Key emptyKey, __gm__ uint64_t* offsets,
                                                                 __gm__ Key* outputKeys, __gm__ Value* outputValues)
{
    auto chunks = (capacity - 1) / CHUNK + 1;
    for (auto c = ThreadIndex(); c < chunks; c += ThreadStride()) {
        uint64_t out = 0;
        if constexpr (Write) {
            out = offsets[c];
        }
        auto end = (c + 1) * CHUNK;
        if (end > capacity) {
            end = capacity;
        }
        for (auto i = c * CHUNK; i < end; ++i) {
            if (table[i].first == emptyKey) {
                continue;
            }
            if constexpr (Write) {
                outputKeys[out] = table[i].first;
                outputValues[out] = table[i].second;
            }
            ++out;
        }
        if constexpr (!Write) {
            offsets[c] = out;
        }
    }
}

template <class Key, class Value>
COLLECTION_SIMT_DEVICE bool Less(Key a, Value av, Key b, Value bv)
{
    return a < b || (a == b && av < bv);
}

// Dense payload positions are assigned from accepted input ranks. Exporting in
// position order therefore preserves accepted insertion order for every input
// distribution without imposing a sorted-output contract.
template <class Key, class Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void ExportDenseSimt(__gm__ Pair<Key, Value>* pairs, std::size_t n,
                                                                     __gm__ Key* keys, __gm__ Value* values)
{
    for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
        auto pair = pairs[i];
        keys[i] = pair.first;
        values[i] = pair.second;
    }
}
template <class Key, class Value>
COLLECTION_AIV_GLOBAL void ExportDense(__gm__ Pair<Key, Value>* pairs, std::size_t n, __gm__ Key* keys,
                                       __gm__ Value* values)
{
    AscendC::Simt::VF_CALL<ExportDenseSimt<Key, Value>>(AscendC::Simt::Dim3{THREADS}, pairs, n, keys, values);
}

// Merge adjacent sorted runs by binary-searching each element's stable rank
// in the opposite run. Separate input/output buffers prevent overwrite races.
template <class Key, class Value>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void MergeSimt(__gm__ Key* keys, __gm__ Value* values,
                                                               __gm__ Key* outKeys, __gm__ Value* outValues,
                                                               std::size_t n, std::size_t width)
{
    for (auto i = ThreadIndex(); i < n; i += ThreadStride()) {
        auto base = (i / (2 * width)) * (2 * width);
        bool left = i - base < width;
        auto partner = left ? base + width : base;
        auto lo = partner;
        auto hi = partner + width;
        if (lo > n) {
            lo = n;
        }
        if (hi > n) {
            hi = n;
        }
        Key key = keys[i];
        Value value = values[i];
        while (lo < hi) {
            auto mid = lo + (hi - lo) / 2;
            bool before = left ? Less(keys[mid], values[mid], key, value) : !Less(key, value, keys[mid], values[mid]);
            if (before) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        auto rank = lo - (partner < n ? partner : n);
        auto dest = base + (left ? i - base : i - base - width) + rank;
        outKeys[dest] = key;
        outValues[dest] = value;
    }
}

template <class Key, class Value>
COLLECTION_AIV_GLOBAL void Clear(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t n, Key emptyKey,
                                 Value emptyValue)
{
    AscendC::Simt::VF_CALL<ClearSimt<Key, Value>>(AscendC::Simt::Dim3{THREADS}, table, indices, n, emptyKey,
                                                  emptyValue);
}

template <class Key, class Value, class Stencil, class Predicate, bool Conditional>
COLLECTION_AIV_GLOBAL void Validate(__gm__ Pair<Key, Value>* input, __gm__ Stencil* stencil, std::size_t n,
                                    Key emptyKey, Value emptyValue, __gm__ uint64_t* counts, __gm__ uint32_t* invalid)
{
    AscendC::Simt::VF_CALL<ValidateSimt<Key, Value, Stencil, Predicate, Conditional>>(
        AscendC::Simt::Dim3{THREADS}, input, stencil, n, emptyKey, emptyValue, counts, invalid);
}

template <int = 0>
COLLECTION_AIV_GLOBAL void Scan(__gm__ uint64_t* values, __gm__ uint64_t* sums, std::size_t n)
{
    AscendC::Simt::VF_CALL<ScanSimt>(AscendC::Simt::Dim3{THREADS}, values, sums, n);
}

template <int = 0>
COLLECTION_AIV_GLOBAL void Offset(__gm__ uint64_t* values, __gm__ uint64_t* sums, std::size_t n, std::size_t chunks)
{
    AscendC::Simt::VF_CALL<OffsetSimt>(AscendC::Simt::Dim3{THREADS}, values, sums, n, chunks);
}

template <class Stencil, class Predicate>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREADS) inline void RankSimt(__gm__ Stencil* stencil, __gm__ uint64_t* ranks,
                                                              std::size_t n, __gm__ uint64_t* offsets)
{
    auto chunks = (n - 1) / CHUNK + 1;
    for (auto c = ThreadIndex(); c < chunks; c += ThreadStride()) {
        auto rank = offsets[c];
        auto end = (c + 1) * CHUNK;
        if (end > n) {
            end = n;
        }
        for (auto i = c * CHUNK; i < end; ++i) {
            if (Predicate{}(stencil[i])) {
                ranks[i] = rank;
                ++rank;
            }
        }
    }
}

template <class Stencil, class Predicate>
COLLECTION_AIV_GLOBAL void Rank(__gm__ Stencil* stencil, __gm__ uint64_t* ranks, std::size_t n,
                                __gm__ uint64_t* offsets)
{
    AscendC::Simt::VF_CALL<RankSimt<Stencil, Predicate>>(AscendC::Simt::Dim3{THREADS}, stencil, ranks, n, offsets);
}

template <class Key, class Value, class Stencil, class Predicate, bool Conditional>
COLLECTION_AIV_GLOBAL void Insert(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity,
                                  Key emptyKey, Value emptyValue, __gm__ Pair<Key, Value>* input,
                                  __gm__ Stencil* stencil, std::size_t n, __gm__ uint64_t* offsets,
                                  __gm__ uint64_t* ranks, std::size_t accepted, std::size_t size,
                                  __gm__ uint32_t* failed)
{
    AscendC::Simt::VF_CALL<InsertSimt<Key, Value, Stencil, Predicate, Conditional>>(
        AscendC::Simt::Dim3{INSERT_THREADS}, table, indices, capacity, emptyKey, emptyValue, input, stencil, n, offsets,
        ranks, accepted, size, failed);
}

template <class Key, class Value, class Stencil, class Predicate, bool Conditional, int Mode>
COLLECTION_AIV_GLOBAL void Query(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity,
                                 Key emptyKey, Value emptyValue, __gm__ Key* keys, __gm__ Stencil* stencil,
                                 __gm__ uint8_t* output, std::size_t n)
{
    AscendC::Simt::VF_CALL<QuerySimt<Key, Value, Stencil, Predicate, Conditional, Mode>>(
        AscendC::Simt::Dim3{THREADS}, table, indices, capacity, emptyKey, emptyValue, keys, stencil, output, n);
}

template <class Key, class Value>
COLLECTION_AIV_GLOBAL void RetrieveCounts(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices,
                                          std::size_t capacity, Key emptyKey, Value emptyValue, __gm__ Key* keys,
                                          std::size_t n, __gm__ uint64_t* counts, __gm__ uint64_t* chunkCounts,
                                          __gm__ RetrievalMatch<Value>* matches)
{
    AscendC::Simt::VF_CALL<QueryCountsSimt<Key, Value, true>>(AscendC::Simt::Dim3{THREADS}, table, indices, capacity,
                                                              emptyKey, emptyValue, keys, n, counts, chunkCounts,
                                                              matches);
}

template <class Key, class Value>
COLLECTION_AIV_GLOBAL void Count(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity,
                                 Key emptyKey, Value emptyValue, __gm__ Key* keys, std::size_t n,
                                 __gm__ uint64_t* counts)
{
    AscendC::Simt::VF_CALL<QueryCountsSimt<Key, Value, false>>(
        AscendC::Simt::Dim3{THREADS}, table, indices, capacity, emptyKey, emptyValue, keys, n,
        static_cast<__gm__ uint64_t*>(nullptr), counts, static_cast<__gm__ RetrievalMatch<Value>*>(nullptr));
}

template <class Key, class Value>
COLLECTION_AIV_GLOBAL void Retrieve(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity,
                                    Key emptyKey, Value emptyValue, __gm__ Key* keys, std::size_t n,
                                    __gm__ uint64_t* counts, __gm__ uint64_t* chunkOffsets,
                                    __gm__ RetrievalMatch<Value>* matches, __gm__ Key* outputKeys,
                                    __gm__ Value* outputValues)
{
    AscendC::Simt::VF_CALL<RetrieveSimt<Key, Value>>(AscendC::Simt::Dim3{THREADS}, table, indices, capacity, emptyKey,
                                                     emptyValue, keys, n, counts, chunkOffsets, matches, outputKeys,
                                                     outputValues);
}

template <class Key, class Value, bool Write>
COLLECTION_AIV_GLOBAL void Compact(__gm__ Pair<Key, Value>* table, __gm__ uint64_t* indices, std::size_t capacity,
                                   Key emptyKey, __gm__ uint64_t* offsets, __gm__ Key* outputKeys,
                                   __gm__ Value* outputValues)
{
    AscendC::Simt::VF_CALL<CompactSimt<Key, Value, Write>>(AscendC::Simt::Dim3{THREADS}, table, indices, capacity,
                                                           emptyKey, offsets, outputKeys, outputValues);
}

template <class Key, class Value>
COLLECTION_AIV_GLOBAL void Merge(__gm__ Key* keys, __gm__ Value* values, __gm__ Key* outKeys, __gm__ Value* outValues,
                                 std::size_t n, std::size_t width)
{
    AscendC::Simt::VF_CALL<MergeSimt<Key, Value>>(AscendC::Simt::Dim3{THREADS}, keys, values, outKeys, outValues, n,
                                                  width);
}
} // namespace aclco::detail::multimap

#include "detail/static_multimap/insert_warp.inl"
