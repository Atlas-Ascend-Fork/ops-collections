/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>
#include "extent.h"
#include "detail/static_multimap/kernels.h"
#include "detail/static_multimap/storage.h"
#include "tiling/platform/platform_ascendc.h"

namespace aclco {
namespace detail::multimap {
struct StaticMultimapTestAccess;
enum class StorageMode { Automatic, Direct };
inline uint32_t Blocks()
{
    auto count = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    if (!count) {
        throw std::runtime_error("StaticMultimap: no AIV cores");
    }
    return count;
}
inline std::size_t Chunks(std::size_t n) { return (n - 1) / CHUNK + 1; }

// Workspace ownership outlives every queued kernel, including recursive scans.
inline void PreparePrefix(std::size_t n, std::vector<Buffer<uint64_t>>& workspace)
{
    for (std::size_t level = 0; n; ++level) {
        auto chunks = Chunks(n);
        if (level == workspace.size()) workspace.emplace_back();
        workspace[level].Reserve(chunks + 1);
        if (chunks == 1) break;
        n = chunks;
    }
}

inline void PrefixPrepared(uint64_t* values, std::size_t n, aclrtStream stream,
                           std::vector<Buffer<uint64_t>>& workspace, std::size_t level)
{
    auto chunks = Chunks(n);
    auto sums = workspace[level].Data();
    Scan<><<<Blocks(), 0, stream>>>(values, sums, n);
    if (chunks > 1) {
        PrefixPrepared(sums, chunks, stream, workspace, level + 1);
    }
    Offset<><<<Blocks(), 0, stream>>>(values, sums, n, chunks);
}

inline void Prefix(uint64_t* values, std::size_t n, aclrtStream stream, std::vector<Buffer<uint64_t>>& workspace)
{
    if (!n) return;
    PreparePrefix(n, workspace);
    PrefixPrepared(values, n, stream, workspace, 0);
}
} // namespace detail::multimap

/**
 * @brief Fixed-capacity, synchronous Device multimap for signed I32 and I64 pairs.
 * @note Nonempty Device input/output buffers must have the declared type and
 * sufficient allocation size. Query outputs must not alias inputs or each other.
 * All batch methods synchronize the supplied stream before returning.
 * Destroy the object before resetting or finalizing its ACL device context.
 * Duplicate pairs occupy independent slots. Find returns the minimum Value.
 * Retrieve preserves query order and sorts each query's values. RetrieveAll
 * returns the complete pair multiset in an unspecified order. Concurrent calls
 * require external serialization, including calls on different ACL streams.
 */
template <class Key, class Value>
class StaticMultimap {
    static_assert((std::is_same_v<Key, int32_t> || std::is_same_v<Key, int64_t>) &&
                      (std::is_same_v<Value, int32_t> || std::is_same_v<Value, int64_t>),
                  "StaticMultimap supports signed I32/I64 keys and values");
    struct QueryScratch {
        detail::multimap::Buffer<uint64_t> counts, offsets;
        detail::multimap::Buffer<detail::multimap::RetrievalMatch<Value>> matches;
        std::vector<detail::multimap::Buffer<uint64_t>> workspace;
        aclrtStream pendingStream = nullptr;
        bool pending = false;

        void Wait()
        {
            if (pending) {
                detail::multimap::Sync(pendingStream);
                pending = false;
            }
        }
        void Begin(aclrtStream stream) { pendingStream = stream; pending = true; }
        void Prepare(std::size_t n, std::size_t chunks, bool retrieve)
        {
            Wait();
            counts.Reserve(retrieve ? n : chunks + 1);
            if (retrieve) {
                offsets.Reserve(chunks + 1);
                matches.Reserve(n);
            }
            detail::multimap::PreparePrefix(chunks, workspace);
        }
        ~QueryScratch()
        {
            // On an exceptional exit, scratch and table must outlive queued work.
            if (pending) (void)aclrtSynchronizeStream(pendingStream);
        }
    };
    struct State {
        std::size_t capacity;
        std::size_t size = 0;
        Key emptyKey;
        Value emptyValue;
        detail::multimap::Buffer<Pair<Key, Value>> table;
        detail::multimap::Buffer<uint64_t> indices;
        detail::multimap::Buffer<uint32_t> status;
        // Destroyed before table/indices; externally serialized const queries reuse it.
        mutable QueryScratch queryScratch;
        State(std::size_t n, Key k, Value v, detail::multimap::StorageMode mode)
            : capacity(n),
              emptyKey(k),
              emptyValue(v),
              table(n),
              indices(mode == detail::multimap::StorageMode::Automatic && n <= std::numeric_limits<uint32_t>::max() ?
                          n :
                          0),
              status(1)
        {}
    };
    std::unique_ptr<State> state_;
    friend struct detail::multimap::StaticMultimapTestAccess;

    // Allows small-capacity tests to exercise direct storage.
    StaticMultimap(aclco::Extent<std::size_t> capacity, Key emptyKey, Value emptyValue, aclrtStream stream,
                   detail::multimap::StorageMode mode)
    {
        if (std::size_t(capacity) == 0) {
            throw std::invalid_argument("StaticMultimap: capacity must be positive");
        }
        state_ = std::make_unique<State>(std::size_t(capacity), emptyKey, emptyValue, mode);
        Clear(stream);
    }

    State& Get() const
    {
        if (!state_) {
            throw std::logic_error("StaticMultimap: moved-from object");
        }
        state_->queryScratch.Wait();
        return *state_;
    }
    void CheckCount(std::size_t n) const
    {
        detail::multimap::Bytes<Key>(n);
        auto const size = Get().size;
        if (size == 0) {
            return;
        }
        if (n > std::numeric_limits<uint64_t>::max() / size) {
            throw std::overflow_error("StaticMultimap: potential total count overflow");
        }
    }

    std::size_t QueryChunks(void* keys, std::size_t n) const
    {
        if (!n) {
            return 0;
        }
        detail::multimap::RequirePointer(keys);
        CheckCount(n);
        return detail::multimap::Chunks(n);
    }

    template <class Stencil, class Predicate, bool Conditional>
    std::size_t CountSelected(void* pairs, Stencil* stencil, std::size_t n,
                              detail::multimap::Buffer<uint64_t>& counts, aclrtStream stream)
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        mm::RequirePointer(pairs);
        if constexpr (Conditional) {
            mm::RequirePointer(stencil);
            mm::Bytes<Stencil>(n);
        }
        mm::Bytes<Pair<Key, Value>>(n);
        auto chunks = mm::Chunks(n);
        if constexpr (Conditional) {
            counts = mm::Buffer<uint64_t>(chunks + 1);
        }
        auto status = s.status.Data();
        mm::Check(aclrtMemsetAsync(status, sizeof(uint32_t), 0, sizeof(uint32_t), stream), "clear status");
        mm::Validate<Key, Value, Stencil, Predicate, Conditional><<<mm::Blocks(), 0, stream>>>(
            static_cast<Pair<Key, Value>*>(pairs), stencil, n, s.emptyKey, s.emptyValue, counts.Data(), status);
        std::vector<mm::Buffer<uint64_t>> workspace;
        if constexpr (Conditional) {
            mm::Prefix(counts.Data(), chunks, stream, workspace);
        }
        mm::Sync(stream);
        if (mm::Read(status)) {
            throw std::invalid_argument("StaticMultimap: selected input contains a reserved key or value");
        }
        if constexpr (Conditional) {
            return mm::Read(counts.Data() + chunks);
        }
        return n;
    }

    template <class Stencil, class Predicate, bool Conditional>
    std::size_t InsertImpl(void* pairs, Stencil* stencil, std::size_t n, aclrtStream stream)
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        if (!n) {
            return 0;
        }
        mm::Buffer<uint64_t> counts;
        auto selected = CountSelected<Stencil, Predicate, Conditional>(pairs, stencil, n, counts, stream);
        auto accepted = std::min(selected, s.capacity - s.size);
        if (!accepted) {
            return selected;
        }
        auto status = s.status.Data();
        auto* input = static_cast<Pair<Key, Value>*>(pairs);
        auto insertRanked = [&](uint64_t* ranks) {
            mm::Insert<Key, Value, Stencil, Predicate, Conditional><<<mm::Blocks(), 0, stream>>>(
                s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey, s.emptyValue, input, stencil, n,
                counts.Data(), ranks, accepted, s.size, status);
        };
        if constexpr (Conditional) {
            if (n <= mm::rankedInsertMaximumKeys) {
                mm::Buffer<uint64_t> ranks(n);
                mm::Rank<Stencil, Predicate><<<mm::Blocks(), 0, stream>>>(stencil, ranks.Data(), n, counts.Data());
                insertRanked(ranks.Data());
            } else {
                mm::InsertWarp<Key, Value, Stencil, Predicate><<<mm::Blocks(), 0, stream>>>(
                    s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey, s.emptyValue, input, stencil, n,
                    counts.Data(), accepted, s.size, status);
            }
        } else {
            insertRanked(nullptr);
        }
        mm::Sync(stream);
        if (mm::Read(status)) {
            // The admitted prefix fits by construction. A failure indicates a
            // violated execution invariant, not normal capacity exhaustion.
            throw std::runtime_error("StaticMultimap: bounded insertion failed despite available capacity");
        }
        s.size += accepted;
        return selected - accepted;
    }

    template <class Stencil, class Predicate, bool Conditional, int Mode>
    void QueryImpl(void* keys, Stencil* stencil, void* output, std::size_t n, aclrtStream stream) const
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        if (!n) {
            return;
        }
        mm::RequirePointer(keys);
        mm::RequirePointer(output);
        mm::Bytes<Key>(n);
        if constexpr (Mode == 1) {
            mm::Bytes<Value>(n);
        }
        if constexpr (Conditional) {
            mm::RequirePointer(stencil);
            mm::Bytes<Stencil>(n);
        }
        mm::Query<Key, Value, Stencil, Predicate, Conditional, Mode>
            <<<mm::Blocks(), 0, stream>>>(s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey, s.emptyValue,
                                          static_cast<Key*>(keys), stencil, static_cast<uint8_t*>(output), n);
        mm::Sync(stream);
    }

public:
    using ExtentType = aclco::Extent<std::size_t>;
    using SizeType = std::size_t;
    /**
     * @brief Construct a fixed-capacity synchronous Device multimap.
     * @throws std::invalid_argument If capacity is zero.
     * @note Device allocation failures are reported as exceptions.
     */
    StaticMultimap(ExtentType capacity, Key emptyKey, Value emptyValue, aclrtStream stream = nullptr)
        : StaticMultimap(capacity, emptyKey, emptyValue, stream, detail::multimap::StorageMode::Automatic)
    {}
    ~StaticMultimap() = default;
    StaticMultimap(StaticMultimap const&) = delete;
    StaticMultimap& operator=(StaticMultimap const&) = delete;
    StaticMultimap(StaticMultimap&&) noexcept = default;
    StaticMultimap& operator=(StaticMultimap&&) noexcept = default;

    /**
     * @brief Return the exact slot capacity.
     * @return Capacity, or zero for a moved-from object.
     */
    SizeType Capacity() const noexcept { return state_ ? state_->capacity : 0; }
    /**
     * @brief Return the number of retained pairs, including duplicates.
     * @return Number of pairs.
     * @throws std::logic_error If the object was moved from.
     */
    SizeType Size(aclrtStream stream = nullptr) const
    {
        detail::multimap::Sync(stream);
        return Get().size;
    }
    void Clear(aclrtStream stream = nullptr)
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        mm::Clear<Key, Value>
            <<<mm::Blocks(), 0, stream>>>(s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey, s.emptyValue);
        mm::Sync(stream);
        s.size = 0;
    }
    /**
     * @brief Insert the input prefix that fits; retain duplicate pairs.
     * @return Number of pairs that were not admitted.
     * @throws std::invalid_argument If a nonempty input contains a reserved key or value.
     * @note All input pairs are validated before any insertion.
     */
    SizeType Insert(void* pairs, ExtentType n, aclrtStream stream = nullptr)
    {
        (void)Get();
        if (!pairs) {
            return SizeType(n);
        }
        return InsertImpl<uint8_t, detail::multimap::SelectAll, false>(pairs, nullptr, n, stream);
    }
    /**
     * @brief Insert the selected input prefix that fits, preserving selection order.
     * @return Number of selected pairs that were not admitted.
     * @throws std::invalid_argument If a required pointer is null or a selected pair uses a sentinel.
     * @note Predicate must be deterministic and free of side effects across invocations.
     */
    template <class Stencil, class Predicate>
    SizeType InsertIf(void* pairs, Stencil* stencil, ExtentType n, aclrtStream stream = nullptr)
    {
        return InsertImpl<Stencil, Predicate, true>(pairs, stencil, n, stream);
    }
    void Contains(void* keys, void* output, ExtentType n, aclrtStream stream = nullptr) const
    {
        QueryImpl<uint8_t, detail::multimap::SelectAll, false, 0>(keys, nullptr, output, n, stream);
    }
    template <class Stencil, class Predicate>
    void ContainsIf(void* keys, Stencil* stencil, void* output, ExtentType n, aclrtStream stream = nullptr) const
    {
        QueryImpl<Stencil, Predicate, true, 0>(keys, stencil, output, n, stream);
    }
    void Find(void* keys, void* output, ExtentType n, aclrtStream stream = nullptr) const
    {
        QueryImpl<uint8_t, detail::multimap::SelectAll, false, 1>(keys, nullptr, output, n, stream);
    }
    template <class Stencil, class Predicate>
    void FindIf(void* keys, Stencil* stencil, void* output, ExtentType n, aclrtStream stream = nullptr) const
    {
        QueryImpl<Stencil, Predicate, true, 1>(keys, stencil, output, n, stream);
    }
    /**
     * @brief Sum matching multiplicities over all queries, including repeated queries.
     * @return Total number of matching query-pair combinations.
     * @throws std::overflow_error If the possible total or buffer size exceeds its type.
     */
    uint64_t Count(void* keys, ExtentType count, aclrtStream stream = nullptr) const
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        auto chunks = QueryChunks(keys, count);
        if (!chunks) {
            return 0;
        }
        SizeType n = count;
        auto& scratch = s.queryScratch;
        scratch.Prepare(n, chunks, false);
        auto& counts = scratch.counts;
        scratch.Begin(stream);
        mm::Count<Key, Value><<<mm::Blocks(), 0, stream>>>(s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey,
                                                           s.emptyValue, static_cast<Key*>(keys), n, counts.Data());
        mm::PrefixPrepared(counts.Data(), chunks, stream, scratch.workspace, 0);
        scratch.Wait();
        return mm::Read(counts.Data() + chunks);
    }
    /**
     * @brief Retrieve matches in query order, with ascending values per query.
     * @return Total number of output pairs.
     * @throws std::length_error If outputCapacity is too small; outputs remain unchanged.
     * @note Null outputs are allowed only when no result is produced.
     */
    SizeType Retrieve(void* keys, ExtentType count, void* probeOut, void* valueOut, ExtentType outputCapacity,
                      aclrtStream stream = nullptr) const
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        auto chunks = QueryChunks(keys, count);
        if (!chunks) {
            return 0;
        }
        SizeType n = count;
        auto& scratch = s.queryScratch;
        scratch.Prepare(n, chunks, true);
        auto& counts = scratch.counts;
        auto& offsets = scratch.offsets;
        auto& matches = scratch.matches;
        scratch.Begin(stream);
        mm::RetrieveCounts<Key, Value><<<mm::Blocks(), 0, stream>>>(s.table.Data(), s.indices.Data(), s.capacity,
                                                                    s.emptyKey, s.emptyValue, static_cast<Key*>(keys),
                                                                    n, counts.Data(), offsets.Data(), matches.Data());
        mm::PrefixPrepared(offsets.Data(), chunks, stream, scratch.workspace, 0);
        scratch.Wait();
        auto total = mm::Read(offsets.Data() + chunks);
        if (total > SizeType(outputCapacity)) {
            throw std::length_error("StaticMultimap: output capacity is insufficient");
        }
        if (!total) {
            return 0;
        }
        mm::RequirePointer(probeOut);
        mm::RequirePointer(valueOut);
        mm::Bytes<Key>(total);
        mm::Bytes<Value>(total);
        scratch.Begin(stream);
        mm::Retrieve<Key, Value><<<mm::Blocks(), 0, stream>>>(
            s.table.Data(), s.indices.Data(), s.capacity, s.emptyKey, s.emptyValue, static_cast<Key*>(keys), n,
            counts.Data(), offsets.Data(), matches.Data(), static_cast<Key*>(probeOut), static_cast<Value*>(valueOut));
        scratch.Wait();
        return total;
    }
    /**
     * @brief Export all retained pairs, including duplicates.
     * @return Number of output pairs.
     * @throws std::length_error If outputCapacity is too small; outputs remain unchanged.
     * @note Output order is unspecified. Dense storage currently preserves accepted
     * insertion order, but callers must compare the returned pair multiset.
     * Empty containers accept null outputs.
     */
    SizeType RetrieveAll(void* keyOut, void* valueOut, ExtentType outputCapacity, aclrtStream stream = nullptr) const
    {
        namespace mm = detail::multimap;
        auto& s = Get();
        if (s.size > SizeType(outputCapacity)) {
            throw std::length_error("StaticMultimap: output capacity is insufficient");
        }
        if (!s.size) {
            return 0;
        }
        mm::RequirePointer(keyOut);
        mm::RequirePointer(valueOut);
        auto keys = static_cast<Key*>(keyOut);
        auto values = static_cast<Value*>(valueOut);
        if (s.indices.Data()) {
            mm::ExportDense<Key, Value><<<mm::Blocks(), 0, stream>>>(s.table.Data(), s.size, keys, values);
            mm::Sync(stream);
            return s.size;
        } else {
            auto chunks = mm::Chunks(s.capacity);
            mm::Buffer<uint64_t> offsets(chunks + 1);
            std::vector<mm::Buffer<uint64_t>> workspace;
            mm::Compact<Key, Value, false><<<mm::Blocks(), 0, stream>>>(s.table.Data(), nullptr, s.capacity, s.emptyKey,
                                                                        offsets.Data(), nullptr, nullptr);
            mm::Prefix(offsets.Data(), chunks, stream, workspace);
            mm::Compact<Key, Value, true><<<mm::Blocks(), 0, stream>>>(s.table.Data(), nullptr, s.capacity, s.emptyKey,
                                                                       offsets.Data(), keys, values);
            mm::Sync(stream);
        }
        // Output order is unspecified; compaction already exports all pairs.
        return s.size;
    }
};
} // namespace aclco
