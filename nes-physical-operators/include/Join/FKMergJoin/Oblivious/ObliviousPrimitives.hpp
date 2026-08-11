/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

/// Oblivious building blocks for the FK-MERG-L4 join, ported from the reference
/// implementation of "Privacy-Preserving Stream Joins" (tee-bench-obliv-stream,
/// Enclave/ObliviousComputationApproach/{outil.h, osort.h}).
///
/// Obliviousness contract: the sequence of memory addresses touched by the sort
/// and merge networks depends only on the (public) array length, never on the
/// data. Every compare-and-swap goes through oMemSwap, which reads and writes
/// both slots regardless of the comparison outcome. The comparator itself may
/// branch on the loaded values (as in the reference); the *addresses* stay fixed.
/// We are not running inside an enclave, so the fidelity target is the
/// algorithmic access pattern, not resistance against a co-located attacker.
///
/// Unlike the reference, which templates the network on the tuple type, NES slot
/// sizes are runtime values (they depend on the query's schemas). All primitives
/// therefore operate on byte arrays of `slotSize`-byte slots whose first
/// sizeof(SlotHeader) bytes are a SlotHeader.
namespace NES::FKMerg
{

/// Fixed 16-byte header at offset 0 of every slot handled by the sort/merge
/// networks. `sortKey` is an order-preserving unsigned normalization of the join
/// key (see normalizeKey in FKMergAlgorithm.hpp); `flags` carries the dummy/null
/// markers and the stream tag. The reference hijacks key == UINT32_MAX as the
/// dummy marker; we use an explicit flag bit since any key value is legal data.
struct SlotHeader
{
    uint64_t sortKey;
    uint64_t flags;

    static constexpr uint64_t FLAG_DUMMY = 1ULL << 0;
    static constexpr uint64_t FLAG_KEY_NULL = 1ULL << 1;
    /// Stream tag for cross-stream merges: 0 = PK side (R), 1 = FK side (S).
    /// PK sorts before FK on equal keys so the scan sees the PK tuple first.
    static constexpr uint64_t FLAG_FK = 1ULL << 2;

    [[nodiscard]] bool isDummy() const { return (flags & FLAG_DUMMY) != 0; }
    [[nodiscard]] bool isKeyNull() const { return (flags & FLAG_KEY_NULL) != 0; }
    [[nodiscard]] bool isFkSide() const { return (flags & FLAG_FK) != 0; }
    /// True only for a real (non-dummy) PK-side slot.
    [[nodiscard]] bool isRealPk() const { return (flags & (FLAG_DUMMY | FLAG_KEY_NULL | FLAG_FK)) == 0; }
};

static_assert(sizeof(SlotHeader) == 16, "SlotHeader layout is part of the slot format");

/// Composite slot order: (isDummy, isKeyNull, sortKey, tableId). Dummies sort
/// last (they are the reference's +infinity padding and get trimmed logically),
/// NULL keys sort after real keys (they never match, SQL semantics), and PK(0)
/// precedes FK(1) on ties. Port of outil.h func_comp/table_comp generalized to
/// the flag-based markers.
inline bool slotLess(const uint8_t* a, const uint8_t* b)
{
    SlotHeader ha;
    SlotHeader hb;
    std::memcpy(&ha, a, sizeof(SlotHeader));
    std::memcpy(&hb, b, sizeof(SlotHeader));

    const uint64_t rankA = ((ha.flags & SlotHeader::FLAG_DUMMY) << 1) | (ha.flags & SlotHeader::FLAG_KEY_NULL);
    const uint64_t rankB = ((hb.flags & SlotHeader::FLAG_DUMMY) << 1) | (hb.flags & SlotHeader::FLAG_KEY_NULL);
    if (rankA != rankB)
    {
        return rankA < rankB;
    }
    if (ha.sortKey != hb.sortKey)
    {
        return ha.sortKey < hb.sortKey;
    }
    return (ha.flags & SlotHeader::FLAG_FK) < (hb.flags & SlotHeader::FLAG_FK);
}

/// XOR-swap of one byte, masked by `cond`: cond == false leaves both bytes
/// untouched, cond == true swaps them. Same instructions and same two addresses
/// touched either way. Port of outil.h o_swapc.
inline void oSwapByte(uint8_t* a, uint8_t* b, const bool cond)
{
    const uint8_t mask = ~(static_cast<uint8_t>(cond) - 1U);
    *a ^= *b;
    *b ^= *a & mask;
    *a ^= *b;
}

/// Branch-free conditional swap of two n-byte slots, byte at a time (as in the
/// reference; oMemSwap dominates neither phase, the comparator networks do).
/// Port of outil.h o_memswap.
inline void oMemSwap(uint8_t* a, uint8_t* b, const size_t n, const bool cond)
{
    for (size_t i = 0; i < n; ++i)
    {
        oSwapByte(&a[i], &b[i], cond);
    }
}

/// Branch-free select over byte ranges: dst = chooseA ? a : b. dst may alias a
/// or b. Port of outil.h conditional_select generalized to byte ranges.
inline void oSelectBytes(uint8_t* dst, const uint8_t* a, const uint8_t* b, const size_t n, const bool chooseA)
{
    const uint8_t mask = ~(static_cast<uint8_t>(chooseA) - 1U);
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = static_cast<uint8_t>((a[i] & mask) | (b[i] & static_cast<uint8_t>(~mask)));
    }
}

/// Round up to the next power of two (the comparator networks require
/// power-of-two lengths; callers pad with dummy slots first). Port of
/// outil.h next_power_of_two widened to 64 bit.
inline uint64_t nextPowerOfTwo(uint64_t n)
{
    if (n == 0)
    {
        return 1;
    }
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

/// Comparator over raw slots for the comparator-parameterized network
/// overloads below. Must be pure and data-independent in its access pattern
/// (it may branch on the loaded values, like the reference comparators).
using SlotComparator = bool (*)(const uint8_t*, const uint8_t*);

namespace detail
{
/// Compare-and-conditionally-swap over `count` contiguous slot pairs.
/// `crossover` selects the bitonic "butterfly" pairing (slot a+i with slot
/// b+count-1-i, used when building a bitonic sequence from two sorted halves)
/// vs. the straight pairing (a+i with b+i, used inside merges). Port of
/// osort.h swap.
inline void compareSwapRange(
    uint8_t* arr,
    const size_t slotSize,
    const size_t a,
    const size_t b,
    const size_t count,
    const bool crossover,
    const bool descending,
    const SlotComparator less)
{
    if (crossover)
    {
        for (size_t i = 0; i < count; ++i)
        {
            uint8_t* lhs = arr + ((a + i) * slotSize);
            uint8_t* rhs = arr + ((b + count - 1 - i) * slotSize);
            const bool cond = less(rhs, lhs) != descending;
            oMemSwap(lhs, rhs, slotSize, cond);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            uint8_t* lhs = arr + ((a + i) * slotSize);
            uint8_t* rhs = arr + ((b + i) * slotSize);
            const bool cond = less(rhs, lhs) != descending;
            oMemSwap(lhs, rhs, slotSize, cond);
        }
    }
}
}

/// Merge stage of the bitonic network: assumes [start, start+count) is bitonic
/// (or, with crossover=true, that its two halves are sorted runs) and sorts it
/// in place. Serial port of osort.h bitonic_merge (threading dropped per the
/// single-threaded-primitives decision). `count` must be a power of two.
inline void bitonicMerge(
    uint8_t* arr,
    const size_t slotSize,
    const size_t start,
    const size_t count,
    const bool crossover,
    const bool descending,
    const SlotComparator less = slotLess)
{
    if (count <= 1)
    {
        return;
    }
    if (count == 2)
    {
        detail::compareSwapRange(arr, slotSize, start, start + 1, 1, false, descending, less);
        return;
    }
    const size_t half = count / 2;
    detail::compareSwapRange(arr, slotSize, start, start + half, half, crossover, descending, less);
    bitonicMerge(arr, slotSize, start, half, false, descending, less);
    bitonicMerge(arr, slotSize, start + half, count - half, false, descending, less);
}

/// Obliviously sorts [start, start+count) via the bitonic network. Serial port
/// of osort.h bitonic_sort (Ngai et al.). `count` must be a power of two —
/// callers pad with dummy slots (the reference logs and no-ops on violations;
/// we assert via the caller's padding logic instead).
inline void bitonicSort(
    uint8_t* arr, const size_t slotSize, const size_t start, const size_t count, const bool descending, const SlotComparator less = slotLess)
{
    if (count <= 1)
    {
        return;
    }
    if (count == 2)
    {
        detail::compareSwapRange(arr, slotSize, start, start + 1, 1, false, descending, less);
        return;
    }
    const size_t half = count / 2;
    bitonicSort(arr, slotSize, start, half, descending, less);
    bitonicSort(arr, slotSize, start + half, count - half, descending, less);
    bitonicMerge(arr, slotSize, start, count, true, descending, less);
}

namespace detail
{
/// Merge step of the oblivious compaction: stitches two already-compacted
/// halves of [start, start+length). Which positions get swapped is computed
/// purely from offset/leftMarkedCount (known from the recursion, not from the
/// data), and every oMemSwap touches the same addresses regardless. Port of
/// ocompact.h swap_local_range (:41-59).
inline void compactionSwapRange(
    uint8_t* arr,
    const size_t slotSize,
    const size_t rangeLength,
    const size_t a,
    const size_t b,
    const size_t pairCount,
    const size_t offset,
    const size_t leftMarkedCount)
{
    const size_t half = rangeLength / 2;
    const bool s = ((offset % half) + leftMarkedCount >= half) != (offset >= half);
    for (size_t i = 0; i < pairCount; ++i)
    {
        const bool cond = s != (i >= (offset + leftMarkedCount) % half);
        oMemSwap(arr + ((a + i) * slotSize), arr + ((b + i) * slotSize), slotSize, cond);
    }
}
}

/// Obliviously moves all marked slots of [start, start+count) to the front,
/// preserving their relative order, without the access pattern revealing which
/// or how many were marked (Sasy et al.'s O(N log N) construction). Serial
/// port of ocompact.h oblivious_compaction (:112-207). `count` must be a power
/// of two — callers pad with unmarked dummy slots. `marks` holds one 0/1 byte
/// per slot of the FULL array (indexed absolutely, like the slots) and is
/// never permuted: the recursion only reads marks at positions that have not
/// moved yet (counts happen before the recursive calls, base-case pairs read
/// their marks before swapping). `offset` rotates the compaction target and is
/// 0 for top-level calls.
inline void obliviousCompactPow2(
    uint8_t* arr, const size_t slotSize, const uint8_t* marks, const size_t start, const size_t count, const size_t offset)
{
    if (count < 2)
    {
        return;
    }
    if (count == 2)
    {
        const bool cond = ((marks[start] == 0) && (marks[start + 1] != 0)) != (offset != 0);
        oMemSwap(arr + (start * slotSize), arr + ((start + 1) * slotSize), slotSize, cond);
        return;
    }

    const size_t half = count / 2;
    size_t leftMarkedCount = 0;
    for (size_t i = 0; i < half; ++i)
    {
        leftMarkedCount += marks[start + i];
    }

    obliviousCompactPow2(arr, slotSize, marks, start, half, offset % half);
    obliviousCompactPow2(arr, slotSize, marks, start + half, half, (offset + leftMarkedCount) % half);
    const size_t rangeLength = count;
    const size_t pairCount = half;
    detail::compactionSwapRange(arr, slotSize, rangeLength, start, start + half, pairCount, offset, leftMarkedCount);
}

}
