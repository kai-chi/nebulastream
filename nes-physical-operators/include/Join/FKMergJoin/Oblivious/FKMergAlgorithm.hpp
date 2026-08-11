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

#include <cstdint>
#include <functional>
#include <DataTypes/DataType.hpp>
#include <Join/FKMergJoin/Oblivious/ObliviousPrimitives.hpp>

/// The FK-MERG-L4 algorithm core: OAppend-maintained sorted windows, the tagged
/// cross-stream merge, and the dummy-emitting scan. Ported from
/// tee-bench-obliv-stream (OCA.cpp oblivious_append variants; scan from
/// Enclave/Opaque/OpaqueJoin.cpp, whose serial formulation carries the last
/// PK tuple and performs the key-equality test).
///
/// All functions here are plain native C++ and must only be called from
/// nautilus::invoke proxies or operator-handler code, never from traced code.
namespace NES::FKMerg
{

/// The oblivious FK join variant executed (paper Algorithm 4 and §VII-A).
/// Two families: MERG keeps the windows key-sorted via OAppend and combines
/// them with a cheap bitonic merge (O(N log N)); SORT (the Opaque-style
/// baseline, reference Enclave/Opaque/OpaqueJoin) keeps nothing sorted and
/// runs a full bitonic sort per join invocation (O(N log^2 N)). Three leakage
/// levels each: L4 emits the worst-case padded output, L3 obliviously
/// compacts the dummies away (leaking the per-window cardinality), L2
/// processes tuple-at-a-time in event-time order with per-tuple compaction
/// (leaking each tuple's degree).
enum class FKMergVariant : uint8_t
{
    MERG_L2,
    MERG_L3,
    MERG_L4,
    SORT_L2,
    SORT_L3,
    SORT_L4,
    /// The non-FK generic join (Krastnikov-based, duplicates on both sides,
    /// see NfkJoin.hpp). No L4 variant exists: the non-FK worst case is the
    /// Cartesian product.
    NFK_L2,
    NFK_L3
};

/// L2 variants replay the window tuple-by-tuple at trigger time.
constexpr bool isPerTupleVariant(const FKMergVariant variant)
{
    return variant == FKMergVariant::MERG_L2 || variant == FKMergVariant::SORT_L2 || variant == FKMergVariant::NFK_L2;
}

/// L3 variants obliviously compact the window result before emitting (the NFK
/// variants never materialize dummies in the first place).
constexpr bool compactsWindowResult(const FKMergVariant variant)
{
    return variant == FKMergVariant::MERG_L3 || variant == FKMergVariant::SORT_L3;
}

/// SORT variants keep the window state unsorted (raw arrival logs) and run a
/// full oblivious sort per combine; MERG variants maintain key-sorted windows
/// via OAppend and combine with a bitonic merge.
constexpr bool usesFullSort(const FKMergVariant variant)
{
    return variant == FKMergVariant::SORT_L2 || variant == FKMergVariant::SORT_L3 || variant == FKMergVariant::SORT_L4;
}

/// The NFK variants run the Krastnikov-based expansion join.
constexpr bool isNfkVariant(const FKMergVariant variant)
{
    return variant == FKMergVariant::NFK_L2 || variant == FKMergVariant::NFK_L3;
}

/// Only the MERG L3/L4 variants maintain key-sorted windows via OAppend at
/// build time; everything else stages raw arrival logs.
constexpr bool maintainsSortedWindows(const FKMergVariant variant)
{
    return variant == FKMergVariant::MERG_L3 || variant == FKMergVariant::MERG_L4;
}

/// Describes one input side's fixed-size row format (DefaultPagedVectorTupleLayout:
/// fields contiguous in schema order, each occupying getSizeInBytesWithNull()
/// bytes, nullable fields prefixed by one null-indicator byte).
struct SideLayout
{
    /// Total bytes of one row.
    uint64_t rowSize = 0;
    /// Byte offset of the join-key field within the row (points at the null
    /// indicator byte if the key is nullable, else at the value).
    uint64_t keyOffset = 0;
    DataType::Type keyType = DataType::Type::UNDEFINED;
    bool keyNullable = false;
    /// Event-time field of the row, used by the L2 variant to replay tuples in
    /// arrival (event-time) order. Unused by L3/L4.
    uint64_t tsOffset = 0;
    DataType::Type tsType = DataType::Type::UNDEFINED;
    bool tsNullable = false;

    /// Slots in the per-side sorted arrays: SlotHeader followed by the raw row.
    [[nodiscard]] uint64_t slotSize() const { return sizeof(SlotHeader) + rowSize; }

    /// A view of this layout whose "key" is the event-time field, for building
    /// timestamp-ordered slot headers.
    [[nodiscard]] SideLayout timestampView() const
    {
        SideLayout view = *this;
        view.keyOffset = tsOffset;
        view.keyType = tsType;
        view.keyNullable = tsNullable;
        return view;
    }
};

/// A sorted-by-key slot array for one side of one window. Memory is owned by
/// the caller (the join slice retains the backing TupleBuffers); this struct
/// only tracks the current view.
struct SortedSide
{
    uint8_t* slots = nullptr;
    uint64_t sizeSlots = 0;
    uint64_t capacitySlots = 0;
};

/// Allocation callback for growing a SortedSide: returns a pointer to at least
/// `bytes` bytes whose lifetime the caller manages (backed by an unpooled
/// TupleBuffer retained by the slice).
using AllocateFn = std::function<uint8_t*(uint64_t bytes)>;

/// Order-preserving normalization of the join key at `row + layout.keyOffset`
/// into the 64-bit unsigned sortKey domain. Sets `isNull` from the null
/// indicator byte (false for non-nullable keys). Signed types are
/// sign-flipped so that unsigned comparison preserves their order; the mapping
/// is injective per type, so sortKey equality is key equality.
uint64_t normalizeKey(const uint8_t* row, const SideLayout& layout, bool& isNull);

/// OAppend (paper Algorithm 3, port of OCA.cpp:713-758): obliviously appends
/// `numBatchRows` contiguous raw rows to the sorted side. Grows the slot array
/// to nextPowerOfTwo(2 * max(size, batch)) if needed (allocation strategy
/// improved over the reference's realloc churn; per-element access patterns
/// unchanged), lays out [sorted ascending | dummy gap | batch right-aligned],
/// bitonic-sorts the power-of-two batch region descending, then runs the
/// straight bitonic merge over the whole array. Dummies sort to the tail and
/// are trimmed logically (sizeSlots grows by numBatchRows, no shrink).
void oAppend(SortedSide& side, const SideLayout& layout, bool fkSide, const uint8_t* batchRows, uint64_t numBatchRows, const AllocateFn& allocate);

/// Slot size of the tagged cross-stream merge arrays: SlotHeader plus a
/// payload union area large enough for either side's row.
uint64_t mergedSlotSize(const SideLayout& pkLayout, const SideLayout& fkLayout);

/// Number of (padded) slots crossMerge requires in its scratch array.
uint64_t crossMergePaddedSlots(uint64_t numPkSlots, uint64_t numFkSlots);

/// Tagged cross-stream merge (port of OCA.cpp:815-889): writes PK slots
/// ascending tagged tableId=0, a dummy gap, and FK slots reversed (descending,
/// a fixed data-independent permutation since the side is already sorted)
/// right-aligned tagged tableId=1 into `scratch`, then runs the straight
/// bitonic merge. `scratch` must hold crossMergePaddedSlots(...) slots of
/// mergedSlotSize(...) bytes. Returns the logical slot count (nPk + nFk).
uint64_t crossMerge(const SortedSide& pkSide, const SideLayout& pkLayout, const SortedSide& fkSide, const SideLayout& fkLayout, uint8_t* scratch);

/// Byte size of one output slot: a uint64 flags word (bit0 = dummy) followed
/// by the PK row and the FK row.
uint64_t outputSlotSize(const SideLayout& pkLayout, const SideLayout& fkLayout);

struct ScanResult
{
    /// Real (non-dummy) matches emitted.
    uint64_t realMatches = 0;
    /// Two real PK tuples shared a key: the FK-join precondition is violated
    /// and some matches were silently overwritten. Caller must fail the query.
    bool duplicatePkDetected = false;
};

/// Appends raw rows to an arrival log: slots headered by the normalized event
/// timestamp (via SideLayout::timestampView) in arrival order, no sorting.
/// Used by the L2 variant, which replays the log tuple-by-tuple at trigger
/// time. Grows the log to power-of-two capacities via `allocate`.
void appendToArrivalLog(
    SortedSide& log, const SideLayout& layout, bool fkSide, const uint8_t* batchRows, uint64_t numBatchRows, const AllocateFn& allocate);

/// FK-SORT's combine step (reference Enclave/Opaque/OpaqueJoin.cpp:155-191,
/// generalized to whole windows like crossMerge): copies both sides' rows —
/// in whatever order they arrived — into `scratch` tagged PK/FK with fresh
/// key headers, pads to a power of two with dummies, and runs a FULL bitonic
/// sort (O(N log^2 N); nothing is kept sorted between invocations, that is
/// the point of the baseline). `scratch` must hold
/// nextPowerOfTwo(nPk + nFk) slots of mergedSlotSize(...) bytes. Returns the
/// logical slot count (nPk + nFk); the result feeds obliviousScan exactly
/// like crossMerge's.
uint64_t sortedCrossCombine(
    const SortedSide& pkSide, const SideLayout& pkLayout, const SortedSide& fkSide, const SideLayout& fkLayout, uint8_t* scratch);

/// FK-MERG-L2 / FK-SORT-L2 (reference l2v2_join, OCA.cpp:190-320, and
/// OpaqueJoin's batch-size-1 mode) replayed at window close:
/// obliviously sorts both arrival logs by event time (in place, logs must have
/// nextPowerOfTwo(size) slot capacity), then walks the tuples in global
/// event-time order — ties process the FK side first, mirroring the
/// reference's strict `next_is_R` comparison — maintaining two replay
/// windows. Per tuple: append into its own window and combine the single
/// tuple with the opposite window — fullSort=false (MERG_L2) keeps the
/// windows key-sorted via OAppend and combines with the bitonic merge,
/// fullSort=true (SORT_L2) keeps them as raw logs and runs a full bitonic
/// sort per tuple — then scan and compact the per-tuple output (leaking that
/// tuple's degree), appending the real matches to `outSlots` (capacity: one
/// slot per FK-side tuple). Timestamps are public in
/// the threat model, so the data-dependent walk order is permitted. Duplicate
/// PK keys are detected by a final adjacent-key pass over the PK replay
/// window. Returns the total match count and the duplicate-PK flag.
ScanResult perTupleReplay(
    SortedSide& pkLog,
    const SideLayout& pkLayout,
    SortedSide& fkLog,
    const SideLayout& fkLayout,
    bool fullSort,
    uint8_t* outSlots,
    uint64_t outCapacitySlots,
    const AllocateFn& allocate);

/// The L3 result trim (paper Algorithm 4, line 10 — OCompaction): obliviously
/// moves the real output slots to the front (stable) so that only they are
/// emitted, leaking the per-window join cardinality but nothing else. Pads
/// [numSlots, nextPowerOfTwo(numSlots)) with dummy output slots first — the
/// caller must have allocated that many slots — then runs the power-of-two
/// oblivious compaction on the dummy bit. Returns the number of real slots.
uint64_t trimDummies(uint8_t* outSlots, uint64_t numSlots, uint64_t outSlotSize);

/// The dummy-emitting scan (port of Enclave/Opaque/OpaqueJoin.cpp:20-37):
/// walks the merged array carrying the last PK slot via branch-free selects
/// and emits exactly one output slot per merged slot — the joined (pkRow,
/// fkRow) pair if the slot is a real FK tuple whose key equals the carried
/// PK's, else a dummy (flags bit0 set, both row regions all-zero sentinel
/// bytes). Output volume therefore equals the merged slot count, independent
/// of the data. Duplicate real PK keys are detected branch-free during the
/// walk and reported in the result.
ScanResult obliviousScan(
    const uint8_t* merged,
    uint64_t numMergedSlots,
    const SideLayout& pkLayout,
    const SideLayout& fkLayout,
    uint8_t* out);

}
