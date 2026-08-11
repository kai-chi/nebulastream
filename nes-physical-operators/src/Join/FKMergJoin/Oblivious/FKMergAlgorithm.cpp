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

#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <DataTypes/DataType.hpp>
#include <Join/FKMergJoin/Oblivious/ObliviousPrimitives.hpp>
#include <ErrorHandling.hpp>

namespace NES::FKMerg
{

namespace
{

constexpr uint64_t SIGN_FLIP = 1ULL << 63U;

/// A dummy slot header: the reference's +infinity padding tuple. The sortKey is
/// set to the maximum so that even a comparator ignoring the dummy flag keeps
/// them last; the composite comparator ranks on the flag first regardless.
constexpr SlotHeader DUMMY_HEADER{.sortKey = UINT64_MAX, .flags = SlotHeader::FLAG_DUMMY};

/// Writes a dummy slot: dummy header plus a zeroed payload region, so the whole
/// slot content is deterministic (slots move through the comparator networks in
/// full).
void writeDummySlot(uint8_t* slot, const uint64_t slotSize)
{
    std::memcpy(slot, &DUMMY_HEADER, sizeof(SlotHeader));
    std::memset(slot + sizeof(SlotHeader), 0, slotSize - sizeof(SlotHeader));
}

/// Builds the slot header for a raw row of the given side.
SlotHeader makeHeader(const uint8_t* row, const SideLayout& layout, const bool fkSide)
{
    bool isNull = false;
    const uint64_t sortKey = normalizeKey(row, layout, isNull);
    uint64_t flags = 0;
    flags |= isNull ? SlotHeader::FLAG_KEY_NULL : 0;
    flags |= fkSide ? SlotHeader::FLAG_FK : 0;
    return SlotHeader{.sortKey = sortKey, .flags = flags};
}

}

uint64_t normalizeKey(const uint8_t* row, const SideLayout& layout, bool& isNull)
{
    const uint8_t* keyPtr = row + layout.keyOffset;
    isNull = false;
    if (layout.keyNullable)
    {
        isNull = *keyPtr != 0;
        keyPtr += 1;
    }
    switch (layout.keyType)
    {
        case DataType::Type::INT32: {
            int32_t value = 0;
            std::memcpy(&value, keyPtr, sizeof(value));
            return static_cast<uint64_t>(static_cast<int64_t>(value)) ^ SIGN_FLIP;
        }
        case DataType::Type::INT64: {
            int64_t value = 0;
            std::memcpy(&value, keyPtr, sizeof(value));
            return static_cast<uint64_t>(value) ^ SIGN_FLIP;
        }
        case DataType::Type::UINT32: {
            uint32_t value = 0;
            std::memcpy(&value, keyPtr, sizeof(value));
            return value;
        }
        case DataType::Type::UINT64: {
            uint64_t value = 0;
            std::memcpy(&value, keyPtr, sizeof(value));
            return value;
        }
        default:
            INVARIANT(false, "FK_MERG_L4 supports INT32/INT64/UINT32/UINT64 join keys, the lowering rule must reject anything else");
            return 0;
    }
}

void oAppend(
    SortedSide& side,
    const SideLayout& layout,
    const bool fkSide,
    const uint8_t* batchRows,
    const uint64_t numBatchRows,
    const AllocateFn& allocate)
{
    if (numBatchRows == 0)
    {
        return;
    }
    const uint64_t slotSize = layout.slotSize();
    const uint64_t half = std::max(side.sizeSlots, numBatchRows);
    const uint64_t padded = nextPowerOfTwo(2 * half);

    /// Grow the slot array if needed. The reference reallocs up and down every
    /// append (OCA.cpp:727,756, "TODO: remove reallocs"); we allocate a larger
    /// array and copy the sorted prefix sequentially — a data-independent
    /// access pattern — and keep the capacity.
    if (side.capacitySlots < padded)
    {
        uint8_t* grown = allocate(padded * slotSize);
        if (side.sizeSlots > 0)
        {
            std::memcpy(grown, side.slots, side.sizeSlots * slotSize);
        }
        side.slots = grown;
        side.capacitySlots = padded;
    }

    /// Layout (OCA.cpp:729-731): sorted window ascending in [0, size), dummy
    /// gap in [size, padded - batch), batch right-aligned in [padded - batch,
    /// padded).
    for (uint64_t i = side.sizeSlots; i < padded - numBatchRows; ++i)
    {
        writeDummySlot(side.slots + (i * slotSize), slotSize);
    }
    for (uint64_t j = 0; j < numBatchRows; ++j)
    {
        uint8_t* slot = side.slots + ((padded - numBatchRows + j) * slotSize);
        const uint8_t* row = batchRows + (j * layout.rowSize);
        const SlotHeader header = makeHeader(row, layout, fkSide);
        std::memcpy(slot, &header, sizeof(SlotHeader));
        std::memcpy(slot + sizeof(SlotHeader), row, layout.rowSize);
    }

    /// Sort the power-of-two batch region descending (OCA.cpp:734-735). The
    /// region may include leading dummy slots from the gap; they sort to its
    /// front (dummies are greatest), preserving the bitonic shape
    /// [ascending | +inf plateau | descending].
    const uint64_t batchRegion = nextPowerOfTwo(numBatchRows);
    bitonicSort(side.slots, slotSize, padded - batchRegion, batchRegion, /*descending=*/true);

    /// Straight (non-crossover) bitonic merge over the whole array
    /// (OCA.cpp:746-753): the array is bitonic, so this sorts it ascending
    /// with all dummies at the tail.
    bitonicMerge(side.slots, slotSize, 0, padded, /*crossover=*/false, /*descending=*/false);

    side.sizeSlots += numBatchRows;
}

uint64_t mergedSlotSize(const SideLayout& pkLayout, const SideLayout& fkLayout)
{
    return sizeof(SlotHeader) + std::max(pkLayout.rowSize, fkLayout.rowSize);
}

uint64_t crossMergePaddedSlots(const uint64_t numPkSlots, const uint64_t numFkSlots)
{
    return nextPowerOfTwo(2 * std::max(numPkSlots, numFkSlots));
}

uint64_t crossMerge(
    const SortedSide& pkSide, const SideLayout& pkLayout, const SortedSide& fkSide, const SideLayout& fkLayout, uint8_t* scratch)
{
    const uint64_t slotSize = mergedSlotSize(pkLayout, fkLayout);
    const uint64_t pkSlotSize = pkLayout.slotSize();
    const uint64_t fkSlotSize = fkLayout.slotSize();
    const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);

    /// PK side ascending in [0, nPk), tagged tableId=0 (its headers carry no
    /// FLAG_FK already). Payload areas beyond the row are zeroed for
    /// deterministic slot content.
    for (uint64_t i = 0; i < pkSide.sizeSlots; ++i)
    {
        uint8_t* dst = scratch + (i * slotSize);
        std::memcpy(dst, pkSide.slots + (i * pkSlotSize), pkSlotSize);
        std::memset(dst + pkSlotSize, 0, slotSize - pkSlotSize);
    }
    /// Dummy gap (OCA.cpp copy():792-798).
    for (uint64_t i = pkSide.sizeSlots; i < padded - fkSide.sizeSlots; ++i)
    {
        writeDummySlot(scratch + (i * slotSize), slotSize);
    }
    /// FK side right-aligned descending: written in reverse — a fixed,
    /// data-independent permutation of the already-sorted side, equivalent to
    /// the reference's descending sort of the (sorted) batch region.
    for (uint64_t j = 0; j < fkSide.sizeSlots; ++j)
    {
        uint8_t* dst = scratch + ((padded - 1 - j) * slotSize);
        std::memcpy(dst, fkSide.slots + (j * fkSlotSize), fkSlotSize);
        std::memset(dst + fkSlotSize, 0, slotSize - fkSlotSize);
    }

    /// Straight bitonic merge over the bitonic array (OCA.cpp:878-885).
    bitonicMerge(scratch, slotSize, 0, padded, /*crossover=*/false, /*descending=*/false);

    return pkSide.sizeSlots + fkSide.sizeSlots;
}

uint64_t outputSlotSize(const SideLayout& pkLayout, const SideLayout& fkLayout)
{
    return sizeof(uint64_t) + pkLayout.rowSize + fkLayout.rowSize;
}

uint64_t sortedCrossCombine(
    const SortedSide& pkSide, const SideLayout& pkLayout, const SortedSide& fkSide, const SideLayout& fkLayout, uint8_t* scratch)
{
    const uint64_t slotSize = mergedSlotSize(pkLayout, fkLayout);
    const uint64_t pkSlotSize = pkLayout.slotSize();
    const uint64_t fkSlotSize = fkLayout.slotSize();
    const uint64_t total = pkSide.sizeSlots + fkSide.sizeSlots;
    const uint64_t padded = nextPowerOfTwo(total);

    /// Concatenate both sides tagged, recomputing key headers from the raw
    /// rows (the input slots may be timestamp-headered arrival logs); the
    /// arrangement does not matter — the full sort below orders everything.
    for (uint64_t i = 0; i < pkSide.sizeSlots; ++i)
    {
        uint8_t* dst = scratch + (i * slotSize);
        const uint8_t* row = pkSide.slots + (i * pkSlotSize) + sizeof(SlotHeader);
        const SlotHeader header = makeHeader(row, pkLayout, /*fkSide=*/false);
        std::memcpy(dst, &header, sizeof(SlotHeader));
        std::memcpy(dst + sizeof(SlotHeader), row, pkLayout.rowSize);
        std::memset(dst + pkSlotSize, 0, slotSize - pkSlotSize);
    }
    for (uint64_t j = 0; j < fkSide.sizeSlots; ++j)
    {
        uint8_t* dst = scratch + ((pkSide.sizeSlots + j) * slotSize);
        const uint8_t* row = fkSide.slots + (j * fkSlotSize) + sizeof(SlotHeader);
        const SlotHeader header = makeHeader(row, fkLayout, /*fkSide=*/true);
        std::memcpy(dst, &header, sizeof(SlotHeader));
        std::memcpy(dst + sizeof(SlotHeader), row, fkLayout.rowSize);
        std::memset(dst + fkSlotSize, 0, slotSize - fkSlotSize);
    }
    for (uint64_t i = total; i < padded; ++i)
    {
        writeDummySlot(scratch + (i * slotSize), slotSize);
    }

    /// FULL bitonic sort (OpaqueJoin.cpp:158-191) — the SORT family's cost.
    bitonicSort(scratch, slotSize, 0, padded, /*descending=*/false);

    return total;
}

void appendToArrivalLog(
    SortedSide& log,
    const SideLayout& layout,
    const bool fkSide,
    const uint8_t* batchRows,
    const uint64_t numBatchRows,
    const AllocateFn& allocate)
{
    if (numBatchRows == 0)
    {
        return;
    }
    const uint64_t slotSize = layout.slotSize();
    const SideLayout tsView = layout.timestampView();

    if (log.capacitySlots < log.sizeSlots + numBatchRows)
    {
        const uint64_t newCapacity = nextPowerOfTwo(2 * (log.sizeSlots + numBatchRows));
        uint8_t* grown = allocate(newCapacity * slotSize);
        if (log.sizeSlots > 0)
        {
            std::memcpy(grown, log.slots, log.sizeSlots * slotSize);
        }
        log.slots = grown;
        log.capacitySlots = newCapacity;
    }

    for (uint64_t j = 0; j < numBatchRows; ++j)
    {
        uint8_t* slot = log.slots + ((log.sizeSlots + j) * slotSize);
        const uint8_t* row = batchRows + (j * layout.rowSize);
        /// Timestamp headers are only needed by the L2 replay's event-time
        /// walk. The SORT_L3/L4 logs carry no timestamp field (their headers
        /// are recomputed from the rows during the full sort) and get a
        /// neutral header instead.
        const SlotHeader header = layout.tsType == DataType::Type::UNDEFINED
            ? SlotHeader{.sortKey = 0, .flags = fkSide ? SlotHeader::FLAG_FK : 0}
            : makeHeader(row, tsView, fkSide);
        std::memcpy(slot, &header, sizeof(SlotHeader));
        std::memcpy(slot + sizeof(SlotHeader), row, layout.rowSize);
    }
    log.sizeSlots += numBatchRows;
}

namespace
{

/// Pads an arrival log to power-of-two size with dummy slots and obliviously
/// sorts it by the (timestamp) headers. The log's capacity is a power of two
/// >= its size by construction (appendToArrivalLog growth), so the padded
/// range always fits.
void sortLogByTimestamp(SortedSide& log, const uint64_t slotSize)
{
    if (log.sizeSlots == 0)
    {
        return;
    }
    const uint64_t padded = nextPowerOfTwo(log.sizeSlots);
    for (uint64_t i = log.sizeSlots; i < padded; ++i)
    {
        writeDummySlot(log.slots + (i * slotSize), slotSize);
    }
    bitonicSort(log.slots, slotSize, 0, padded, /*descending=*/false);
}

/// Orders two (timestamp-headered) slots across the two logs: true if the FK
/// side's next tuple goes first. Mirrors the reference's strict
/// `next_is_R = ts_R < ts_S` (R = PK): the PK tuple only goes first if its
/// timestamp is strictly smaller, ties process the FK side. NULL timestamps
/// (keyIsNull) sort last via the rank comparison.
bool fkGoesFirst(const SlotHeader& pkHeader, const SlotHeader& fkHeader)
{
    const uint64_t pkRank = pkHeader.isKeyNull() ? 1 : 0;
    const uint64_t fkRank = fkHeader.isKeyNull() ? 1 : 0;
    if (pkRank != fkRank)
    {
        return fkRank < pkRank;
    }
    return pkHeader.sortKey >= fkHeader.sortKey;
}

}

ScanResult perTupleReplay(
    SortedSide& pkLog,
    const SideLayout& pkLayout,
    SortedSide& fkLog,
    const SideLayout& fkLayout,
    const bool fullSort,
    uint8_t* outSlots,
    const uint64_t outCapacitySlots,
    const AllocateFn& allocate)
{
    const uint64_t pkSlotSize = pkLayout.slotSize();
    const uint64_t fkSlotSize = fkLayout.slotSize();
    const uint64_t mergedSlot = mergedSlotSize(pkLayout, fkLayout);
    const uint64_t outSlot = outputSlotSize(pkLayout, fkLayout);
    const uint64_t numPk = pkLog.sizeSlots;
    const uint64_t numFk = fkLog.sizeSlots;

    /// Restore global event-time order (staging batches of concurrent workers
    /// may interleave; the sort is oblivious).
    sortLogByTimestamp(pkLog, pkSlotSize);
    sortLogByTimestamp(fkLog, fkSlotSize);

    /// Replay windows and reusable per-tuple scratches, sized for the final
    /// window cardinalities so no allocation happens inside the loop.
    SortedSide pkWindow;
    SortedSide fkWindow;
    const uint64_t maxSide = std::max<uint64_t>({numPk, numFk, 1});
    std::vector<uint8_t> mergedScratch(crossMergePaddedSlots(maxSide, maxSide) * mergedSlot);
    std::vector<uint8_t> tupleOutScratch(nextPowerOfTwo(maxSide + 1) * outSlot);
    std::vector<uint8_t> singletonSlot(std::max(pkSlotSize, fkSlotSize));

    ScanResult total;
    uint64_t outCount = 0;
    uint64_t pkIndex = 0;
    uint64_t fkIndex = 0;

    while (pkIndex < numPk || fkIndex < numFk)
    {
        /// Pick the next tuple in event-time order (timestamps are public).
        bool processFk = fkIndex < numFk;
        if (pkIndex < numPk && fkIndex < numFk)
        {
            SlotHeader pkTsHeader;
            SlotHeader fkTsHeader;
            std::memcpy(&pkTsHeader, pkLog.slots + (pkIndex * pkSlotSize), sizeof(SlotHeader));
            std::memcpy(&fkTsHeader, fkLog.slots + (fkIndex * fkSlotSize), sizeof(SlotHeader));
            processFk = fkGoesFirst(pkTsHeader, fkTsHeader);
        }

        const SideLayout& ownLayout = processFk ? fkLayout : pkLayout;
        SortedSide& ownWindow = processFk ? fkWindow : pkWindow;
        const uint8_t* row = (processFk ? fkLog.slots + (fkIndex * fkSlotSize) : pkLog.slots + (pkIndex * pkSlotSize)) + sizeof(SlotHeader);

        /// Append the tuple to its own window, then combine it with the
        /// opposite window and scan (reference: t1/t2 of the chosen branch).
        /// MERG_L2 maintains key-sorted windows (OAppend) and combines with
        /// the bitonic merge; SORT_L2 keeps raw logs and pays a full bitonic
        /// sort per tuple (OpaqueJoin with batch size 1).
        if (fullSort)
        {
            appendToArrivalLog(ownWindow, ownLayout, processFk, row, 1, allocate);
        }
        else
        {
            oAppend(ownWindow, ownLayout, processFk, row, 1, allocate);
        }

        const SlotHeader keyHeader = makeHeader(row, ownLayout, processFk);
        std::memcpy(singletonSlot.data(), &keyHeader, sizeof(SlotHeader));
        std::memcpy(singletonSlot.data() + sizeof(SlotHeader), row, ownLayout.rowSize);
        const SortedSide singleton{.slots = singletonSlot.data(), .sizeSlots = 1, .capacitySlots = 1};

        uint64_t numMerged = 0;
        if (fullSort)
        {
            numMerged = processFk ? sortedCrossCombine(pkWindow, pkLayout, singleton, fkLayout, mergedScratch.data())
                                  : sortedCrossCombine(singleton, pkLayout, fkWindow, fkLayout, mergedScratch.data());
        }
        else
        {
            numMerged = processFk ? crossMerge(pkWindow, pkLayout, singleton, fkLayout, mergedScratch.data())
                                  : crossMerge(singleton, pkLayout, fkWindow, fkLayout, mergedScratch.data());
        }

        const auto scanResult = obliviousScan(mergedScratch.data(), numMerged, pkLayout, fkLayout, tupleOutScratch.data());
        total.duplicatePkDetected |= scanResult.duplicatePkDetected;

        /// Per-tuple compaction (the L2 leakage: this tuple's degree).
        const uint64_t realSlots = trimDummies(tupleOutScratch.data(), numMerged, outSlot);
        /// More matches than FK tuples means a duplicate PK matched some FK
        /// twice — clamp the copy (the caller throws on the flag below).
        const uint64_t copySlots = std::min(realSlots, outCapacitySlots - outCount);
        total.duplicatePkDetected |= copySlots < realSlots;
        std::memcpy(outSlots + (outCount * outSlot), tupleOutScratch.data(), copySlots * outSlot);
        outCount += copySlots;
        total.realMatches += copySlots;

        if (processFk)
        {
            ++fkIndex;
        }
        else
        {
            ++pkIndex;
        }
    }

    /// The per-tuple merges only see the PK window plus one tuple, so a
    /// duplicate PK pair without any later FK arrival would go unnoticed by
    /// the scans. One adjacent-key pass over a key-sorted view of the PK
    /// window catches every duplicate deterministically. MERG_L2's window is
    /// already key-sorted; SORT_L2's is a raw log, so sort it once into the
    /// merged scratch first.
    const uint8_t* dupCheckSlots = pkWindow.slots;
    uint64_t dupCheckSlotSize = pkLayout.slotSize();
    if (fullSort && pkWindow.sizeSlots > 0)
    {
        const SortedSide emptyFkSide{};
        sortedCrossCombine(pkWindow, pkLayout, emptyFkSide, fkLayout, mergedScratch.data());
        dupCheckSlots = mergedScratch.data();
        dupCheckSlotSize = mergedSlot;
    }
    for (uint64_t i = 1; i < pkWindow.sizeSlots; ++i)
    {
        SlotHeader previous;
        SlotHeader current;
        std::memcpy(&previous, dupCheckSlots + ((i - 1) * dupCheckSlotSize), sizeof(SlotHeader));
        std::memcpy(&current, dupCheckSlots + (i * dupCheckSlotSize), sizeof(SlotHeader));
        total.duplicatePkDetected |= previous.isRealPk() & current.isRealPk() & (previous.sortKey == current.sortKey);
    }

    return total;
}

uint64_t trimDummies(uint8_t* outSlots, const uint64_t numSlots, const uint64_t outSlotSize)
{
    const uint64_t padded = nextPowerOfTwo(numSlots);

    /// Pad with dummy output slots (flags bit0 set, zeroed rows) so the
    /// power-of-two compaction applies; the padding slots are unmarked and end
    /// up in the trimmed tail like the scan's own dummies.
    for (uint64_t i = numSlots; i < padded; ++i)
    {
        uint8_t* slot = outSlots + (i * outSlotSize);
        constexpr uint64_t dummyFlags = 1;
        std::memcpy(slot, &dummyFlags, sizeof(dummyFlags));
        std::memset(slot + sizeof(uint64_t), 0, outSlotSize - sizeof(uint64_t));
    }

    /// Mark the real slots. The marks array is indexed like the slots and
    /// never permuted (see obliviousCompactPow2).
    std::vector<uint8_t> marks(padded, 0);
    uint64_t realSlots = 0;
    for (uint64_t i = 0; i < numSlots; ++i)
    {
        uint64_t flags = 0;
        std::memcpy(&flags, outSlots + (i * outSlotSize), sizeof(flags));
        const auto real = static_cast<uint8_t>((flags & 1ULL) == 0);
        marks[i] = real;
        realSlots += real;
    }

    obliviousCompactPow2(outSlots, outSlotSize, marks.data(), 0, padded, 0);
    return realSlots;
}

ScanResult obliviousScan(
    const uint8_t* merged, const uint64_t numMergedSlots, const SideLayout& pkLayout, const SideLayout& fkLayout, uint8_t* out)
{
    const uint64_t slotSize = mergedSlotSize(pkLayout, fkLayout);
    const uint64_t outSlot = outputSlotSize(pkLayout, fkLayout);

    /// Carried PK slot, initialized to a dummy (the reference's `inf`).
    std::vector<uint8_t> carried(slotSize);
    writeDummySlot(carried.data(), slotSize);
    /// All-zero sentinel rows for dummy output slots: every field decodes as a
    /// non-null zero value.
    const std::vector<uint8_t> zeroRow(std::max(pkLayout.rowSize, fkLayout.rowSize), 0);

    ScanResult result;
    bool duplicatePk = false;

    for (uint64_t i = 0; i < numMergedSlots; ++i)
    {
        const uint8_t* slot = merged + (i * slotSize);
        SlotHeader cur;
        std::memcpy(&cur, slot, sizeof(SlotHeader));
        SlotHeader carriedHeader;
        std::memcpy(&carriedHeader, carried.data(), sizeof(SlotHeader));

        const bool isRealPk = cur.isRealPk();
        const bool keysEqual = cur.sortKey == carriedHeader.sortKey;

        /// Two adjacent real PK slots with equal keys violate the FK-join
        /// precondition; accumulate branch-free, report after the walk.
        duplicatePk |= isRealPk & carriedHeader.isRealPk() & keysEqual;

        /// carried = isRealPk ? cur : carried (OpaqueJoin.cpp:27).
        oSelectBytes(carried.data(), slot, carried.data(), slotSize, isRealPk);
        std::memcpy(&carriedHeader, carried.data(), sizeof(SlotHeader));

        /// Emit a real joined pair iff the slot is a real FK tuple whose key
        /// equals the carried real PK's key (OpaqueJoin.cpp:28).
        const bool emitReal = cur.isFkSide() & !cur.isDummy() & !cur.isKeyNull() & carriedHeader.isRealPk()
            & (cur.sortKey == carriedHeader.sortKey);
        result.realMatches += static_cast<uint64_t>(emitReal);

        uint8_t* outPtr = out + (i * outSlot);
        const uint64_t flags = static_cast<uint64_t>(!emitReal);
        std::memcpy(outPtr, &flags, sizeof(flags));
        oSelectBytes(outPtr + sizeof(uint64_t), carried.data() + sizeof(SlotHeader), zeroRow.data(), pkLayout.rowSize, emitReal);
        oSelectBytes(
            outPtr + sizeof(uint64_t) + pkLayout.rowSize, slot + sizeof(SlotHeader), zeroRow.data(), fkLayout.rowSize, emitReal);
    }

    result.duplicatePkDetected = duplicatePk;
    return result;
}

}
