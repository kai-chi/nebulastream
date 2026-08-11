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

#include <Join/FKMergJoin/Oblivious/NfkJoin.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/FKMergJoin/Oblivious/ObliviousPrimitives.hpp>
#include <ErrorHandling.hpp>

namespace NES::FKMerg
{

namespace
{

NfkSlotHeader readNfkHeader(const uint8_t* slot)
{
    NfkSlotHeader header;
    std::memcpy(&header, slot, sizeof(header));
    return header;
}

void writeNfkHeader(uint8_t* slot, const NfkSlotHeader& header)
{
    std::memcpy(slot, &header, sizeof(header));
}

void writeEmptyNfkSlot(uint8_t* slot, const uint64_t slotSize)
{
    constexpr NfkSlotHeader empty{
        .sortKey = UINT64_MAX, .flags = NfkSlotHeader::FLAG_EMPTY, .blockHeight = 0, .blockWidth = 0, .index = 0, .t1index = 0};
    writeNfkHeader(slot, empty);
    std::memset(slot + sizeof(NfkSlotHeader), 0, slotSize - sizeof(NfkSlotHeader));
}

/// Group identity: two non-empty slots belong to the same key group iff their
/// normalized keys AND null flags agree (NULL keys cluster separately and are
/// suppressed from contributing below, per SQL semantics).
bool sameGroup(const NfkSlotHeader& a, const NfkSlotHeader& b)
{
    return a.sortKey == b.sortKey && a.isKeyNull() == b.isKeyNull();
}

/// (key, tableId), empties last — the reference's attr_comp.
bool attrLess(const uint8_t* a, const uint8_t* b)
{
    const auto ha = readNfkHeader(a);
    const auto hb = readNfkHeader(b);
    const uint64_t rankA = (ha.isEmpty() ? 2U : 0U) | (ha.isKeyNull() ? 1U : 0U);
    const uint64_t rankB = (hb.isEmpty() ? 2U : 0U) | (hb.isKeyNull() ? 1U : 0U);
    if (rankA != rankB)
    {
        return rankA < rankB;
    }
    if (ha.sortKey != hb.sortKey)
    {
        return ha.sortKey < hb.sortKey;
    }
    return static_cast<uint64_t>(ha.isRight()) < static_cast<uint64_t>(hb.isRight());
}

/// (tableId, key), empties last — the reference's tid_comp (payload tie-break
/// dropped; any within-group order yields the same output set).
bool tidLess(const uint8_t* a, const uint8_t* b)
{
    const auto ha = readNfkHeader(a);
    const auto hb = readNfkHeader(b);
    if (ha.isEmpty() != hb.isEmpty())
    {
        return hb.isEmpty();
    }
    if (ha.isRight() != hb.isRight())
    {
        return static_cast<uint64_t>(ha.isRight()) < static_cast<uint64_t>(hb.isRight());
    }
    if (ha.isKeyNull() != hb.isKeyNull())
    {
        return hb.isKeyNull();
    }
    return ha.sortKey < hb.sortKey;
}

/// (key, t1index), empties last — the reference's t1_comp, aligning the
/// expanded right side.
bool t1Less(const uint8_t* a, const uint8_t* b)
{
    const auto ha = readNfkHeader(a);
    const auto hb = readNfkHeader(b);
    if (ha.isEmpty() != hb.isEmpty())
    {
        return hb.isEmpty();
    }
    if (ha.sortKey != hb.sortKey)
    {
        return ha.sortKey < hb.sortKey;
    }
    return ha.t1index < hb.t1index;
}

/// Target-index order for the distribute pre-sort, empties last — the
/// reference's entry_ind_func_comp.
bool indexLess(const uint8_t* a, const uint8_t* b)
{
    const auto ha = readNfkHeader(a);
    const auto hb = readNfkHeader(b);
    if (ha.isEmpty() != hb.isEmpty())
    {
        return hb.isEmpty();
    }
    return ha.index < hb.index;
}

/// One side's working array during expansion.
struct NfkSide
{
    uint8_t* slots = nullptr;
    uint64_t size = 0; /// logical entries
    uint64_t capacitySlots = 0;
};

/// Concatenates both logs into `combined` as NFK slots (fresh flags per the
/// selector), key-sorts it, and runs the reference's marking + dimension
/// scans (OCAKras write_block_sizes). Returns the output cardinality.
uint64_t buildSortAndMeasure(
    const SortedSide& leftLog,
    const SideLayout& leftLayout,
    const SortedSide& rightLog,
    const SideLayout& rightLayout,
    const NfkFreshSelector& fresh,
    uint8_t* combined,
    const uint64_t slotSize)
{
    const uint64_t total = leftLog.sizeSlots + rightLog.sizeSlots;
    const uint64_t padded = nextPowerOfTwo(std::max<uint64_t>(total, 1));

    /// Build the tagged, fresh-marked slots from the raw log rows.
    for (uint64_t side = 0; side < 2; ++side)
    {
        const bool isRight = side == 1;
        const SortedSide& log = isRight ? rightLog : leftLog;
        const SideLayout& layout = isRight ? rightLayout : leftLayout;
        const uint64_t logSlotSize = layout.slotSize();
        const uint64_t base = isRight ? leftLog.sizeSlots : 0;
        for (uint64_t i = 0; i < log.sizeSlots; ++i)
        {
            uint8_t* dst = combined + ((base + i) * slotSize);
            const uint8_t* row = log.slots + (i * logSlotSize) + sizeof(SlotHeader);
            bool isNull = false;
            const uint64_t sortKey = normalizeKey(row, layout, isNull);
            const bool isFresh = fresh.allFresh || (fresh.freshSideIsRight == isRight && fresh.freshIndex == i);
            NfkSlotHeader header{
                .sortKey = sortKey,
                .flags = (isRight ? NfkSlotHeader::FLAG_RIGHT : 0) | (isNull ? NfkSlotHeader::FLAG_KEY_NULL : 0)
                    | (isFresh ? NfkSlotHeader::FLAG_FRESH : 0),
                .blockHeight = 0,
                .blockWidth = 0,
                .index = 0,
                .t1index = 0};
            writeNfkHeader(dst, header);
            std::memcpy(dst + sizeof(NfkSlotHeader), row, layout.rowSize);
            std::memset(dst + sizeof(NfkSlotHeader) + layout.rowSize, 0, slotSize - sizeof(NfkSlotHeader) - layout.rowSize);
        }
    }
    for (uint64_t i = total; i < padded; ++i)
    {
        writeEmptyNfkSlot(combined + (i * slotSize), slotSize);
    }

    /// Sort by (key, tableId).
    bitonicSort(combined, slotSize, 0, padded, /*descending=*/false, attrLess);

    /// The reference's write_block_sizes: two marking passes (contributes =
    /// this row or any same-key row of the other side is fresh), then the
    /// stock Krastnikov dimension scans gated on that flag. All passes are
    /// linear with a fixed access pattern.
    auto at = [&](const uint64_t i) { return readNfkHeader(combined + (i * slotSize)); };
    auto put = [&](const uint64_t i, const NfkSlotHeader& header) { writeNfkHeader(combined + (i * slotSize), header); };

    /// Forward marking pass.
    {
        NfkSlotHeader last{};
        bool haveLast = false;
        bool newInKey = false;
        for (uint64_t i = 0; i < total; ++i)
        {
            auto e = at(i);
            const bool sameAttr = haveLast && sameGroup(e, last);
            bool contributes = false;
            if (!e.isRight())
            {
                contributes = e.isFresh();
                newInKey = sameAttr ? (newInKey || e.isFresh()) : e.isFresh();
            }
            else
            {
                contributes = sameAttr ? (newInKey || e.isFresh()) : e.isFresh();
                if (!sameAttr)
                {
                    newInKey = false;
                }
            }
            e.flags = contributes ? (e.flags | NfkSlotHeader::FLAG_CONTRIBUTES) : (e.flags & ~NfkSlotHeader::FLAG_CONTRIBUTES);
            put(i, e);
            last = e;
            haveLast = true;
        }
    }
    /// Backward marking pass (mirror: OR fresh right rows into left rows),
    /// then suppress NULL-key contributions.
    {
        NfkSlotHeader last{};
        bool haveLast = false;
        bool newInKey = false;
        for (uint64_t i = total; i-- > 0;)
        {
            auto e = at(i);
            const bool sameAttr = haveLast && sameGroup(e, last);
            if (e.isRight())
            {
                newInKey = sameAttr ? (newInKey || e.isFresh()) : e.isFresh();
            }
            else if (sameAttr && newInKey)
            {
                e.flags |= NfkSlotHeader::FLAG_CONTRIBUTES;
            }
            if (e.isKeyNull())
            {
                e.flags &= ~NfkSlotHeader::FLAG_CONTRIBUTES;
            }
            put(i, e);
            last = e;
            haveLast = true;
        }
    }
    /// Forward: heights onto right rows (# contributing left rows per key).
    {
        NfkSlotHeader last{};
        bool haveLast = false;
        int64_t height = 0;
        for (uint64_t i = 0; i < total; ++i)
        {
            auto e = at(i);
            const bool sameAttr = haveLast && sameGroup(e, last);
            if (!e.isRight())
            {
                height = sameAttr ? height + static_cast<int64_t>(e.contributes()) : static_cast<int64_t>(e.contributes());
            }
            else
            {
                if (!sameAttr)
                {
                    height = 0;
                }
                e.blockHeight = height;
            }
            put(i, e);
            last = e;
            haveLast = true;
        }
    }
    /// Backward: widths (# contributing right rows) and heights onto left rows.
    {
        NfkSlotHeader last{};
        bool haveLast = false;
        int64_t width = 0;
        int64_t height = 0;
        for (uint64_t i = total; i-- > 0;)
        {
            auto e = at(i);
            const bool sameAttr = haveLast && sameGroup(e, last);
            if (e.isRight())
            {
                width = sameAttr ? width + static_cast<int64_t>(e.contributes()) : static_cast<int64_t>(e.contributes());
                height = e.blockHeight;
            }
            else if (sameAttr)
            {
                e.blockWidth = width;
                e.blockHeight = height;
            }
            else
            {
                width = 0;
                height = 0;
                e.blockWidth = 0;
                e.blockHeight = 0;
            }
            put(i, e);
            last = e;
            haveLast = true;
        }
    }
    /// Forward: widths onto right rows + the output cardinality.
    uint64_t outputSize = 0;
    {
        NfkSlotHeader last{};
        bool haveLast = false;
        int64_t width = 0;
        for (uint64_t i = 0; i < total; ++i)
        {
            auto e = at(i);
            const bool sameAttr = haveLast && sameGroup(e, last);
            if (!e.isRight())
            {
                width = e.blockWidth;
                if (!sameAttr)
                {
                    outputSize += static_cast<uint64_t>(e.blockHeight * e.blockWidth);
                }
            }
            else
            {
                if (!sameAttr)
                {
                    width = 0;
                }
                e.blockWidth = width;
            }
            put(i, e);
            last = e;
            haveLast = true;
        }
    }
    return outputSize;
}

/// The reference's oblivious_distribute: routes every entry to its target
/// index via a reverse-butterfly network whose access pattern depends only on
/// the array length.
void obliviousDistribute(NfkSide& side, const uint64_t slotSize, const uint64_t targetSize)
{
    /// Pre-sort by target index (empties last); pad to pow2 for the network.
    const uint64_t sortSpan = nextPowerOfTwo(std::max<uint64_t>(std::max(side.size, targetSize), 1));
    INVARIANT(sortSpan <= side.capacitySlots, "NFK distribute scratch too small: {} > {}", sortSpan, side.capacitySlots);
    for (uint64_t i = side.size; i < sortSpan; ++i)
    {
        writeEmptyNfkSlot(side.slots + (i * slotSize), slotSize);
    }
    bitonicSort(side.slots, slotSize, 0, sortSpan, /*descending=*/false, indexLess);

    const auto m = static_cast<int64_t>(targetSize);
    for (int64_t j = static_cast<int64_t>(nextPowerOfTwo(std::max<int64_t>(m, 1))) / 2; j >= 1; j /= 2)
    {
        for (int64_t i = m - j - 1; i >= 0; --i)
        {
            const auto header = readNfkHeader(side.slots + (i * static_cast<int64_t>(slotSize)));
            const int64_t dest = header.isEmpty() ? -1 : header.index;
            const bool cond = dest >= i + j;
            oMemSwap(side.slots + (i * static_cast<int64_t>(slotSize)), side.slots + ((i + j) * static_cast<int64_t>(slotSize)), slotSize, cond);
        }
    }
    side.size = targetSize;
}

/// The reference's oblivious_expand: assigns prefix-sum target indices by the
/// entry's weight (block width for the left side, height for the right),
/// distributes, then fills the gaps by duplicating the preceding entry while
/// computing the alignment indices.
void obliviousExpand(NfkSide& side, const uint64_t slotSize, const bool weightIsWidth, const uint64_t targetSize)
{
    int64_t csum = 0;
    for (uint64_t i = 0; i < side.size; ++i)
    {
        auto e = readNfkHeader(side.slots + (i * slotSize));
        const int64_t weight = (e.isEmpty() || !e.contributes()) ? 0 : (weightIsWidth ? e.blockWidth : e.blockHeight);
        if (weight == 0)
        {
            e.flags |= NfkSlotHeader::FLAG_EMPTY;
        }
        else
        {
            e.index = csum;
        }
        csum += weight;
        writeNfkHeader(side.slots + (i * slotSize), e);
    }
    INVARIANT(static_cast<uint64_t>(csum) == targetSize, "NFK expansion size mismatch: {} != {}", csum, targetSize);

    obliviousDistribute(side, slotSize, targetSize);

    /// Duplication pass: every EMPTY gap slot becomes a copy of the preceding
    /// real entry; index/t1index enumerate the copies and their transposed
    /// alignment position (reference oblivious_expand tail).
    std::vector<uint8_t> lastSlot(slotSize, 0);
    int64_t duplOff = 0;
    int64_t blockOff = 0;
    for (uint64_t i = 0; i < targetSize; ++i)
    {
        uint8_t* slot = side.slots + (i * slotSize);
        auto e = readNfkHeader(slot);
        if (!e.isEmpty())
        {
            const auto lastHeader = readNfkHeader(lastSlot.data());
            if (i != 0 && !sameGroup(e, lastHeader))
            {
                blockOff = 0;
            }
            std::memcpy(lastSlot.data(), slot, slotSize);
            duplOff = 0;
        }
        else
        {
            std::memcpy(slot, lastSlot.data(), slotSize);
            e = readNfkHeader(slot);
        }
        e.index += duplOff;
        e.t1index = (blockOff / e.blockHeight) + ((blockOff % e.blockHeight) * e.blockWidth);
        writeNfkHeader(slot, e);
        ++duplOff;
        ++blockOff;
    }
    /// NOTE: unlike the reference (OCAKras.h oblivious_expand), the
    /// duplication pass above materializes the copies INTO the array. The
    /// reference redirects its writes into a stack-local `last` for gap slots
    /// and never writes them back, so its expanded tables keep EMPTY gaps with
    /// stale alignment indices — its results were only ever counted, never
    /// validated (the same landmine family as OCA::scan).
}

}

uint64_t nfkSlotSize(const SideLayout& leftLayout, const SideLayout& rightLayout)
{
    return sizeof(NfkSlotHeader) + std::max(leftLayout.rowSize, rightLayout.rowSize);
}

uint64_t nfkJoinSize(
    const SortedSide& leftLog,
    const SideLayout& leftLayout,
    const SortedSide& rightLog,
    const SideLayout& rightLayout,
    const NfkFreshSelector& fresh,
    const AllocateFn& allocateScratch)
{
    const uint64_t total = leftLog.sizeSlots + rightLog.sizeSlots;
    if (total == 0)
    {
        return 0;
    }
    const uint64_t slotSize = nfkSlotSize(leftLayout, rightLayout);
    uint8_t* combined = allocateScratch(nextPowerOfTwo(total) * slotSize);
    return buildSortAndMeasure(leftLog, leftLayout, rightLog, rightLayout, fresh, combined, slotSize);
}

uint64_t nfkJoin(
    const SortedSide& leftLog,
    const SideLayout& leftLayout,
    const SortedSide& rightLog,
    const SideLayout& rightLayout,
    const NfkFreshSelector& fresh,
    uint8_t* outSlots,
    /// only checked via INVARIANT, which release builds compile away
    [[maybe_unused]] const uint64_t outCapacitySlots,
    const AllocateFn& allocateScratch)
{
    const uint64_t numLeft = leftLog.sizeSlots;
    const uint64_t numRight = rightLog.sizeSlots;
    const uint64_t total = numLeft + numRight;
    if (total == 0)
    {
        return 0;
    }
    const uint64_t slotSize = nfkSlotSize(leftLayout, rightLayout);

    uint8_t* combined = allocateScratch(nextPowerOfTwo(total) * slotSize);
    const uint64_t outputSize = buildSortAndMeasure(leftLog, leftLayout, rightLog, rightLayout, fresh, combined, slotSize);
    INVARIANT(outputSize <= outCapacitySlots, "NFK join output {} exceeds the pre-sized capacity {}", outputSize, outCapacitySlots);
    if (outputSize == 0)
    {
        return 0;
    }

    /// Re-sort by (tableId, key) and split into the two sides
    /// (OCAKras::join_windows).
    const uint64_t paddedTotal = nextPowerOfTwo(total);
    bitonicSort(combined, slotSize, 0, paddedTotal, /*descending=*/false, tidLess);

    NfkSide t0;
    NfkSide t1;
    t0.capacitySlots = nextPowerOfTwo(std::max<uint64_t>(std::max(numLeft, outputSize), 1));
    t1.capacitySlots = nextPowerOfTwo(std::max<uint64_t>(std::max(numRight, outputSize), 1));
    t0.slots = allocateScratch(t0.capacitySlots * slotSize);
    t1.slots = allocateScratch(t1.capacitySlots * slotSize);
    std::memcpy(t0.slots, combined, numLeft * slotSize);
    std::memcpy(t1.slots, combined + (numLeft * slotSize), numRight * slotSize);
    t0.size = numLeft;
    t1.size = numRight;

    /// Expand: left rows by their block width, right rows by their height.
    obliviousExpand(t0, slotSize, /*weightIsWidth=*/true, outputSize);
    obliviousExpand(t1, slotSize, /*weightIsWidth=*/false, outputSize);

    /// Align the right side by (key, t1index).
    const uint64_t paddedOut = nextPowerOfTwo(outputSize);
    for (uint64_t i = outputSize; i < paddedOut; ++i)
    {
        writeEmptyNfkSlot(t1.slots + (i * slotSize), slotSize);
    }
    INVARIANT(paddedOut <= t1.capacitySlots, "NFK t1 alignment scratch too small");
    bitonicSort(t1.slots, slotSize, 0, paddedOut, /*descending=*/false, t1Less);

    /// Emit aligned pairs as output slots: [flags=0][rightRow][leftRow]
    /// (matching the probe's PK-side-right region convention).
    const uint64_t outSlotBytes = sizeof(uint64_t) + rightLayout.rowSize + leftLayout.rowSize;
    for (uint64_t i = 0; i < outputSize; ++i)
    {
        uint8_t* out = outSlots + (i * outSlotBytes);
        constexpr uint64_t realFlags = 0;
        std::memcpy(out, &realFlags, sizeof(realFlags));
        std::memcpy(out + sizeof(uint64_t), t1.slots + (i * slotSize) + sizeof(NfkSlotHeader), rightLayout.rowSize);
        std::memcpy(out + sizeof(uint64_t) + rightLayout.rowSize, t0.slots + (i * slotSize) + sizeof(NfkSlotHeader), leftLayout.rowSize);
    }
    return outputSize;
}

uint64_t nfkPerTupleReplay(
    SortedSide& leftLog,
    const SideLayout& leftLayout,
    SortedSide& rightLog,
    const SideLayout& rightLayout,
    uint8_t* outSlots,
    const uint64_t outCapacitySlots,
    const AllocateFn& allocateScratch)
{
    const uint64_t leftSlotSize = leftLayout.slotSize();
    const uint64_t rightSlotSize = rightLayout.slotSize();

    /// Restore global event-time order (logs are SlotHeader-headered by the
    /// normalized event timestamp; pad + sort as in the FK replay).
    auto sortLog = [](SortedSide& log, const uint64_t slotSize)
    {
        if (log.sizeSlots == 0)
        {
            return;
        }
        const uint64_t padded = nextPowerOfTwo(log.sizeSlots);
        for (uint64_t i = log.sizeSlots; i < padded; ++i)
        {
            constexpr SlotHeader dummy{.sortKey = UINT64_MAX, .flags = SlotHeader::FLAG_DUMMY};
            std::memcpy(log.slots + (i * slotSize), &dummy, sizeof(dummy));
            std::memset(log.slots + (i * slotSize) + sizeof(SlotHeader), 0, slotSize - sizeof(SlotHeader));
        }
        bitonicSort(log.slots, slotSize, 0, padded, /*descending=*/false);
    };
    sortLog(leftLog, leftSlotSize);
    sortLog(rightLog, rightSlotSize);

    /// Per-tuple joins reuse three fixed scratch buffers (combined, t0, t1)
    /// instead of allocating per call — every bound below is monotone in the
    /// final log sizes, so one allocation covers the whole replay. Sized to
    /// the largest of: the combined padded table, and the per-side expansion
    /// scratch (per-call output <= the opposite window size).
    const uint64_t slotSize = nfkSlotSize(leftLayout, rightLayout);
    const uint64_t maxSide = std::max<uint64_t>({leftLog.sizeSlots, rightLog.sizeSlots, 1});
    const uint64_t scratchSlots = std::max(nextPowerOfTwo(leftLog.sizeSlots + rightLog.sizeSlots), nextPowerOfTwo(maxSide));
    std::array<uint8_t*, 3> scratches{};
    for (auto& scratch : scratches)
    {
        scratch = allocateScratch(scratchSlots * slotSize);
    }
    uint64_t nextScratch = 0;
    const AllocateFn reusedScratch = [&]([[maybe_unused]] const uint64_t bytes) -> uint8_t*
    {
        INVARIANT(bytes <= scratchSlots * slotSize, "NFK replay scratch bound violated: {} > {}", bytes, scratchSlots * slotSize);
        uint8_t* buffer = scratches[nextScratch % scratches.size()];
        ++nextScratch;
        return buffer;
    };

    uint64_t outCount = 0;
    uint64_t leftIndex = 0;
    uint64_t rightIndex = 0;
    const uint64_t outSlotBytes = sizeof(uint64_t) + rightLayout.rowSize + leftLayout.rowSize;

    while (leftIndex < leftLog.sizeSlots || rightIndex < rightLog.sizeSlots)
    {
        /// Pick the next tuple in event-time order (timestamps are public);
        /// ties process the right side first.
        bool processRight = rightIndex < rightLog.sizeSlots;
        if (leftIndex < leftLog.sizeSlots && rightIndex < rightLog.sizeSlots)
        {
            SlotHeader leftTs;
            SlotHeader rightTs;
            std::memcpy(&leftTs, leftLog.slots + (leftIndex * leftSlotSize), sizeof(SlotHeader));
            std::memcpy(&rightTs, rightLog.slots + (rightIndex * rightSlotSize), sizeof(SlotHeader));
            const uint64_t rankLeft = leftTs.isKeyNull() ? 1 : 0;
            const uint64_t rankRight = rightTs.isKeyNull() ? 1 : 0;
            processRight = rankLeft != rankRight ? rankRight < rankLeft : leftTs.sortKey >= rightTs.sortKey;
        }

        if (processRight)
        {
            ++rightIndex;
        }
        else
        {
            ++leftIndex;
        }

        /// Prefix views over the ts-sorted logs = window contents at this
        /// tuple's arrival; the tuple itself is the last entry of its side.
        const SortedSide leftView{.slots = leftLog.slots, .sizeSlots = leftIndex, .capacitySlots = leftIndex};
        const SortedSide rightView{.slots = rightLog.slots, .sizeSlots = rightIndex, .capacitySlots = rightIndex};
        const NfkFreshSelector fresh{
            .allFresh = false,
            .freshSideIsRight = processRight,
            .freshIndex = (processRight ? rightIndex : leftIndex) - 1};

        nextScratch = 0;
        const uint64_t emitted = nfkJoin(
            leftView,
            leftLayout,
            rightView,
            rightLayout,
            fresh,
            outSlots + (outCount * outSlotBytes),
            outCapacitySlots - outCount,
            reusedScratch);
        outCount += emitted;
    }
    return outCount;
}

}
