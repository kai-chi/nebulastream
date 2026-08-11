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
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/FKMergJoin/Oblivious/ObliviousPrimitives.hpp>

/// NFK-JOIN-L2/L3: the non-foreign-key (generic, duplicates on both sides)
/// oblivious stream join, ported from tee-bench-obliv-stream's OCAKras — a
/// streaming adaptation of Krastnikov et al.'s static oblivious binary join.
///
/// Pipeline (reference OCAKras::join_windows): concatenate both windows,
/// obliviously sort by (key, tableId), compute per-key block dimensions with
/// linear scans (marking only entries whose key group touches a "fresh"
/// tuple, which keeps the per-batch variant incremental), re-sort by
/// (tableId, key), split into the two sides, obliviously expand each side to
/// the output size (prefix-sum target indices + a reverse-butterfly
/// distribute network + a duplication pass), align the second side by its
/// transposed block index, and read off the aligned pairs. Output volume is
/// the true join cardinality — the L3 leakage; there is no L4 variant, the
/// non-FK worst case being the Cartesian product.
namespace NES::FKMerg
{

/// Working slot header of the NFK join (extends the SlotHeader idea with
/// Krastnikov's per-entry bookkeeping). Every slot is
/// [NfkSlotHeader | row bytes], rows padded to max(leftRow, rightRow).
struct NfkSlotHeader
{
    uint64_t sortKey;
    uint64_t flags;
    /// Krastnikov block bookkeeping: for a key group with h contributing
    /// left rows and w contributing right rows, every pair (i, j) is emitted.
    int64_t blockHeight;
    int64_t blockWidth;
    int64_t index;
    int64_t t1index;

    static constexpr uint64_t FLAG_EMPTY = 1ULL << 0;
    static constexpr uint64_t FLAG_RIGHT = 1ULL << 1; /// tableId: 0 = left, 1 = right
    static constexpr uint64_t FLAG_KEY_NULL = 1ULL << 2;
    static constexpr uint64_t FLAG_FRESH = 1ULL << 3;
    static constexpr uint64_t FLAG_CONTRIBUTES = 1ULL << 4;

    [[nodiscard]] bool isEmpty() const { return (flags & FLAG_EMPTY) != 0; }
    [[nodiscard]] bool isRight() const { return (flags & FLAG_RIGHT) != 0; }
    [[nodiscard]] bool isKeyNull() const { return (flags & FLAG_KEY_NULL) != 0; }
    [[nodiscard]] bool isFresh() const { return (flags & FLAG_FRESH) != 0; }
    [[nodiscard]] bool contributes() const { return (flags & FLAG_CONTRIBUTES) != 0; }
};

static_assert(sizeof(NfkSlotHeader) == 48, "NfkSlotHeader layout is part of the slot format");

/// Which entries count as fresh for the incremental marking: ALL (the L3
/// whole-window join), or exactly one entry of one side (the L2 per-tuple
/// replay, emitting only that tuple's matches).
struct NfkFreshSelector
{
    bool allFresh = true;
    bool freshSideIsRight = false;
    uint64_t freshIndex = 0;
};

/// Byte size of one NFK working slot for the given side layouts.
uint64_t nfkSlotSize(const SideLayout& leftLayout, const SideLayout& rightLayout);

/// Computes only the output cardinality of nfkJoin for the given logs and
/// fresh selector (concatenate + sort + dimension scans, no expansion).
/// Callers use it to pre-size output areas.
uint64_t nfkJoinSize(
    const SortedSide& leftLog,
    const SideLayout& leftLayout,
    const SortedSide& rightLog,
    const SideLayout& rightLayout,
    const NfkFreshSelector& fresh,
    const AllocateFn& allocateScratch);

/// Runs the full oblivious non-FK join over the two arrival logs and writes
/// one output slot per result pair into `outSlots` (layout identical to the
/// FK family's: [uint64 flags=0][rightRow][leftRow], matching the probe's
/// PK-side-right region convention; see FKMerg::outputSlotSize with pk=right).
/// Returns the number of emitted pairs (must fit outCapacitySlots; the caller
/// pre-sizes via nfkJoinSize). NULL keys never match. Scratch memory comes
/// from `allocateScratch` (lifetime managed by the caller).
uint64_t nfkJoin(
    const SortedSide& leftLog,
    const SideLayout& leftLayout,
    const SortedSide& rightLog,
    const SideLayout& rightLayout,
    const NfkFreshSelector& fresh,
    uint8_t* outSlots,
    uint64_t outCapacitySlots,
    const AllocateFn& allocateScratch);

/// NFK-JOIN-L2 (reference: OCAKras with batch size 1) replayed at window
/// close, consistent with the FK families' L2 adaptation: walks the tuples of
/// both logs in global event-time order (ties process the right side first)
/// and per tuple runs the full oblivious join with only that tuple fresh —
/// the incremental marking makes it emit exactly the new tuple's matches
/// (leaking its degree). Appends all matches to `outSlots`; the caller
/// pre-sizes it with nfkJoinSize over the complete logs (all fresh), which
/// equals the total across the replay. Returns the total emitted pairs.
uint64_t nfkPerTupleReplay(
    SortedSide& leftLog,
    const SideLayout& leftLayout,
    SortedSide& rightLog,
    const SideLayout& rightLayout,
    uint8_t* outSlots,
    uint64_t outCapacitySlots,
    const AllocateFn& allocateScratch);

}
