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

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>

namespace NES
{

class FKMergJoinSlice;

/// Content of the small per-(side, worker) control buffer that the slice-store
/// extractor hands to the build operator. The build's nautilus::invoke proxy
/// reads it to reach the slice natively (the extractor contract only allows
/// returning a TupleBuffer*).
struct FKMergStagingRef
{
    FKMergJoinSlice* slice;
    uint64_t side; /// 0 = left, 1 = right
    uint64_t workerIndex;
};

struct CreateNewFKMergSliceArgs final : CreateNewSlicesArguments
{
    CreateNewFKMergSliceArgs(
        AbstractBufferProvider& bufferProvider,
        FKMerg::SideLayout leftLayout,
        FKMerg::SideLayout rightLayout,
        const JoinBuildSideType pkSide,
        const uint64_t stagingBatchSize,
        const FKMerg::FKMergVariant variant)
        : bufferProvider(&bufferProvider)
        , leftLayout(leftLayout)
        , rightLayout(rightLayout)
        , pkSide(pkSide)
        , stagingBatchSize(stagingBatchSize)
        , variant(variant)
    {
    }

    ~CreateNewFKMergSliceArgs() override = default;

    AbstractBufferProvider* bufferProvider;
    FKMerg::SideLayout leftLayout;
    FKMerg::SideLayout rightLayout;
    JoinBuildSideType pkSide;
    uint64_t stagingBatchSize;
    FKMerg::FKMergVariant variant;
};

/// A slice of the FK-MERG-L4 join. Each side holds per-worker staging areas
/// (raw rows, appended lock-free by the owning worker) and one shared sorted
/// slot array maintained by OAppend. At trigger time the probe merges the two
/// sorted sides and runs the dummy-emitting scan (see FKMergAlgorithm.hpp).
///
/// Locking: staging appends are worker-local. Everything that touches the
/// shared sorted arrays or the scan result — flushes, growth, mergeAndScan —
/// runs under `stateMutex`, so correctness holds for any worker-thread count
/// (the algorithm itself is single-threaded by design).
class FKMergJoinSlice final : public Slice
{
public:
    FKMergJoinSlice(
        AbstractBufferProvider& bufferProvider,
        SliceStart sliceStart,
        SliceEnd sliceEnd,
        uint64_t numberOfWorkerThreads,
        const FKMerg::SideLayout& leftLayout,
        const FKMerg::SideLayout& rightLayout,
        JoinBuildSideType pkSide,
        uint64_t stagingBatchSize,
        FKMerg::FKMergVariant variant);

    /// The control buffer the slice-store extractor returns for a build of the
    /// given side and worker thread.
    [[nodiscard]] const TupleBuffer* getControlBufferRef(WorkerThreadId workerThreadId, JoinBuildSideType side) const;

    /// Hands out the staging slot for the next row of `side`, flushing the
    /// worker's staging batch into the sorted side first if it is full. Called
    /// from the build proxy of the worker that owns `workerIndex` only.
    [[nodiscard]] int8_t* acquireStagingSlot(uint64_t side, uint64_t workerIndex);

    /// Flushes all residual staging batches of both sides. Called at trigger
    /// time before the probe task is emitted.
    void flushAllStaging();

    /// Produces the window's join output (idempotent; subsequent calls return
    /// the cached result). L4: crossMerge + obliviousScan, dummies included.
    /// L3: additionally trims the dummies obliviously so only real matches
    /// remain. L2: replays the arrival logs tuple-by-tuple in event-time order
    /// with per-tuple compaction (perTupleReplay). Returns a pointer to the
    /// output area: a uint64 slot count followed by that many output slots
    /// (layout per FKMerg::outputSlotSize). Throws FKJoinPreconditionViolated
    /// if a window contains two PK tuples with the same key.
    [[nodiscard]] const int8_t* mergeAndScan();

    [[nodiscard]] uint64_t getNumberOfTuplesLeft() const { return sides[0].sorted.sizeSlots + stagedCount(0); }
    [[nodiscard]] uint64_t getNumberOfTuplesRight() const { return sides[1].sorted.sizeSlots + stagedCount(1); }

private:
    struct SideState
    {
        FKMerg::SideLayout layout;
        bool isFkSide = false;
        /// Per worker: control buffer, staging buffer (stagingBatchSize rows), fill count.
        std::vector<TupleBuffer> controlBuffers;
        std::vector<TupleBuffer> stagingBuffers;
        std::vector<uint64_t> stagingCounts;
        /// Shared slot array (key-sorted window for L3/L4, timestamp-headered
        /// arrival log for L2); backing buffers retained for the growth chain.
        std::vector<TupleBuffer> sortedBackingBuffers;
        FKMerg::SortedSide sorted;
    };

    [[nodiscard]] uint64_t stagedCount(uint64_t side) const;

    /// Flushes one worker's staging batch into the sorted side. Must be called
    /// with `stateMutex` held.
    void flushStagingLocked(uint64_t side, uint64_t workerIndex);

    /// Allocates `bytes` bytes backed by an unpooled buffer retained in
    /// `sideState.sortedBackingBuffers`.
    uint8_t* allocateForSide(SideState& sideState, uint64_t bytes);

    AbstractBufferProvider* bufferProvider;
    uint64_t stagingBatchSize;
    FKMerg::FKMergVariant variant;
    std::array<SideState, 2> sides;

    /// Scan result cache (see mergeAndScan).
    bool probed = false;
    TupleBuffer outputBuffer;

    std::mutex stateMutex;
};

}
