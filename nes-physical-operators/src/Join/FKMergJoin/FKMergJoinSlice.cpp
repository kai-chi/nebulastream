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

#include <Join/FKMergJoin/FKMergJoinSlice.hpp>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/FKMergJoin/Oblivious/NfkJoin.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

FKMergJoinSlice::FKMergJoinSlice(
    AbstractBufferProvider& bufferProvider,
    const SliceStart sliceStart,
    const SliceEnd sliceEnd,
    const uint64_t numberOfWorkerThreads,
    const FKMerg::SideLayout& leftLayout,
    const FKMerg::SideLayout& rightLayout,
    const JoinBuildSideType pkSide,
    const uint64_t stagingBatchSize,
    const FKMerg::FKMergVariant variant)
    : Slice(sliceStart, sliceEnd), bufferProvider(&bufferProvider), stagingBatchSize(stagingBatchSize), variant(variant)
{
    sides[0].layout = leftLayout;
    sides[1].layout = rightLayout;
    sides[0].isFkSide = pkSide != JoinBuildSideType::Left;
    sides[1].isFkSide = pkSide != JoinBuildSideType::Right;

    for (uint64_t side = 0; side < sides.size(); ++side)
    {
        auto& state = sides[side];
        state.stagingCounts.assign(numberOfWorkerThreads, 0);
        for (uint64_t worker = 0; worker < numberOfWorkerThreads; ++worker)
        {
            auto controlBuffer = bufferProvider.getUnpooledBuffer(sizeof(FKMergStagingRef));
            if (not controlBuffer.has_value())
            {
                throw BufferAllocationFailure("No unpooled TupleBuffer available for the FKMerg staging control buffer");
            }
            const FKMergStagingRef ref{.slice = this, .side = side, .workerIndex = worker};
            std::memcpy(controlBuffer->getAvailableMemoryArea<uint8_t>().data(), &ref, sizeof(ref));
            state.controlBuffers.emplace_back(controlBuffer.value());

            auto stagingBuffer = bufferProvider.getUnpooledBuffer(stagingBatchSize * state.layout.rowSize);
            if (not stagingBuffer.has_value())
            {
                throw BufferAllocationFailure("No unpooled TupleBuffer available for the FKMerg staging area");
            }
            state.stagingBuffers.emplace_back(stagingBuffer.value());
        }
    }
}

const TupleBuffer* FKMergJoinSlice::getControlBufferRef(const WorkerThreadId workerThreadId, const JoinBuildSideType side) const
{
    const auto& state = sides[side == JoinBuildSideType::Left ? 0 : 1];
    const auto pos = workerThreadId % state.controlBuffers.size();
    return &state.controlBuffers[pos];
}

int8_t* FKMergJoinSlice::acquireStagingSlot(const uint64_t side, const uint64_t workerIndex)
{
    auto& state = sides[side];
    if (state.stagingCounts[workerIndex] == stagingBatchSize)
    {
        const std::scoped_lock lock(stateMutex);
        flushStagingLocked(side, workerIndex);
    }
    auto* base = state.stagingBuffers[workerIndex].getAvailableMemoryArea<int8_t>().data();
    int8_t* slot = base + (state.stagingCounts[workerIndex] * state.layout.rowSize);
    ++state.stagingCounts[workerIndex];
    return slot;
}

void FKMergJoinSlice::flushAllStaging()
{
    const std::scoped_lock lock(stateMutex);
    for (uint64_t side = 0; side < sides.size(); ++side)
    {
        for (uint64_t worker = 0; worker < sides[side].stagingCounts.size(); ++worker)
        {
            flushStagingLocked(side, worker);
        }
    }
}

uint64_t FKMergJoinSlice::stagedCount(const uint64_t side) const
{
    return std::accumulate(sides[side].stagingCounts.begin(), sides[side].stagingCounts.end(), uint64_t{0});
}

void FKMergJoinSlice::flushStagingLocked(const uint64_t side, const uint64_t workerIndex)
{
    auto& state = sides[side];
    const uint64_t count = state.stagingCounts[workerIndex];
    if (count == 0)
    {
        return;
    }
    const auto* rows = state.stagingBuffers[workerIndex].getAvailableMemoryArea<uint8_t>().data();
    const auto allocate = [this, &state](const uint64_t bytes) { return allocateForSide(state, bytes); };
    if (FKMerg::maintainsSortedWindows(variant))
    {
        FKMerg::oAppend(state.sorted, state.layout, state.isFkSide, rows, count, allocate);
    }
    else
    {
        /// L2 variants keep the tuples in arrival order for the event-time
        /// replay; SORT and NFK variants keep raw logs by design (nothing is
        /// sorted until the per-invocation oblivious sort).
        FKMerg::appendToArrivalLog(state.sorted, state.layout, state.isFkSide, rows, count, allocate);
    }
    state.stagingCounts[workerIndex] = 0;
}

uint8_t* FKMergJoinSlice::allocateForSide(SideState& sideState, const uint64_t bytes)
{
    auto buffer = bufferProvider->getUnpooledBuffer(bytes);
    if (not buffer.has_value())
    {
        throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the FKMerg sorted window", bytes);
    }
    sideState.sortedBackingBuffers.emplace_back(buffer.value());
    return sideState.sortedBackingBuffers.back().getAvailableMemoryArea<uint8_t>().data();
}

const int8_t* FKMergJoinSlice::mergeAndScan()
{
    const std::scoped_lock lock(stateMutex);
    if (probed)
    {
        return outputBuffer.getAvailableMemoryArea<int8_t>().data();
    }

    /// The trigger path flushes before emitting the probe task, but flush again
    /// under the same lock acquisition to be safe against reordered callers.
    for (uint64_t side = 0; side < sides.size(); ++side)
    {
        for (uint64_t worker = 0; worker < sides[side].stagingCounts.size(); ++worker)
        {
            flushStagingLocked(side, worker);
        }
    }

    auto& pkState = sides[0].isFkSide ? sides[1] : sides[0];
    auto& fkState = sides[0].isFkSide ? sides[0] : sides[1];

    if (variant == FKMerg::FKMergVariant::NLJ_L4)
    {
        /// NLJ-L4: the padded oblivious nested loop — output is exactly the
        /// windows' Cartesian product (public given the window cardinalities).
        auto& leftState = sides[0];
        auto& rightState = sides[1];
        const uint64_t outSlot = FKMerg::outputSlotSize(pkState.layout, fkState.layout);
        const uint64_t totalSlots = leftState.sorted.sizeSlots * rightState.sorted.sizeSlots;
        const uint64_t outputBytes = sizeof(uint64_t) + (std::max<uint64_t>(totalSlots, 1) * outSlot);
        auto output = bufferProvider->getUnpooledBuffer(outputBytes);
        if (not output.has_value())
        {
            throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the NLJ-L4 padded output", outputBytes);
        }
        auto* outArea = output->getAvailableMemoryArea<uint8_t>().data();
        FKMerg::nljL4Join(leftState.sorted, leftState.layout, rightState.sorted, rightState.layout, outArea + sizeof(uint64_t));
        std::memcpy(outArea, &totalSlots, sizeof(totalSlots));

        outputBuffer = output.value();
        probed = true;
        return outputBuffer.getAvailableMemoryArea<int8_t>().data();
    }

    if (FKMerg::isNfkVariant(variant))
    {
        /// NFK-JOIN-L2/L3: the Krastnikov-based generic join over left/right
        /// (no PK/FK roles, duplicates allowed on both sides). Output volume
        /// is the true join cardinality (the L3 leakage); pre-size the output
        /// by measuring the whole-window join once.
        auto& leftState = sides[0];
        auto& rightState = sides[1];
        std::vector<TupleBuffer> scratchBuffers;
        const auto scratchAllocate = [this, &scratchBuffers](const uint64_t bytes)
        {
            auto buffer = bufferProvider->getUnpooledBuffer(bytes);
            if (not buffer.has_value())
            {
                throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the NFK join scratch", bytes);
            }
            scratchBuffers.emplace_back(buffer.value());
            return scratchBuffers.back().getAvailableMemoryArea<uint8_t>().data();
        };

        const FKMerg::NfkFreshSelector allFresh{};
        const uint64_t totalMatches
            = FKMerg::nfkJoinSize(leftState.sorted, leftState.layout, rightState.sorted, rightState.layout, allFresh, scratchAllocate);

        const uint64_t outSlot = FKMerg::outputSlotSize(pkState.layout, fkState.layout);
        const uint64_t capacitySlots = std::max<uint64_t>(totalMatches, 1);
        const uint64_t outputBytes = sizeof(uint64_t) + (capacitySlots * outSlot);
        auto output = bufferProvider->getUnpooledBuffer(outputBytes);
        if (not output.has_value())
        {
            throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the NFK join output", outputBytes);
        }
        auto* outArea = output->getAvailableMemoryArea<uint8_t>().data();

        uint64_t emitted = 0;
        if (variant == FKMerg::FKMergVariant::NFK_L3)
        {
            emitted = FKMerg::nfkJoin(
                leftState.sorted,
                leftState.layout,
                rightState.sorted,
                rightState.layout,
                allFresh,
                outArea + sizeof(uint64_t),
                capacitySlots,
                scratchAllocate);
        }
        else
        {
            emitted = FKMerg::nfkPerTupleReplay(
                leftState.sorted,
                leftState.layout,
                rightState.sorted,
                rightState.layout,
                outArea + sizeof(uint64_t),
                capacitySlots,
                scratchAllocate);
        }
        INVARIANT(emitted == totalMatches, "NFK join emitted {} pairs but the pre-measure found {}", emitted, totalMatches);
        std::memcpy(outArea, &emitted, sizeof(emitted));

        outputBuffer = output.value();
        probed = true;
        return outputBuffer.getAvailableMemoryArea<int8_t>().data();
    }

    if (FKMerg::isPerTupleVariant(variant))
    {
        /// FK-MERG-L2 / FK-SORT-L2: replay the arrival logs tuple-by-tuple.
        /// Output bound: each FK tuple matches at most one PK (FK-join
        /// precondition).
        const uint64_t outSlot = FKMerg::outputSlotSize(pkState.layout, fkState.layout);
        const uint64_t capacitySlots = std::max<uint64_t>(fkState.sorted.sizeSlots, 1);
        const uint64_t outputBytes = sizeof(uint64_t) + (capacitySlots * outSlot);
        auto output = bufferProvider->getUnpooledBuffer(outputBytes);
        if (not output.has_value())
        {
            throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the FKMerg L2 replay output", outputBytes);
        }
        auto* outArea = output->getAvailableMemoryArea<uint8_t>().data();

        const auto scanResult = FKMerg::perTupleReplay(
            pkState.sorted,
            pkState.layout,
            fkState.sorted,
            fkState.layout,
            FKMerg::usesFullSort(variant),
            outArea + sizeof(uint64_t),
            capacitySlots,
            [this, &pkState](const uint64_t bytes) { return allocateForSide(pkState, bytes); });
        if (scanResult.duplicatePkDetected)
        {
            throw FKJoinPreconditionViolated(
                "FK-MERG/FK-SORT window [{}, {}) contains two PK-side tuples with the same join key", getSliceStart(), getSliceEnd());
        }
        std::memcpy(outArea, &scanResult.realMatches, sizeof(scanResult.realMatches));

        outputBuffer = output.value();
        probed = true;
        return outputBuffer.getAvailableMemoryArea<int8_t>().data();
    }

    /// MERG combines the two key-sorted windows with a bitonic merge; SORT
    /// concatenates the raw logs and pays a full bitonic sort per invocation.
    const uint64_t paddedSlots = FKMerg::usesFullSort(variant)
        ? FKMerg::nextPowerOfTwo(pkState.sorted.sizeSlots + fkState.sorted.sizeSlots)
        : FKMerg::crossMergePaddedSlots(pkState.sorted.sizeSlots, fkState.sorted.sizeSlots);
    const uint64_t mergedSlot = FKMerg::mergedSlotSize(pkState.layout, fkState.layout);
    auto mergedScratch = bufferProvider->getUnpooledBuffer(paddedSlots * mergedSlot);
    if (not mergedScratch.has_value())
    {
        throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the FKMerg merge scratch", paddedSlots * mergedSlot);
    }
    const uint64_t numMerged = FKMerg::usesFullSort(variant)
        ? FKMerg::sortedCrossCombine(
              pkState.sorted, pkState.layout, fkState.sorted, fkState.layout, mergedScratch->getAvailableMemoryArea<uint8_t>().data())
        : FKMerg::crossMerge(
              pkState.sorted, pkState.layout, fkState.sorted, fkState.layout, mergedScratch->getAvailableMemoryArea<uint8_t>().data());

    const uint64_t outSlot = FKMerg::outputSlotSize(pkState.layout, fkState.layout);
    /// L3 trims the dummies via a power-of-two oblivious compaction, so its
    /// output area needs pow2 slot capacity for the compaction padding.
    const uint64_t outputSlots = FKMerg::compactsWindowResult(variant) ? FKMerg::nextPowerOfTwo(numMerged) : numMerged;
    const uint64_t outputBytes = sizeof(uint64_t) + (outputSlots * outSlot);
    auto output = bufferProvider->getUnpooledBuffer(outputBytes);
    if (not output.has_value())
    {
        throw BufferAllocationFailure("No unpooled TupleBuffer of {}B available for the FKMerg scan output", outputBytes);
    }
    auto* outArea = output->getAvailableMemoryArea<uint8_t>().data();

    const auto scanResult = FKMerg::obliviousScan(
        mergedScratch->getAvailableMemoryArea<uint8_t>().data(), numMerged, pkState.layout, fkState.layout, outArea + sizeof(uint64_t));
    if (scanResult.duplicatePkDetected)
    {
        throw FKJoinPreconditionViolated(
            "FK-MERG/FK-SORT window [{}, {}) contains two PK-side tuples with the same join key", getSliceStart(), getSliceEnd());
    }

    uint64_t emittedSlots = numMerged;
    if (FKMerg::compactsWindowResult(variant))
    {
        /// L3 (paper Algorithm 4, line 10): obliviously compact the real
        /// results to the front and emit only those.
        emittedSlots = FKMerg::trimDummies(outArea + sizeof(uint64_t), numMerged, outSlot);
        INVARIANT(
            emittedSlots == scanResult.realMatches,
            "L3 trim kept {} slots but the scan found {} matches",
            emittedSlots,
            scanResult.realMatches);
    }
    std::memcpy(outArea, &emittedSlots, sizeof(emittedSlots));

    outputBuffer = output.value();
    probed = true;
    return outputBuffer.getAvailableMemoryArea<int8_t>().data();
}

}
