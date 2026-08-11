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

#include <Join/FKMergJoin/FKMergJoinOperatorHandler.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Join/FKMergJoin/FKMergJoinSlice.hpp>
#include <Join/StreamJoinOperatorHandler.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Sequencing/SequenceData.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Time/Timestamp.hpp>
#include <ErrorHandling.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

FKMergJoinOperatorHandler::FKMergJoinOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    const OriginId outputOriginId,
    std::unique_ptr<WindowSlicesStoreInterface> sliceAndWindowStore,
    JoinTriggerStrategy triggerStrategy)
    : StreamJoinOperatorHandler(inputOrigins, outputOriginId, std::move(sliceAndWindowStore), std::move(triggerStrategy))
{
}

std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>
FKMergJoinOperatorHandler::getCreateNewSlicesFunction(const CreateNewSlicesArguments& args) const
{
    PRECONDITION(
        numberOfWorkerThreads > 0, "Number of worker threads not set for window based operator. Was setWorkerThreads() being called?");
    const auto& fkMergArgs = dynamic_cast<const CreateNewFKMergSliceArgs&>(args);
    return std::function(
        [numberOfWorkerThreads = numberOfWorkerThreads,
         bufferProvider = fkMergArgs.bufferProvider,
         leftLayout = fkMergArgs.leftLayout,
         rightLayout = fkMergArgs.rightLayout,
         pkSide = fkMergArgs.pkSide,
         stagingBatchSize = fkMergArgs.stagingBatchSize,
         variant = fkMergArgs.variant](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            return {std::make_shared<FKMergJoinSlice>(
                *bufferProvider, start, end, numberOfWorkerThreads, leftLayout, rightLayout, pkSide, stagingBatchSize, variant)};
        });
}

void FKMergJoinOperatorHandler::emitSlicesToProbe(
    const std::vector<std::shared_ptr<Slice>>& leftSlices,
    const std::vector<std::shared_ptr<Slice>>& rightSlices,
    const ProbeTaskType probeTaskType,
    const WindowInfo& windowInfo,
    const SequenceData& sequenceData,
    PipelineExecutionContext* pipelineCtx)
{
    /// The lowering rule restricts FK_MERG_L4 to tumbling windows (one slice
    /// per window, shared by both sides) and inner joins (MATCH_PAIRS tasks).
    INVARIANT(
        leftSlices.size() == 1 && rightSlices.size() == 1,
        "FK_MERG_L4 expects exactly one slice per window side (tumbling windows), got {} left and {} right",
        leftSlices.size(),
        rightSlices.size());
    INVARIANT(probeTaskType == ProbeTaskType::MATCH_PAIRS, "FK_MERG_L4 only supports inner-join MATCH_PAIRS probe tasks");

    /// Flush residual staging batches so the probe sees the complete windows.
    /// Both vectors typically hold the same Slice object; flushAllStaging is
    /// idempotent, so flushing twice is harmless.
    uint64_t totalNumberOfTuples = 0;
    for (const auto& slice : {leftSlices.front(), rightSlices.front()})
    {
        auto& fkMergSlice = dynamic_cast<FKMergJoinSlice&>(*slice);
        fkMergSlice.flushAllStaging();
    }
    {
        const auto& fkMergSlice = dynamic_cast<const FKMergJoinSlice&>(*leftSlices.front());
        totalNumberOfTuples += fkMergSlice.getNumberOfTuplesLeft();
    }
    {
        const auto& fkMergSlice = dynamic_cast<const FKMergJoinSlice&>(*rightSlices.front());
        totalNumberOfTuples += fkMergSlice.getNumberOfTuplesRight();
    }

    const auto tupleBufferVal = pipelineCtx->getBufferManager()->getUnpooledBuffer(sizeof(EmittedFKMergWindowTrigger));
    if (not tupleBufferVal.has_value())
    {
        throw CannotAllocateBuffer("{}B for the FKMerg window trigger were requested", sizeof(EmittedFKMergWindowTrigger));
    }

    auto tupleBuffer = tupleBufferVal.value();
    tupleBuffer.setOriginId(outputOriginId);
    tupleBuffer.setSequenceNumber(SequenceNumber(sequenceData.sequenceNumber));
    tupleBuffer.setChunkNumber(ChunkNumber(sequenceData.chunkNumber));
    tupleBuffer.setLastChunk(sequenceData.lastChunk);
    tupleBuffer.setWatermark(windowInfo.windowStart);
    tupleBuffer.setNumberOfTuples(totalNumberOfTuples);
    tupleBuffer.setCreationTimestampInMS(Timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()));
    new (tupleBuffer.getAvailableMemoryArea().data()) EmittedFKMergWindowTrigger{
        .windowInfo = windowInfo,
        .leftSliceEnd = leftSlices.front()->getSliceEnd(),
        .rightSliceEnd = rightSlices.front()->getSliceEnd(),
        .probeTaskType = probeTaskType};

    pipelineCtx->emitBuffer(tupleBuffer);
}

}
