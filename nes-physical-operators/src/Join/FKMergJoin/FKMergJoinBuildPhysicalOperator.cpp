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

#include <Join/FKMergJoin/FKMergJoinBuildPhysicalOperator.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <Interface/NautilusBuffer.hpp>
#include <Interface/PagedVector/PagedVectorRef.hpp>
#include <Interface/Record.hpp>
#include <Join/FKMergJoin/FKMergJoinSlice.hpp>
#include <Join/StreamJoinBuildPhysicalOperator.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/SliceStoreRef.hpp>
#include <Watermark/TimeFunction.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <WindowBuildPhysicalOperator.hpp>
#include <function.hpp>
#include <val_ptr.hpp>

namespace NES
{

namespace
{
/// Reads the control buffer handed out by the slice-store extractor and asks
/// the slice for the next staging slot (flushing the worker's full staging
/// batch into the sorted window via OAppend first if necessary).
int8_t* acquireStagingSlotProxy(const TupleBuffer* controlBuffer)
{
    PRECONDITION(controlBuffer != nullptr, "FKMerg control buffer should not be null");
    FKMergStagingRef ref;
    std::memcpy(&ref, controlBuffer->getAvailableMemoryArea<uint8_t>().data(), sizeof(ref));
    PRECONDITION(ref.slice != nullptr, "FKMerg control buffer holds no slice");
    return ref.slice->acquireStagingSlot(ref.side, ref.workerIndex);
}
}

FKMergJoinBuildPhysicalOperator::FKMergJoinBuildPhysicalOperator(
    const OperatorHandlerId operatorHandlerId,
    const JoinBuildSideType joinBuildSide,
    std::unique_ptr<TimeFunction> timeFunction,
    std::shared_ptr<PagedVectorTupleLayout> tupleLayout,
    std::unique_ptr<SliceStoreRef> sliceStoreRef)
    : StreamJoinBuildPhysicalOperator{
          operatorHandlerId, joinBuildSide, std::move(timeFunction), std::move(tupleLayout), std::move(sliceStoreRef)}
{
}

void FKMergJoinBuildPhysicalOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    /// Getting the operator handler from the local state
    auto* const localState = dynamic_cast<WindowOperatorBuildLocalState*>(executionCtx.getLocalState(id));
    auto operatorHandler = localState->getOperatorHandler();

    /// Get the control buffer of the (slice, side, worker) this record belongs to
    const auto timestamp = timeFunction->getTs(executionCtx, record);
    auto controlBuffer = sliceStoreRef->getDataStructureRef(
        timestamp, executionCtx.workerThreadId, operatorHandler, executionCtx.pipelineMemoryProvider.bufferProvider);

    /// Acquire the staging slot natively and write the record's raw row into it.
    /// FK_MERG_L4 rejects VARSIZED fields at lowering time, so the allocate
    /// callback can never be invoked.
    const auto slotPtr = nautilus::invoke(acquireStagingSlotProxy, controlBuffer.asArg());
    tupleLayout->writeRecord(
        record,
        slotPtr,
        [](nautilus::val<int8_t*>, nautilus::val<uint64_t>) -> nautilus::val<int8_t*>
        {
            INVARIANT(false, "FK_MERG_L4 does not support VARSIZED fields; the lowering rule must reject them");
            return nautilus::val<int8_t*>(nullptr);
        });
}

}
