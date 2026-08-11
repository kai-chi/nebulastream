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

#include <Join/FKMergJoin/FKMergJoinProbePhysicalOperator.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <DataTypes/Schema.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Interface/NESStrongTypeRef.hpp>
#include <Interface/PagedVector/PagedVectorRef.hpp>
#include <Interface/Record.hpp>
#include <Interface/RecordBuffer.hpp>
#include <Interface/TimestampRef.hpp>
#include <Join/FKMergJoin/FKMergJoinOperatorHandler.hpp>
#include <Join/FKMergJoin/FKMergJoinSlice.hpp>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/StreamJoinProbePhysicalOperator.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Operators/Windows/WindowMetaData.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Time/Timestamp.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <function.hpp>
#include <val.hpp>
#include <val_ptr.hpp>

namespace NES
{

namespace
{
/// Resolves the slice and runs the (idempotent) native merge+scan. Returns the
/// output area: a uint64 slot count followed by the output slots.
int8_t* mergeAndScanProxy(OperatorHandler* ptrOpHandler, const SliceEnd sliceEnd)
{
    PRECONDITION(ptrOpHandler != nullptr, "op handler context should not be null");
    auto* opHandler = dynamic_cast<FKMergJoinOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "expected an FKMergJoinOperatorHandler");
    const auto slice = opHandler->getSliceAndWindowStore().getSliceBySliceEnd(sliceEnd);
    INVARIANT(slice.has_value(), "Could not find a slice for slice end {}", sliceEnd);
    auto* fkMergSlice = dynamic_cast<FKMergJoinSlice*>(slice.value().get());
    INVARIANT(fkMergSlice != nullptr, "expected an FKMergJoinSlice");
    return const_cast<int8_t*>(fkMergSlice->mergeAndScan());
}
}

FKMergJoinProbePhysicalOperator::FKMergJoinProbePhysicalOperator(
    const OperatorHandlerId operatorHandlerId,
    PhysicalFunction joinFunction,
    WindowMetaData windowMetaData,
    const JoinSchema& joinSchema,
    std::shared_ptr<PagedVectorTupleLayout> leftTupleLayout,
    std::shared_ptr<PagedVectorTupleLayout> rightTupleLayout,
    const JoinBuildSideType pkSide)
    : StreamJoinProbePhysicalOperator(operatorHandlerId, std::move(joinFunction), std::move(windowMetaData), joinSchema)
    , leftTupleLayout(std::move(leftTupleLayout))
    , rightTupleLayout(std::move(rightTupleLayout))
    , pkSide(pkSide)
    , leftRowSize(this->leftTupleLayout->getSchema().getSizeInBytes())
    , rightRowSize(this->rightTupleLayout->getSchema().getSizeInBytes())
{
}

void FKMergJoinProbePhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    StreamJoinProbePhysicalOperator::open(executionCtx, recordBuffer);

    /// Parse the fixed-size trigger buffer.
    const auto triggerRef = static_cast<nautilus::val<EmittedFKMergWindowTrigger*>>(recordBuffer.getMemArea());
    const auto windowInfoRef = getMemberRef(triggerRef, &EmittedFKMergWindowTrigger::windowInfo);
    const auto windowStart = nautilus::val<Timestamp>{readValueFromMemRef<uint64_t>(getMemberRef(windowInfoRef, &WindowInfo::windowStart))};
    const auto windowEnd = nautilus::val<Timestamp>{readValueFromMemRef<uint64_t>(getMemberRef(windowInfoRef, &WindowInfo::windowEnd))};
    const nautilus::val<SliceEnd> sliceEnd{
        readValueFromMemRef<SliceEnd::Underlying>(getMemberRef(triggerRef, &EmittedFKMergWindowTrigger::leftSliceEnd))};

    /// Run the native merge+scan and walk the output slots. Output slot layout
    /// (FKMerg::outputSlotSize): [uint64 flags][pkRow][fkRow].
    const auto operatorHandlerRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    const auto outArea = nautilus::invoke(mergeAndScanProxy, operatorHandlerRef, sliceEnd);
    const auto numSlots = readValueFromMemRef<uint64_t>(outArea);

    const bool pkIsLeft = pkSide == JoinBuildSideType::Left;
    const uint64_t pkRowSize = pkIsLeft ? leftRowSize : rightRowSize;
    const uint64_t outSlotSize = sizeof(uint64_t) + leftRowSize + rightRowSize;
    const uint64_t leftRegionOffset = sizeof(uint64_t) + (pkIsLeft ? 0 : pkRowSize);
    const uint64_t rightRegionOffset = sizeof(uint64_t) + (pkIsLeft ? pkRowSize : 0);

    const auto leftFields = getOrderedFieldNames(leftTupleLayout->getSchema());
    const auto rightFields = getOrderedFieldNames(rightTupleLayout->getSchema());

    /// FK_MERG_L4 rejects VARSIZED fields at lowering time, so the load
    /// callback can never be invoked.
    const auto unreachableLoad = [](nautilus::val<int8_t*>) -> std::pair<nautilus::val<int8_t*>, nautilus::val<uint64_t>>
    {
        INVARIANT(false, "FK_MERG_L4 does not support VARSIZED fields; the lowering rule must reject them");
        return {nautilus::val<int8_t*>(nullptr), nautilus::val<uint64_t>(0)};
    };

    const auto slotsBase = outArea + nautilus::val<uint64_t>(sizeof(uint64_t));
    for (nautilus::val<uint64_t> i = 0; i < numSlots; ++i)
    {
        const auto slotPtr = slotsBase + (i * nautilus::val<uint64_t>(outSlotSize));
        /// Every slot emits exactly one record — real matches and dummies
        /// alike (dummies carry the all-zero sentinel rows). No join-function
        /// filter: the scan already performed the key-equality test.
        const auto leftRecord = leftTupleLayout->readRecord(slotPtr + nautilus::val<uint64_t>(leftRegionOffset), unreachableLoad);
        const auto rightRecord = rightTupleLayout->readRecord(slotPtr + nautilus::val<uint64_t>(rightRegionOffset), unreachableLoad);
        auto joinedRecord = createJoinedRecord(leftRecord, rightRecord, windowStart, windowEnd, leftFields, rightFields);
        executeChild(executionCtx, joinedRecord);
    }
}

}
