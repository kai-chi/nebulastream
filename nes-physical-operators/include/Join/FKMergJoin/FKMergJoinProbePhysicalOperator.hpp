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
#include <memory>
#include <Functions/PhysicalFunction.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVectorRef.hpp>
#include <Interface/RecordBuffer.hpp>
#include <Join/StreamJoinProbePhysicalOperator.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Operators/Windows/WindowMetaData.hpp>
#include <ExecutionContext.hpp>

namespace NES
{

/// Probe phase of the FK-MERG-L4 join. At window-trigger time it runs the
/// tagged cross-stream merge and the dummy-emitting scan natively on the
/// slice's sorted sides, then walks the fixed-size output slots in traced
/// code, emitting one record per slot — real matches and dummies alike, with
/// no join-function filter (the padding is the point of the L4 profile).
class FKMergJoinProbePhysicalOperator final : public StreamJoinProbePhysicalOperator
{
public:
    FKMergJoinProbePhysicalOperator(
        OperatorHandlerId operatorHandlerId,
        PhysicalFunction joinFunction,
        WindowMetaData windowMetaData,
        const JoinSchema& joinSchema,
        std::shared_ptr<PagedVectorTupleLayout> leftTupleLayout,
        std::shared_ptr<PagedVectorTupleLayout> rightTupleLayout,
        JoinBuildSideType pkSide);

    static constexpr bool supportsJoinType(const JoinLogicalOperator::JoinType joinType)
    {
        return joinType == JoinLogicalOperator::JoinType::INNER_JOIN;
    }

    void open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const override;

private:
    std::shared_ptr<PagedVectorTupleLayout> leftTupleLayout;
    std::shared_ptr<PagedVectorTupleLayout> rightTupleLayout;
    JoinBuildSideType pkSide;
    /// Row sizes as stored in the scan output slots (fixed-size schemas only).
    uint64_t leftRowSize;
    uint64_t rightRowSize;
};

}
