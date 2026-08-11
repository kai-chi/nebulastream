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

#include <memory>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVectorRef.hpp>
#include <Join/StreamJoinBuildPhysicalOperator.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <SliceStore/SliceStoreRef.hpp>
#include <Watermark/TimeFunction.hpp>
#include <ExecutionContext.hpp>

namespace NES
{

/// Build phase of the FK-MERG-L4 join. Each record is written into the
/// worker-local staging area of its slice; every full staging batch is
/// obliviously appended (OAppend) into the slice's shared sorted window. The
/// slice-store extractor hands out a small control buffer (FKMergStagingRef)
/// instead of the data structure itself, which the execute() proxy uses to
/// reach the slice natively.
class FKMergJoinBuildPhysicalOperator final : public StreamJoinBuildPhysicalOperator
{
public:
    FKMergJoinBuildPhysicalOperator(
        OperatorHandlerId operatorHandlerId,
        JoinBuildSideType joinBuildSide,
        std::unique_ptr<TimeFunction> timeFunction,
        std::shared_ptr<PagedVectorTupleLayout> tupleLayout,
        std::unique_ptr<SliceStoreRef> sliceStoreRef);

    void execute(ExecutionContext& executionCtx, Record& record) const override;
};

}
