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

#include <LoweringRules/LowerToPhysical/LowerToPhysicalFKMergJoin.hpp>

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <DataTypes/SchemaFwd.hpp>
#include <DataTypes/UnboundField.hpp>
#include <Functions/FieldAccessLogicalFunction.hpp>
#include <Functions/FunctionProvider.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Identifiers/QualifiedIdentifier.hpp>
#include <Interface/PagedVector/PagedVectorRef.hpp>
#include <Iterators/BFSIterator.hpp>
#include <Join/FKMergJoin/FKMergJoinBuildPhysicalOperator.hpp>
#include <Join/FKMergJoin/FKMergJoinOperatorHandler.hpp>
#include <Join/FKMergJoin/FKMergJoinProbePhysicalOperator.hpp>
#include <Join/FKMergJoin/FKMergJoinSlice.hpp>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/JoinTriggerStrategy.hpp>
#include <Join/StreamJoinOperatorHandler.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <LoweringRules/AbstractLoweringRule.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <SliceStore/DefaultTimeBasedSliceStore.hpp>
#include <SliceStore/Slice.hpp>
#include <Traits/FieldMappingTrait.hpp>
#include <Traits/JoinImplementationTypeTrait.hpp>
#include <Traits/MemoryLayoutTypeTrait.hpp>
#include <Traits/OutputOriginIdsTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/SchemaFactory.hpp>
#include <Watermark/TimeFunction.hpp>
#include <WindowTypes/Measures/TimeCharacteristic.hpp>
#include <WindowTypes/Types/TimeBasedWindowType.hpp>
#include <magic_enum/magic_enum.hpp>
#include <ErrorHandling.hpp>
#include <LoweringRuleRegistry.hpp>
#include <PhysicalOperator.hpp>
#include <WindowBasedOperatorHandler.hpp>

namespace NES
{

namespace
{

/// Rows per staging batch, i.e. the OAppend micro-batch size m. The reference
/// evaluation's default batch size is 1024 tuples (tee-bench-obliv-stream
/// commons.c, dataset synth-1).
constexpr uint64_t STAGING_BATCH_SIZE = 1024;

/// The PK side (unique keys within every window) is the right input by
/// convention; a violation surfaces as FKJoinPreconditionViolated at runtime.
constexpr auto PK_SIDE = JoinBuildSideType::Right;

std::vector<QualifiedIdentifier>
getJoinFieldNames(const Schema<QualifiedUnboundField, Ordered>& inputSchema, const LogicalFunction& joinFunction)
{
    return BFSRange(joinFunction)
        | std::views::filter([](const auto& child) { return child.template tryGetAs<FieldAccessLogicalFunction>().has_value(); })
        | std::views::transform([](const auto& child) { return child.template tryGetAs<FieldAccessLogicalFunction>().value()->getField(); })
        | std::views::filter([&](const auto& field) { return inputSchema.contains(field.getLastName()); })
        | std::views::transform([](const auto& field) { return QualifiedIdentifier{field.getLastName()}; })
        | std::ranges::to<std::vector>();
};

/// Builds the FKMerg::SideLayout of one input side: the row size, and the byte
/// offset/type/nullability of the join-key field within the
/// DefaultPagedVectorTupleLayout row format (fields contiguous in schema
/// order, each occupying getSizeInBytesWithNull() bytes).
FKMerg::SideLayout
buildSideLayout(const Schema<QualifiedUnboundField, Ordered>& schema, const QualifiedIdentifier& keyFieldName, const std::string& sideName)
{
    const auto keyLastName = *std::prev(keyFieldName.end());

    FKMerg::SideLayout layout;
    layout.rowSize = schema.getSizeInBytes();

    uint64_t offset = 0;
    bool found = false;
    for (const auto& field : schema)
    {
        const auto& dataType = field.getDataType();
        if (dataType.type == DataType::Type::VARSIZED)
        {
            throw UnsupportedQuery("FK-MERG does not support VARSIZED fields, but the {} input schema contains one", sideName);
        }
        const auto& fieldName = field.getFullyQualifiedName();
        if (not found and *std::prev(fieldName.end()) == keyLastName)
        {
            layout.keyOffset = offset;
            layout.keyType = dataType.type;
            layout.keyNullable = dataType.nullable;
            found = true;
        }
        offset += dataType.getSizeInBytesWithNull();
    }
    if (not found)
    {
        throw UnsupportedQuery("FK-MERG could not locate the join key field {} in the {} input schema", keyFieldName, sideName);
    }
    switch (layout.keyType)
    {
        case DataType::Type::INT32:
        case DataType::Type::INT64:
        case DataType::Type::UINT32:
        case DataType::Type::UINT64:
            break;
        default:
            throw UnsupportedQuery(
                "FK-MERG supports INT32/INT64/UINT32/UINT64 join keys, but the {} key {} has type {}",
                sideName,
                keyFieldName,
                magic_enum::enum_name(layout.keyType));
    }
    return layout;
}

/// Locates the event-time field within the side's row format and stores its
/// offset/type in the layout — needed by the L2 variant, which replays tuples
/// in event-time order.
void setTimestampField(
    FKMerg::SideLayout& layout,
    const Schema<QualifiedUnboundField, Ordered>& schema,
    const Windowing::BoundEventTimeCharacteristic& eventTime,
    const std::string& sideName)
{
    const auto tsLastName = eventTime.field->getField().getLastName();
    uint64_t offset = 0;
    for (const auto& field : schema)
    {
        const auto& dataType = field.getDataType();
        const auto& fieldName = field.getFullyQualifiedName();
        if (*std::prev(fieldName.end()) == tsLastName)
        {
            layout.tsOffset = offset;
            layout.tsType = dataType.type;
            layout.tsNullable = dataType.nullable;
            return;
        }
        offset += dataType.getSizeInBytesWithNull();
    }
    throw UnsupportedQuery("The L2 variants could not locate the event-time field {} in the {} input schema", tsLastName, sideName);
}

}

LoweringRuleResultSubgraph LowerToPhysicalFKMergJoin::apply(LogicalOperator logicalOperator)
{
    auto join = logicalOperator.getAs<JoinLogicalOperator>();
    auto children = join->getBothChildren();
    const auto traitSet = logicalOperator.getTraitSet();

    auto outputOriginIds = traitSet.get<OutputOriginIdsTrait>();
    PRECONDITION(std::ranges::size(*outputOriginIds) == 1, "Expected one output origin id");

    const auto memoryLayoutTypeTrait = traitSet.get<MemoryLayoutTypeTrait>();
    const auto memoryLayoutType = memoryLayoutTypeTrait->memoryLayout;

    /// L4 emits the dummy padding, L3 obliviously compacts it away, L2 replays
    /// tuple-at-a-time in event-time order with per-tuple compaction. MERG
    /// maintains key-sorted windows; SORT re-sorts everything per invocation.
    const auto implementationTrait = traitSet.get<JoinImplementationTypeTrait>();
    const auto variant = [&]
    {
        switch (implementationTrait->implementationType)
        {
            case JoinImplementation::FK_MERG_L2:
                return FKMerg::FKMergVariant::MERG_L2;
            case JoinImplementation::FK_MERG_L3:
                return FKMerg::FKMergVariant::MERG_L3;
            case JoinImplementation::FK_SORT_L2:
                return FKMerg::FKMergVariant::SORT_L2;
            case JoinImplementation::FK_SORT_L3:
                return FKMerg::FKMergVariant::SORT_L3;
            case JoinImplementation::FK_SORT_L4:
                return FKMerg::FKMergVariant::SORT_L4;
            case JoinImplementation::NFK_JOIN_L2:
                return FKMerg::FKMergVariant::NFK_L2;
            case JoinImplementation::NFK_JOIN_L3:
                return FKMerg::FKMergVariant::NFK_L3;
            case JoinImplementation::NLJ_L4:
                return FKMerg::FKMergVariant::NLJ_L4;
            default:
                return FKMerg::FKMergVariant::MERG_L4;
        }
    }();

    auto handlerId = getNextOperatorHandlerId();

    auto leftInputSchema = createPhysicalOutputSchema(children[0]->getTraitSet());
    auto rightInputSchema = createPhysicalOutputSchema(children[1]->getTraitSet());
    auto outputSchema = createPhysicalOutputSchema(traitSet);
    auto outputOriginId = (*outputOriginIds)[0];
    auto logicalJoinFunction = join->getJoinFunction();
    auto windowType = join->getWindowType();

    /// FK-MERG constraints beyond what DecideJoinTypesRule checks.
    if (join->getJoinType() != JoinLogicalOperator::JoinType::INNER_JOIN)
    {
        throw UnsupportedQuery("FK-MERG only supports inner joins, got {}", magic_enum::enum_name(join->getJoinType()));
    }
    if (windowType.getSize().getTime() != windowType.getSlide().getTime())
    {
        throw UnsupportedQuery(
            "FK-MERG only supports tumbling windows, got size {} and slide {}",
            windowType.getSize().getTime(),
            windowType.getSlide().getTime());
    }

    const auto leftKeyFieldNames = getJoinFieldNames(leftInputSchema, logicalJoinFunction);
    const auto rightKeyFieldNames = getJoinFieldNames(rightInputSchema, logicalJoinFunction);
    if (leftKeyFieldNames.size() != 1 or rightKeyFieldNames.size() != 1)
    {
        throw UnsupportedQuery(
            "FK-MERG requires a single-key equi-join, got {} left and {} right key fields",
            leftKeyFieldNames.size(),
            rightKeyFieldNames.size());
    }

    auto leftSideLayout = buildSideLayout(leftInputSchema, leftKeyFieldNames.front(), "left");
    auto rightSideLayout = buildSideLayout(rightInputSchema, rightKeyFieldNames.front(), "right");
    if (leftSideLayout.keyType != rightSideLayout.keyType)
    {
        throw UnsupportedQuery(
            "FK-MERG requires identical join key types on both sides, got {} and {}",
            magic_enum::enum_name(leftSideLayout.keyType),
            magic_enum::enum_name(rightSideLayout.keyType));
    }

    const auto inputOriginIds
        = join.getChildren()
        | std::views::transform(
              [](const auto& child)
              {
                  auto childOutputOriginIds = getTrait<OutputOriginIdsTrait>(child.getTraitSet());
                  PRECONDITION(childOutputOriginIds.has_value(), "Expected the outputOriginIds trait of the child to be set");
                  return *childOutputOriginIds.value();
              })
        | std::views::join | std::ranges::to<std::vector<OriginId>>();

    auto combinedFieldMappingVec = join->getChildren()
        | std::views::transform([](const auto& child)
                                { return child.getTraitSet().template get<FieldMappingTrait>()->getUnderlying() | std::views::all; })
        | std::views::join | std::views::common | std::ranges::to<std::unordered_map>();
    auto combinedFieldMapping = FieldMappingTrait{std::move(combinedFieldMappingVec)};

    auto joinFunction = QueryCompilation::FunctionProvider::lowerFunction(logicalJoinFunction, combinedFieldMapping);
    auto leftTupleLayout = std::make_shared<DefaultPagedVectorTupleLayout>(leftInputSchema);
    auto rightTupleLayout = std::make_shared<DefaultPagedVectorTupleLayout>(rightInputSchema);

    const auto& joinTimeCharacteristicsVariant = join->getJoinTimeCharacteristics();
    auto characteristicsAreBound
        = std::holds_alternative<std::array<Windowing::BoundTimeCharacteristic, 2>>(joinTimeCharacteristicsVariant);
    PRECONDITION(characteristicsAreBound, "Expected the join time characteristics to be bound");
    const auto& [timeStampFieldLeft, timeStampFieldRight]
        = std::get<std::array<Windowing::BoundTimeCharacteristic, 2>>(joinTimeCharacteristicsVariant);

    if (FKMerg::isPerTupleVariant(variant))
    {
        /// L2 replays tuples in event-time order, so both sides must carry an
        /// event-time field whose row offset the replay can read.
        const auto* leftEventTime = std::get_if<Windowing::BoundEventTimeCharacteristic>(&timeStampFieldLeft);
        const auto* rightEventTime = std::get_if<Windowing::BoundEventTimeCharacteristic>(&timeStampFieldRight);
        if (leftEventTime == nullptr or rightEventTime == nullptr)
        {
            throw UnsupportedQuery("The L2 variants require event-time characteristics on both inputs");
        }
        setTimestampField(leftSideLayout, leftInputSchema, *leftEventTime, "left");
        setTimestampField(rightSideLayout, rightInputSchema, *rightEventTime, "right");
    }

    auto sliceAndWindowStore = std::make_unique<DefaultTimeBasedSliceStore>(
        windowType.getSize().getTime(), windowType.getSlide().getTime(), conf.sliceCacheConfiguration);
    auto sliceStoreRefLeft = sliceAndWindowStore->createSliceStoreRef(
        [](Slice& slice, const WorkerThreadId workerThreadId, AbstractBufferProvider&)
        {
            auto& fkMergSlice = dynamic_cast<FKMergJoinSlice&>(slice);
            return fkMergSlice.getControlBufferRef(workerThreadId, JoinBuildSideType::Left);
        },
        [leftSideLayout, rightSideLayout, variant](const WindowBasedOperatorHandler& handler, AbstractBufferProvider& bufferProvider)
        {
            const CreateNewFKMergSliceArgs sliceArgs{bufferProvider, leftSideLayout, rightSideLayout, PK_SIDE, STAGING_BATCH_SIZE, variant};
            return handler.getCreateNewSlicesFunction(sliceArgs);
        });
    auto sliceStoreRefRight = sliceAndWindowStore->createSliceStoreRef(
        [](Slice& slice, const WorkerThreadId workerThreadId, AbstractBufferProvider&)
        {
            auto& fkMergSlice = dynamic_cast<FKMergJoinSlice&>(slice);
            return fkMergSlice.getControlBufferRef(workerThreadId, JoinBuildSideType::Right);
        },
        [leftSideLayout, rightSideLayout, variant](const WindowBasedOperatorHandler& handler, AbstractBufferProvider& bufferProvider)
        {
            const CreateNewFKMergSliceArgs sliceArgs{bufferProvider, leftSideLayout, rightSideLayout, PK_SIDE, STAGING_BATCH_SIZE, variant};
            return handler.getCreateNewSlicesFunction(sliceArgs);
        });

    auto handler = std::make_shared<FKMergJoinOperatorHandler>(
        inputOriginIds, outputOriginId, std::move(sliceAndWindowStore), InnerJoinTriggerStrategy{});

    const FKMergJoinBuildPhysicalOperator leftBuildOperator{
        handlerId, JoinBuildSideType::Left, TimeFunction::create(timeStampFieldLeft), leftTupleLayout, std::move(sliceStoreRefLeft)};
    const FKMergJoinBuildPhysicalOperator rightBuildOperator{
        handlerId, JoinBuildSideType::Right, TimeFunction::create(timeStampFieldRight), rightTupleLayout, std::move(sliceStoreRefRight)};

    auto joinSchema = JoinSchema(leftInputSchema, rightInputSchema, outputSchema);

    auto leftBuildWrapper = std::make_shared<PhysicalOperatorWrapper>(
        std::move(leftBuildOperator),
        leftInputSchema,
        outputSchema,
        memoryLayoutType,
        memoryLayoutType,
        handlerId,
        handler,
        PhysicalOperatorWrapper::PipelineLocation::EMIT);

    auto rightBuildWrapper = std::make_shared<PhysicalOperatorWrapper>(
        std::move(rightBuildOperator),
        rightInputSchema,
        outputSchema,
        memoryLayoutType,
        memoryLayoutType,
        handlerId,
        handler,
        PhysicalOperatorWrapper::PipelineLocation::EMIT);

    static_assert(JoinProbeOperator<FKMergJoinProbePhysicalOperator>);
    PRECONDITION(
        FKMergJoinProbePhysicalOperator::supportsJoinType(join->getJoinType()),
        "FKMergJoinProbePhysicalOperator does not support join type");

    auto probeWrapper = std::make_shared<PhysicalOperatorWrapper>(
        FKMergJoinProbePhysicalOperator(
            handlerId,
            joinFunction,
            WindowMetaData{join->getStartField(), join->getEndField()},
            joinSchema,
            leftTupleLayout,
            rightTupleLayout,
            PK_SIDE),
        outputSchema,
        outputSchema,
        memoryLayoutType,
        memoryLayoutType,
        handlerId,
        handler,
        PhysicalOperatorWrapper::PipelineLocation::SCAN,
        std::vector{leftBuildWrapper, rightBuildWrapper});

    return {.root = {probeWrapper}, .leaves = {leftBuildWrapper, rightBuildWrapper}};
};

std::unique_ptr<AbstractLoweringRule>
LoweringRuleGeneratedRegistrar::RegisterFKMergJoinLoweringRule(LoweringRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalFKMergJoin>(argument.conf);
}

}
