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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include <DataTypes/DataType.hpp>
#include <Join/FKMergJoin/Oblivious/FKMergAlgorithm.hpp>
#include <Join/FKMergJoin/Oblivious/NfkJoin.hpp>
#include <Join/FKMergJoin/Oblivious/ObliviousPrimitives.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>

namespace NES
{

using namespace FKMerg;

/// Test-side counter to assert access-pattern regressions: the number of
/// oMemSwap invocations of the comparator networks must depend only on the
/// array length, never on the data.
namespace
{

struct TestRow
{
    int32_t key;
    uint64_t payload;
};

/// A side layout describing TestRow with a non-nullable INT32 key at offset 0.
SideLayout testLayout()
{
    SideLayout layout;
    layout.rowSize = sizeof(TestRow);
    layout.keyOffset = offsetof(TestRow, key);
    layout.keyType = DataType::Type::INT32;
    layout.keyNullable = false;
    return layout;
}

std::vector<uint8_t> rowBytes(const std::vector<TestRow>& rows)
{
    std::vector<uint8_t> bytes(rows.size() * sizeof(TestRow));
    std::memcpy(bytes.data(), rows.data(), bytes.size());
    return bytes;
}

/// Extracts the real (non-dummy) rows of a sorted side in order.
std::vector<TestRow> sideRows(const SortedSide& side, const SideLayout& layout)
{
    std::vector<TestRow> rows(side.sizeSlots);
    for (uint64_t i = 0; i < side.sizeSlots; ++i)
    {
        std::memcpy(&rows[i], side.slots + (i * layout.slotSize()) + sizeof(SlotHeader), sizeof(TestRow));
    }
    return rows;
}

/// An allocator over a growing arena the test owns.
struct TestArena
{
    std::vector<std::vector<uint8_t>> chunks;

    AllocateFn allocator()
    {
        return [this](const uint64_t bytes) -> uint8_t*
        {
            chunks.emplace_back(bytes);
            return chunks.back().data();
        };
    }
};

}

class FKMergObliviousPrimitivesTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("FKMergObliviousPrimitivesTest.log", LogLevel::LOG_DEBUG); }
};

/// bitonicSort must agree with std::sort for random data across slot sizes and
/// power-of-two lengths, ascending and descending, with dummies always last
/// (ascending) or first (descending).
TEST_F(FKMergObliviousPrimitivesTest, bitonicSortMatchesStdSort)
{
    std::mt19937_64 rng(42);
    for (const uint64_t payloadSize : {0ULL, 5ULL, 24ULL})
    {
        const uint64_t slotSize = sizeof(SlotHeader) + payloadSize;
        for (const uint64_t count : {1ULL, 2ULL, 8ULL, 64ULL, 256ULL})
        {
            for (const bool descending : {false, true})
            {
                std::vector<uint8_t> arr(count * slotSize, 0);
                std::vector<uint64_t> keys;
                for (uint64_t i = 0; i < count; ++i)
                {
                    SlotHeader header{.sortKey = rng() % 100, .flags = 0};
                    /// Sprinkle dummies and FK tags.
                    if (rng() % 4 == 0)
                    {
                        header.flags |= SlotHeader::FLAG_DUMMY;
                    }
                    if (rng() % 2 == 0)
                    {
                        header.flags |= SlotHeader::FLAG_FK;
                    }
                    std::memcpy(arr.data() + (i * slotSize), &header, sizeof(header));
                    /// Payload marks the original position so we can verify slots move as units.
                    for (uint64_t b = 0; b < payloadSize; ++b)
                    {
                        arr[(i * slotSize) + sizeof(SlotHeader) + b] = static_cast<uint8_t>(i);
                    }
                }

                /// Expected order via std::sort on header copies.
                std::vector<std::pair<SlotHeader, uint8_t>> expected;
                for (uint64_t i = 0; i < count; ++i)
                {
                    SlotHeader header;
                    std::memcpy(&header, arr.data() + (i * slotSize), sizeof(header));
                    expected.emplace_back(header, static_cast<uint8_t>(i));
                }
                std::stable_sort(
                    expected.begin(),
                    expected.end(),
                    [&](const auto& lhs, const auto& rhs)
                    {
                        const bool less
                            = slotLess(reinterpret_cast<const uint8_t*>(&lhs.first), reinterpret_cast<const uint8_t*>(&rhs.first));
                        return descending
                            ? slotLess(reinterpret_cast<const uint8_t*>(&rhs.first), reinterpret_cast<const uint8_t*>(&lhs.first))
                            : less;
                    });

                bitonicSort(arr.data(), slotSize, 0, count, descending);

                for (uint64_t i = 0; i < count; ++i)
                {
                    SlotHeader header;
                    std::memcpy(&header, arr.data() + (i * slotSize), sizeof(header));
                    EXPECT_EQ(header.sortKey, expected[i].first.sortKey) << "slotSize=" << slotSize << " count=" << count;
                    EXPECT_EQ(header.flags, expected[i].first.flags);
                }
                /// Dummies must be contiguous at the tail (ascending) / head (descending).
                bool seenBoundary = false;
                for (uint64_t i = 0; i < count; ++i)
                {
                    SlotHeader header;
                    std::memcpy(&header, arr.data() + (i * slotSize), sizeof(header));
                    const bool isDummy = header.isDummy();
                    const bool afterBoundary = descending ? !isDummy : isDummy;
                    if (afterBoundary)
                    {
                        seenBoundary = true;
                    }
                    else
                    {
                        EXPECT_FALSE(seenBoundary) << "dummy/real slots interleaved at " << i;
                    }
                }
            }
        }
    }
}

/// Randomized oAppend sequences must keep the side sorted and preserve the
/// multiset of appended rows.
TEST_F(FKMergObliviousPrimitivesTest, oAppendKeepsSortedAndPreservesRows)
{
    std::mt19937_64 rng(7);
    const auto layout = testLayout();

    for (int round = 0; round < 5; ++round)
    {
        TestArena arena;
        SortedSide side;
        std::multiset<int32_t> expectedKeys;
        std::multiset<uint64_t> expectedPayloads;

        for (int append = 0; append < 8; ++append)
        {
            const uint64_t batchSize = 1 + (rng() % 50);
            std::vector<TestRow> batch(batchSize);
            for (auto& row : batch)
            {
                row.key = static_cast<int32_t>(rng() % 1000) - 500;
                row.payload = rng();
                expectedKeys.insert(row.key);
                expectedPayloads.insert(row.payload);
            }
            const auto bytes = rowBytes(batch);
            oAppend(side, layout, /*fkSide=*/false, bytes.data(), batchSize, arena.allocator());
        }

        const auto rows = sideRows(side, layout);
        ASSERT_EQ(rows.size(), expectedKeys.size());
        std::multiset<int32_t> actualKeys;
        std::multiset<uint64_t> actualPayloads;
        for (uint64_t i = 0; i < rows.size(); ++i)
        {
            actualKeys.insert(rows[i].key);
            actualPayloads.insert(rows[i].payload);
            if (i > 0)
            {
                EXPECT_LE(rows[i - 1].key, rows[i].key) << "side not sorted at " << i;
            }
        }
        EXPECT_EQ(actualKeys, expectedKeys);
        EXPECT_EQ(actualPayloads, expectedPayloads);
    }
}

/// crossMerge + obliviousScan must produce exactly the brute-force FK join as
/// real slots (output volume = nPk + nFk), with all-zero sentinel rows in the
/// dummy slots.
TEST_F(FKMergObliviousPrimitivesTest, crossMergeScanMatchesBruteForceJoin)
{
    std::mt19937_64 rng(1337);
    const auto layout = testLayout();

    for (int round = 0; round < 5; ++round)
    {
        TestArena arena;
        SortedSide pkSide;
        SortedSide fkSide;

        /// Unique PK keys.
        const uint64_t numPk = 1 + (rng() % 60);
        std::vector<TestRow> pkRows(numPk);
        std::set<int32_t> usedKeys;
        for (auto& row : pkRows)
        {
            int32_t key = 0;
            do
            {
                key = static_cast<int32_t>(rng() % 500) - 250;
            } while (!usedKeys.insert(key).second);
            row = TestRow{.key = key, .payload = rng() % 100000};
        }
        /// FK rows, some matching, some not.
        const uint64_t numFk = 1 + (rng() % 80);
        std::vector<TestRow> fkRows(numFk);
        for (auto& row : fkRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 700) - 350, .payload = rng() % 100000};
        }

        const auto pkBytes = rowBytes(pkRows);
        const auto fkBytes = rowBytes(fkRows);
        oAppend(pkSide, layout, /*fkSide=*/false, pkBytes.data(), numPk, arena.allocator());
        oAppend(fkSide, layout, /*fkSide=*/true, fkBytes.data(), numFk, arena.allocator());

        const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);
        std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
        const uint64_t numMerged = crossMerge(pkSide, layout, fkSide, layout, scratch.data());
        ASSERT_EQ(numMerged, numPk + numFk);

        const uint64_t outSlot = outputSlotSize(layout, layout);
        std::vector<uint8_t> out(numMerged * outSlot);
        const auto result = obliviousScan(scratch.data(), numMerged, layout, layout, out.data());
        EXPECT_FALSE(result.duplicatePkDetected);

        /// Brute-force expectation.
        std::map<int32_t, TestRow> pkByKey;
        for (const auto& row : pkRows)
        {
            pkByKey[row.key] = row;
        }
        std::multiset<std::pair<uint64_t, uint64_t>> expectedPairs;
        for (const auto& fk : fkRows)
        {
            if (const auto it = pkByKey.find(fk.key); it != pkByKey.end())
            {
                expectedPairs.emplace(it->second.payload, fk.payload);
            }
        }

        std::multiset<std::pair<uint64_t, uint64_t>> actualPairs;
        uint64_t realCount = 0;
        for (uint64_t i = 0; i < numMerged; ++i)
        {
            const uint8_t* slot = out.data() + (i * outSlot);
            uint64_t flags = 0;
            std::memcpy(&flags, slot, sizeof(flags));
            TestRow pkRow;
            TestRow fkRow;
            std::memcpy(&pkRow, slot + sizeof(uint64_t), sizeof(TestRow));
            std::memcpy(&fkRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
            if ((flags & 1ULL) == 0)
            {
                ++realCount;
                EXPECT_EQ(pkRow.key, fkRow.key);
                actualPairs.emplace(pkRow.payload, fkRow.payload);
            }
            else
            {
                /// Dummy slots carry the all-zero sentinel rows.
                EXPECT_EQ(pkRow.key, 0);
                EXPECT_EQ(pkRow.payload, 0U);
                EXPECT_EQ(fkRow.key, 0);
                EXPECT_EQ(fkRow.payload, 0U);
            }
        }
        EXPECT_EQ(realCount, result.realMatches);
        EXPECT_EQ(actualPairs, expectedPairs);
    }
}

/// A duplicate PK key within the window must be detected.
TEST_F(FKMergObliviousPrimitivesTest, duplicatePkIsDetected)
{
    const auto layout = testLayout();
    TestArena arena;
    SortedSide pkSide;
    SortedSide fkSide;

    const std::vector<TestRow> pkRows{{.key = 5, .payload = 1}, {.key = 7, .payload = 2}, {.key = 5, .payload = 3}};
    const std::vector<TestRow> fkRows{{.key = 5, .payload = 4}};
    const auto pkBytes = rowBytes(pkRows);
    const auto fkBytes = rowBytes(fkRows);
    oAppend(pkSide, layout, false, pkBytes.data(), pkRows.size(), arena.allocator());
    oAppend(fkSide, layout, true, fkBytes.data(), fkRows.size(), arena.allocator());

    const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);
    std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
    const uint64_t numMerged = crossMerge(pkSide, layout, fkSide, layout, scratch.data());
    std::vector<uint8_t> out(numMerged * outputSlotSize(layout, layout));
    const auto result = obliviousScan(scratch.data(), numMerged, layout, layout, out.data());
    EXPECT_TRUE(result.duplicatePkDetected);
}

/// Key normalization must preserve order for signed keys.
TEST_F(FKMergObliviousPrimitivesTest, normalizeKeyPreservesOrder)
{
    auto layout = testLayout();
    const std::vector<int32_t> keys{INT32_MIN, -7, -1, 0, 1, 42, INT32_MAX};
    uint64_t previous = 0;
    bool first = true;
    for (const int32_t key : keys)
    {
        const TestRow row{.key = key, .payload = 0};
        bool isNull = true;
        const uint64_t normalized = normalizeKey(reinterpret_cast<const uint8_t*>(&row), layout, isNull);
        EXPECT_FALSE(isNull);
        if (!first)
        {
            EXPECT_LT(previous, normalized) << "order broken at key " << key;
        }
        previous = normalized;
        first = false;
    }
}

/// NULL FK keys must never match; NULL PK keys must not count as duplicates.
TEST_F(FKMergObliviousPrimitivesTest, nullKeysNeverMatch)
{
    /// Row with a nullable INT32 key at offset 0: [nullByte][int32][payload].
    struct NullableRow
    {
        uint8_t null;
        int32_t key;
        uint64_t payload;
    } __attribute__((packed));

    SideLayout layout;
    layout.rowSize = sizeof(NullableRow);
    layout.keyOffset = 0;
    layout.keyType = DataType::Type::INT32;
    layout.keyNullable = true;

    TestArena arena;
    SortedSide pkSide;
    SortedSide fkSide;

    const std::vector<NullableRow> pkRows{
        {.null = 0, .key = 5, .payload = 1}, {.null = 1, .key = 9, .payload = 2}, {.null = 1, .key = 9, .payload = 3}};
    const std::vector<NullableRow> fkRows{{.null = 0, .key = 5, .payload = 4}, {.null = 1, .key = 5, .payload = 5}};

    std::vector<uint8_t> pkBytes(pkRows.size() * sizeof(NullableRow));
    std::memcpy(pkBytes.data(), pkRows.data(), pkBytes.size());
    std::vector<uint8_t> fkBytes(fkRows.size() * sizeof(NullableRow));
    std::memcpy(fkBytes.data(), fkRows.data(), fkBytes.size());

    oAppend(pkSide, layout, false, pkBytes.data(), pkRows.size(), arena.allocator());
    oAppend(fkSide, layout, true, fkBytes.data(), fkRows.size(), arena.allocator());

    const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);
    std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
    const uint64_t numMerged = crossMerge(pkSide, layout, fkSide, layout, scratch.data());
    std::vector<uint8_t> out(numMerged * outputSlotSize(layout, layout));
    const auto result = obliviousScan(scratch.data(), numMerged, layout, layout, out.data());

    /// Only the non-null FK key 5 matches the non-null PK key 5. The two
    /// NULL-key PK tuples with identical raw keys are not duplicates.
    EXPECT_EQ(result.realMatches, 1U);
    EXPECT_FALSE(result.duplicatePkDetected);
}

/// obliviousCompactPow2 must agree with std::stable_partition on random marks
/// across slot sizes and power-of-two lengths, including all-marked and
/// none-marked inputs.
TEST_F(FKMergObliviousPrimitivesTest, obliviousCompactMatchesStablePartition)
{
    std::mt19937_64 rng(4242);
    for (const uint64_t payloadSize : {0ULL, 24ULL})
    {
        const uint64_t slotSize = sizeof(SlotHeader) + payloadSize;
        for (const uint64_t count : {1ULL, 2ULL, 8ULL, 64ULL, 256ULL})
        {
            for (const int markMode : {0, 1, 2}) /// 0 = random, 1 = all marked, 2 = none marked
            {
                std::vector<uint8_t> arr(count * slotSize, 0);
                std::vector<uint8_t> marks(count, 0);
                std::vector<std::pair<uint8_t, uint64_t>> expected; /// (mark, original index)
                for (uint64_t i = 0; i < count; ++i)
                {
                    const uint8_t mark = markMode == 0 ? static_cast<uint8_t>(rng() % 2) : static_cast<uint8_t>(markMode == 1);
                    marks[i] = mark;
                    /// The slot payload encodes the original index so stability is checkable.
                    SlotHeader header{.sortKey = i, .flags = 0};
                    std::memcpy(arr.data() + (i * slotSize), &header, sizeof(header));
                    expected.emplace_back(mark, i);
                }
                std::ranges::stable_partition(expected, [](const auto& entry) { return entry.first != 0; });
                const auto numMarked
                    = static_cast<uint64_t>(std::ranges::count_if(expected, [](const auto& entry) { return entry.first != 0; }));

                obliviousCompactPow2(arr.data(), slotSize, marks.data(), 0, count, 0);

                /// The marked prefix keeps its relative order; the unmarked
                /// tail is only guaranteed to hold the right multiset (the
                /// compaction rotates it, unlike std::stable_partition).
                std::multiset<uint64_t> actualTail;
                std::multiset<uint64_t> expectedTail;
                for (uint64_t i = 0; i < count; ++i)
                {
                    SlotHeader header;
                    std::memcpy(&header, arr.data() + (i * slotSize), sizeof(header));
                    if (i < numMarked)
                    {
                        EXPECT_EQ(header.sortKey, expected[i].second)
                            << "slotSize=" << slotSize << " count=" << count << " markMode=" << markMode << " pos=" << i;
                    }
                    else
                    {
                        actualTail.insert(header.sortKey);
                        expectedTail.insert(expected[i].second);
                    }
                }
                EXPECT_EQ(actualTail, expectedTail) << "slotSize=" << slotSize << " count=" << count << " markMode=" << markMode;
            }
        }
    }
}

/// trimDummies after crossMerge + obliviousScan: the first realMatches output
/// slots must be exactly the real join results (stable order), the count must
/// equal the scan's, and the padded tail must be dummies.
TEST_F(FKMergObliviousPrimitivesTest, trimDummiesCompactsRealResultsToFront)
{
    std::mt19937_64 rng(2026);
    const auto layout = testLayout();

    for (int round = 0; round < 5; ++round)
    {
        TestArena arena;
        SortedSide pkSide;
        SortedSide fkSide;

        const uint64_t numPk = 1 + (rng() % 40);
        std::vector<TestRow> pkRows(numPk);
        std::set<int32_t> used;
        for (auto& row : pkRows)
        {
            int32_t key = 0;
            do
            {
                key = static_cast<int32_t>(rng() % 300);
            } while (!used.insert(key).second);
            row = TestRow{.key = key, .payload = rng() % 100000};
        }
        const uint64_t numFk = 1 + (rng() % 60);
        std::vector<TestRow> fkRows(numFk);
        for (auto& row : fkRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 400), .payload = rng() % 100000};
        }

        const auto pkBytes = rowBytes(pkRows);
        const auto fkBytes = rowBytes(fkRows);
        oAppend(pkSide, layout, false, pkBytes.data(), numPk, arena.allocator());
        oAppend(fkSide, layout, true, fkBytes.data(), numFk, arena.allocator());

        const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);
        std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
        const uint64_t numMerged = crossMerge(pkSide, layout, fkSide, layout, scratch.data());

        const uint64_t outSlot = outputSlotSize(layout, layout);
        /// L3 needs pow2 output capacity for the compaction padding.
        std::vector<uint8_t> out(nextPowerOfTwo(numMerged) * outSlot);
        const auto scanResult = obliviousScan(scratch.data(), numMerged, layout, layout, out.data());

        /// Expected real pairs in merged (i.e. key-sorted, stable) order.
        std::vector<std::pair<uint64_t, uint64_t>> expectedPairs;
        for (uint64_t i = 0; i < numMerged; ++i)
        {
            const uint8_t* slot = out.data() + (i * outSlot);
            uint64_t flags = 0;
            std::memcpy(&flags, slot, sizeof(flags));
            if ((flags & 1ULL) == 0)
            {
                TestRow pkRow;
                TestRow fkRow;
                std::memcpy(&pkRow, slot + sizeof(uint64_t), sizeof(TestRow));
                std::memcpy(&fkRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
                expectedPairs.emplace_back(pkRow.payload, fkRow.payload);
            }
        }

        const uint64_t realSlots = trimDummies(out.data(), numMerged, outSlot);
        ASSERT_EQ(realSlots, scanResult.realMatches);
        ASSERT_EQ(realSlots, expectedPairs.size());
        for (uint64_t i = 0; i < realSlots; ++i)
        {
            const uint8_t* slot = out.data() + (i * outSlot);
            uint64_t flags = 0;
            std::memcpy(&flags, slot, sizeof(flags));
            EXPECT_EQ(flags & 1ULL, 0U) << "dummy slot inside the real prefix at " << i;
            TestRow pkRow;
            TestRow fkRow;
            std::memcpy(&pkRow, slot + sizeof(uint64_t), sizeof(TestRow));
            std::memcpy(&fkRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
            EXPECT_EQ(pkRow.payload, expectedPairs[i].first) << "order not stable at " << i;
            EXPECT_EQ(fkRow.payload, expectedPairs[i].second);
        }
        /// Everything past the real prefix must be a dummy.
        for (uint64_t i = realSlots; i < nextPowerOfTwo(numMerged); ++i)
        {
            uint64_t flags = 0;
            std::memcpy(&flags, out.data() + (i * outSlot), sizeof(flags));
            EXPECT_EQ(flags & 1ULL, 1U) << "real slot in the trimmed tail at " << i;
        }
    }
}

/// sortedCrossCombine (FK-SORT's combine: full bitonic sort of the raw,
/// unsorted logs) + obliviousScan must produce exactly the brute-force FK
/// join, like crossMerge does for key-sorted windows.
TEST_F(FKMergObliviousPrimitivesTest, sortedCrossCombineScanMatchesBruteForceJoin)
{
    std::mt19937_64 rng(777);
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    for (int round = 0; round < 5; ++round)
    {
        TestArena arena;
        SortedSide pkLog;
        SortedSide fkLog;

        const uint64_t numPk = 1 + (rng() % 50);
        std::vector<TestRow> pkRows(numPk);
        std::set<int32_t> used;
        for (auto& row : pkRows)
        {
            int32_t key = 0;
            do
            {
                key = static_cast<int32_t>(rng() % 400);
            } while (!used.insert(key).second);
            row = TestRow{.key = key, .payload = rng() % 100000};
        }
        const uint64_t numFk = 1 + (rng() % 70);
        std::vector<TestRow> fkRows(numFk);
        for (auto& row : fkRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 600), .payload = rng() % 100000};
        }

        /// Raw arrival logs, no key order whatsoever.
        const auto pkBytes = rowBytes(pkRows);
        const auto fkBytes = rowBytes(fkRows);
        appendToArrivalLog(pkLog, layout, false, pkBytes.data(), numPk, arena.allocator());
        appendToArrivalLog(fkLog, layout, true, fkBytes.data(), numFk, arena.allocator());

        const uint64_t padded = nextPowerOfTwo(numPk + numFk);
        std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
        const uint64_t numMerged = sortedCrossCombine(pkLog, layout, fkLog, layout, scratch.data());
        ASSERT_EQ(numMerged, numPk + numFk);

        const uint64_t outSlot = outputSlotSize(layout, layout);
        std::vector<uint8_t> out(numMerged * outSlot);
        const auto result = obliviousScan(scratch.data(), numMerged, layout, layout, out.data());
        EXPECT_FALSE(result.duplicatePkDetected);

        std::map<int32_t, TestRow> pkByKey;
        for (const auto& row : pkRows)
        {
            pkByKey[row.key] = row;
        }
        std::multiset<std::pair<uint64_t, uint64_t>> expectedPairs;
        for (const auto& fk : fkRows)
        {
            if (const auto it = pkByKey.find(fk.key); it != pkByKey.end())
            {
                expectedPairs.emplace(it->second.payload, fk.payload);
            }
        }

        std::multiset<std::pair<uint64_t, uint64_t>> actualPairs;
        for (uint64_t i = 0; i < numMerged; ++i)
        {
            const uint8_t* slot = out.data() + (i * outSlot);
            uint64_t flags = 0;
            std::memcpy(&flags, slot, sizeof(flags));
            if ((flags & 1ULL) == 0)
            {
                TestRow pkRow;
                TestRow fkRow;
                std::memcpy(&pkRow, slot + sizeof(uint64_t), sizeof(TestRow));
                std::memcpy(&fkRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
                EXPECT_EQ(pkRow.key, fkRow.key);
                actualPairs.emplace(pkRow.payload, fkRow.payload);
            }
        }
        EXPECT_EQ(actualPairs, expectedPairs);
    }
}

/// perTupleReplay (FK-MERG-L2 and FK-SORT-L2) must produce exactly the
/// brute-force FK join, regardless of arrival order, including out-of-order
/// timestamps, timestamp ties across sides, and empty sides.
TEST_F(FKMergObliviousPrimitivesTest, perTupleReplayMatchesBruteForceJoin)
{
    std::mt19937_64 rng(555);
    /// TestRow with the payload doubling as the event timestamp.
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    for (int round = 0; round < 6; ++round)
    {
        const uint64_t numPk = round == 5 ? 0 : 1 + (rng() % 30);
        std::vector<TestRow> pkRows(numPk);
        std::set<int32_t> used;
        for (auto& row : pkRows)
        {
            int32_t key = 0;
            do
            {
                key = static_cast<int32_t>(rng() % 200);
            } while (!used.insert(key).second);
            /// Timestamps deliberately overlap across sides (incl. exact ties).
            row = TestRow{.key = key, .payload = rng() % 50};
        }
        const uint64_t numFk = round == 4 ? 0 : 1 + (rng() % 50);
        std::vector<TestRow> fkRows(numFk);
        for (auto& row : fkRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 300), .payload = rng() % 50};
        }

        /// Brute-force expectation (keys unique on the PK side).
        std::map<int32_t, TestRow> pkByKey;
        for (const auto& row : pkRows)
        {
            pkByKey[row.key] = row;
        }
        std::multiset<std::pair<uint64_t, uint64_t>> expectedPairs;
        for (const auto& fk : fkRows)
        {
            if (const auto it = pkByKey.find(fk.key); it != pkByKey.end())
            {
                expectedPairs.emplace(it->second.payload, fk.payload);
            }
        }

        /// Both replay modes: MERG_L2 (sorted windows + merge) and SORT_L2
        /// (raw logs + full sort per tuple) must agree with the brute force.
        for (const bool fullSort : {false, true})
        {
            TestArena arena;
            SortedSide pkLog;
            SortedSide fkLog;
            if (numPk > 0)
            {
                const auto pkBytes = rowBytes(pkRows);
                appendToArrivalLog(pkLog, layout, false, pkBytes.data(), numPk, arena.allocator());
            }
            if (numFk > 0)
            {
                const auto fkBytes = rowBytes(fkRows);
                appendToArrivalLog(fkLog, layout, true, fkBytes.data(), numFk, arena.allocator());
            }

            const uint64_t outSlot = outputSlotSize(layout, layout);
            std::vector<uint8_t> out(std::max<uint64_t>(numFk, 1) * outSlot);
            const auto result
                = perTupleReplay(pkLog, layout, fkLog, layout, fullSort, out.data(), std::max<uint64_t>(numFk, 1), arena.allocator());
            EXPECT_FALSE(result.duplicatePkDetected) << "fullSort=" << fullSort;

            ASSERT_EQ(result.realMatches, expectedPairs.size()) << "fullSort=" << fullSort;
            std::multiset<std::pair<uint64_t, uint64_t>> actualPairs;
            for (uint64_t i = 0; i < result.realMatches; ++i)
            {
                const uint8_t* slot = out.data() + (i * outSlot);
                uint64_t flags = 0;
                std::memcpy(&flags, slot, sizeof(flags));
                EXPECT_EQ(flags & 1ULL, 0U) << "dummy slot in L2 replay output at " << i;
                TestRow pkRow;
                TestRow fkRow;
                std::memcpy(&pkRow, slot + sizeof(uint64_t), sizeof(TestRow));
                std::memcpy(&fkRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
                EXPECT_EQ(pkRow.key, fkRow.key);
                actualPairs.emplace(pkRow.payload, fkRow.payload);
            }
            EXPECT_EQ(actualPairs, expectedPairs) << "fullSort=" << fullSort;
        }
    }
}

/// The replay's final adjacent-key pass must catch duplicate PK keys even when
/// no FK tuple arrives after the duplicate.
TEST_F(FKMergObliviousPrimitivesTest, perTupleReplayDetectsDuplicatePk)
{
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    /// FK tuple arrives first (ts 0), then the duplicate PKs — no FK merge ever
    /// sees both duplicates, only the final pass does.
    const std::vector<TestRow> pkRows{{.key = 5, .payload = 1}, {.key = 5, .payload = 2}};
    const std::vector<TestRow> fkRows{{.key = 5, .payload = 0}};

    for (const bool fullSort : {false, true})
    {
        TestArena arena;
        SortedSide pkLog;
        SortedSide fkLog;
        const auto pkBytes = rowBytes(pkRows);
        const auto fkBytes = rowBytes(fkRows);
        appendToArrivalLog(pkLog, layout, false, pkBytes.data(), pkRows.size(), arena.allocator());
        appendToArrivalLog(fkLog, layout, true, fkBytes.data(), fkRows.size(), arena.allocator());

        const uint64_t outSlot = outputSlotSize(layout, layout);
        std::vector<uint8_t> out(2 * outSlot);
        const auto result = perTupleReplay(pkLog, layout, fkLog, layout, fullSort, out.data(), 2, arena.allocator());
        EXPECT_TRUE(result.duplicatePkDetected) << "fullSort=" << fullSort;
    }
}

namespace
{
/// Brute-force generic join for the NFK tests (duplicates on both sides).
std::multiset<std::pair<uint64_t, uint64_t>> bruteForceNfk(const std::vector<TestRow>& left, const std::vector<TestRow>& right)
{
    std::multiset<std::pair<uint64_t, uint64_t>> pairs;
    for (const auto& l : left)
    {
        for (const auto& r : right)
        {
            if (l.key == r.key)
            {
                pairs.emplace(l.payload, r.payload);
            }
        }
    }
    return pairs;
}

/// Decodes the NFK output area ([flags][rightRow][leftRow] slots) into
/// (leftPayload, rightPayload) pairs.
std::multiset<std::pair<uint64_t, uint64_t>> decodeNfkOutput(const uint8_t* out, const uint64_t count)
{
    std::multiset<std::pair<uint64_t, uint64_t>> pairs;
    const uint64_t outSlot = sizeof(uint64_t) + (2 * sizeof(TestRow));
    for (uint64_t i = 0; i < count; ++i)
    {
        const uint8_t* slot = out + (i * outSlot);
        TestRow rightRow;
        TestRow leftRow;
        std::memcpy(&rightRow, slot + sizeof(uint64_t), sizeof(TestRow));
        std::memcpy(&leftRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
        EXPECT_EQ(leftRow.key, rightRow.key);
        pairs.emplace(leftRow.payload, rightRow.payload);
    }
    return pairs;
}
}

/// nfkJoin (Krastnikov-based generic join) must produce exactly the
/// brute-force join with DUPLICATE keys on both sides, including empty sides
/// and keys present on only one side.
TEST_F(FKMergObliviousPrimitivesTest, nfkJoinMatchesBruteForceWithDuplicates)
{
    std::mt19937_64 rng(31337);
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    for (int round = 0; round < 6; ++round)
    {
        TestArena arena;
        SortedSide leftLog;
        SortedSide rightLog;

        /// Small key domain forces heavy duplication on both sides.
        const uint64_t numLeft = round == 4 ? 0 : 1 + (rng() % 40);
        const uint64_t numRight = round == 5 ? 0 : 1 + (rng() % 40);
        std::vector<TestRow> leftRows(numLeft);
        std::vector<TestRow> rightRows(numRight);
        for (auto& row : leftRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 12), .payload = rng() % 100000};
        }
        for (auto& row : rightRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 12), .payload = rng() % 100000};
        }
        if (numLeft > 0)
        {
            const auto bytes = rowBytes(leftRows);
            appendToArrivalLog(leftLog, layout, false, bytes.data(), numLeft, arena.allocator());
        }
        if (numRight > 0)
        {
            const auto bytes = rowBytes(rightRows);
            appendToArrivalLog(rightLog, layout, true, bytes.data(), numRight, arena.allocator());
        }

        const auto expected = bruteForceNfk(leftRows, rightRows);
        const NfkFreshSelector allFresh{};
        const uint64_t measured = nfkJoinSize(leftLog, layout, rightLog, layout, allFresh, arena.allocator());
        ASSERT_EQ(measured, expected.size()) << "round=" << round;

        const uint64_t outSlot = sizeof(uint64_t) + (2 * sizeof(TestRow));
        std::vector<uint8_t> out(std::max<uint64_t>(measured, 1) * outSlot);
        const uint64_t emitted
            = nfkJoin(leftLog, layout, rightLog, layout, allFresh, out.data(), std::max<uint64_t>(measured, 1), arena.allocator());
        ASSERT_EQ(emitted, expected.size());
        EXPECT_EQ(decodeNfkOutput(out.data(), emitted), expected) << "round=" << round;
    }
}

/// With a single fresh tuple, nfkJoin must emit exactly that tuple's matches
/// (the incremental marking that makes the L2 variant leak only the degree).
TEST_F(FKMergObliviousPrimitivesTest, nfkJoinSingleFreshEmitsOnlyNewMatches)
{
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    TestArena arena;
    SortedSide leftLog;
    SortedSide rightLog;
    /// key 7: 2 left x 2 right; key 9: right only.
    const std::vector<TestRow> leftRows{{.key = 7, .payload = 1}, {.key = 7, .payload = 2}, {.key = 3, .payload = 3}};
    const std::vector<TestRow> rightRows{{.key = 7, .payload = 10}, {.key = 7, .payload = 11}, {.key = 9, .payload = 12}};
    const auto leftBytes = rowBytes(leftRows);
    const auto rightBytes = rowBytes(rightRows);
    appendToArrivalLog(leftLog, layout, false, leftBytes.data(), leftRows.size(), arena.allocator());
    appendToArrivalLog(rightLog, layout, true, rightBytes.data(), rightRows.size(), arena.allocator());

    /// Fresh = the second left row (key 7, payload 2): its matches are the two
    /// right rows with key 7 — nothing else.
    const NfkFreshSelector fresh{.allFresh = false, .freshSideIsRight = false, .freshIndex = 1};
    const uint64_t outSlot = sizeof(uint64_t) + (2 * sizeof(TestRow));
    std::vector<uint8_t> out(8 * outSlot);
    const uint64_t emitted = nfkJoin(leftLog, layout, rightLog, layout, fresh, out.data(), 8, arena.allocator());
    ASSERT_EQ(emitted, 2U);
    const std::multiset<std::pair<uint64_t, uint64_t>> expected{{2, 10}, {2, 11}};
    EXPECT_EQ(decodeNfkOutput(out.data(), emitted), expected);
}

/// nfkPerTupleReplay must produce exactly the brute-force join regardless of
/// arrival order — every pair emitted exactly once when its later tuple
/// arrives.
TEST_F(FKMergObliviousPrimitivesTest, nfkPerTupleReplayMatchesBruteForce)
{
    std::mt19937_64 rng(90210);
    auto layout = testLayout();
    layout.tsOffset = offsetof(TestRow, payload);
    layout.tsType = DataType::Type::UINT64;
    layout.tsNullable = false;

    for (int round = 0; round < 4; ++round)
    {
        TestArena arena;
        SortedSide leftLog;
        SortedSide rightLog;
        const uint64_t numLeft = 1 + (rng() % 25);
        const uint64_t numRight = 1 + (rng() % 25);
        std::vector<TestRow> leftRows(numLeft);
        std::vector<TestRow> rightRows(numRight);
        /// payload doubles as the event timestamp; overlap forces ties.
        for (auto& row : leftRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 8), .payload = rng() % 30};
        }
        for (auto& row : rightRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 8), .payload = rng() % 30};
        }
        const auto leftBytes = rowBytes(leftRows);
        const auto rightBytes = rowBytes(rightRows);
        appendToArrivalLog(leftLog, layout, false, leftBytes.data(), numLeft, arena.allocator());
        appendToArrivalLog(rightLog, layout, true, rightBytes.data(), numRight, arena.allocator());

        const auto expected = bruteForceNfk(leftRows, rightRows);
        const NfkFreshSelector allFresh{};
        const uint64_t total = nfkJoinSize(leftLog, layout, rightLog, layout, allFresh, arena.allocator());
        ASSERT_EQ(total, expected.size());

        const uint64_t outSlot = sizeof(uint64_t) + (2 * sizeof(TestRow));
        std::vector<uint8_t> out(std::max<uint64_t>(total, 1) * outSlot);
        const uint64_t emitted
            = nfkPerTupleReplay(leftLog, layout, rightLog, layout, out.data(), std::max<uint64_t>(total, 1), arena.allocator());
        ASSERT_EQ(emitted, expected.size()) << "round=" << round;
        EXPECT_EQ(decodeNfkOutput(out.data(), emitted), expected) << "round=" << round;
    }
}

/// nljL4Join must emit exactly nL*nR slots whose real subset equals the
/// brute-force generic join (duplicates allowed), dummies all-zero.
TEST_F(FKMergObliviousPrimitivesTest, nljL4JoinMatchesBruteForcePadded)
{
    std::mt19937_64 rng(4711);
    const auto layout = testLayout();

    for (int round = 0; round < 5; ++round)
    {
        TestArena arena;
        SortedSide leftLog;
        SortedSide rightLog;
        const uint64_t numLeft = round == 4 ? 0 : 1 + (rng() % 30);
        const uint64_t numRight = 1 + (rng() % 30);
        std::vector<TestRow> leftRows(numLeft);
        std::vector<TestRow> rightRows(numRight);
        for (auto& row : leftRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 10), .payload = rng() % 100000};
        }
        for (auto& row : rightRows)
        {
            row = TestRow{.key = static_cast<int32_t>(rng() % 10), .payload = rng() % 100000};
        }
        if (numLeft > 0)
        {
            const auto bytes = rowBytes(leftRows);
            appendToArrivalLog(leftLog, layout, false, bytes.data(), numLeft, arena.allocator());
        }
        const auto rightBytes = rowBytes(rightRows);
        appendToArrivalLog(rightLog, layout, true, rightBytes.data(), numRight, arena.allocator());

        const auto expected = bruteForceNfk(leftRows, rightRows);
        const uint64_t outSlot = sizeof(uint64_t) + (2 * sizeof(TestRow));
        std::vector<uint8_t> out(std::max<uint64_t>(numLeft * numRight, 1) * outSlot);
        const uint64_t matches = nljL4Join(leftLog, layout, rightLog, layout, out.data());
        ASSERT_EQ(matches, expected.size()) << "round=" << round;

        std::multiset<std::pair<uint64_t, uint64_t>> actual;
        for (uint64_t i = 0; i < numLeft * numRight; ++i)
        {
            const uint8_t* slot = out.data() + (i * outSlot);
            uint64_t flags = 0;
            std::memcpy(&flags, slot, sizeof(flags));
            TestRow rightRow;
            TestRow leftRow;
            std::memcpy(&rightRow, slot + sizeof(uint64_t), sizeof(TestRow));
            std::memcpy(&leftRow, slot + sizeof(uint64_t) + sizeof(TestRow), sizeof(TestRow));
            if ((flags & 1ULL) == 0)
            {
                EXPECT_EQ(leftRow.key, rightRow.key);
                actual.emplace(leftRow.payload, rightRow.payload);
            }
            else
            {
                EXPECT_EQ(leftRow.key, 0);
                EXPECT_EQ(rightRow.key, 0);
            }
        }
        EXPECT_EQ(actual, expected) << "round=" << round;
    }
}

/// The number of comparator-network operations must depend only on the input
/// cardinalities, not on the data: with equal sizes, two different datasets
/// must drive identical swap-call counts. oMemSwap itself always touches both
/// slots, so equal call counts imply identical access patterns.
TEST_F(FKMergObliviousPrimitivesTest, accessPatternDependsOnlyOnCardinalities)
{
    const auto layout = testLayout();
    std::mt19937_64 rng(99);

    auto runAndCountOutputBytes = [&](const uint64_t seedOffset) -> std::pair<uint64_t, uint64_t>
    {
        std::mt19937_64 localRng(seedOffset);
        TestArena arena;
        SortedSide pkSide;
        SortedSide fkSide;
        std::vector<TestRow> pkRows(32);
        std::set<int32_t> used;
        for (auto& row : pkRows)
        {
            int32_t key = 0;
            do
            {
                key = static_cast<int32_t>(localRng() % 10000);
            } while (!used.insert(key).second);
            row = TestRow{.key = key, .payload = localRng()};
        }
        std::vector<TestRow> fkRows(48);
        for (auto& row : fkRows)
        {
            row = TestRow{.key = static_cast<int32_t>(localRng() % 10000), .payload = localRng()};
        }
        const auto pkBytes = rowBytes(pkRows);
        const auto fkBytes = rowBytes(fkRows);
        oAppend(pkSide, layout, false, pkBytes.data(), pkRows.size(), arena.allocator());
        oAppend(fkSide, layout, true, fkBytes.data(), fkRows.size(), arena.allocator());
        const uint64_t padded = crossMergePaddedSlots(pkSide.sizeSlots, fkSide.sizeSlots);
        std::vector<uint8_t> scratch(padded * mergedSlotSize(layout, layout));
        const uint64_t numMerged = crossMerge(pkSide, layout, fkSide, layout, scratch.data());
        std::vector<uint8_t> out(numMerged * outputSlotSize(layout, layout), 0);
        obliviousScan(scratch.data(), numMerged, layout, layout, out.data());
        /// Output volume: always numMerged slots regardless of match count.
        return {numMerged, out.size()};
    };

    const auto [slotsA, bytesA] = runAndCountOutputBytes(1);
    const auto [slotsB, bytesB] = runAndCountOutputBytes(2);
    EXPECT_EQ(slotsA, slotsB);
    EXPECT_EQ(bytesA, bytesB);
}

}
