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
#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/Enums/EnumOption.hpp>
#include <QueryOptimizerNetworkConfiguration.hpp>

namespace NES
{

enum class StreamJoinStrategy : uint8_t
{
    NESTED_LOOP_JOIN,
    HASH_JOIN,
    FK_MERG_L2,
    FK_MERG_L3,
    FK_MERG_L4,
    FK_SORT_L2,
    FK_SORT_L3,
    FK_SORT_L4,
    NFK_JOIN_L2,
    NFK_JOIN_L3,
    OPTIMIZER_CHOOSES
};

class QueryOptimizerConfiguration : public BaseConfiguration
{
public:
    QueryOptimizerConfiguration() = default;
    QueryOptimizerConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    EnumOption<StreamJoinStrategy> joinStrategy
        = {"join_strategy",
           StreamJoinStrategy::OPTIMIZER_CHOOSES,
           "Join Strategy"
           "[NESTED_LOOP_JOIN|HASH_JOIN|FK_MERG_L2|FK_MERG_L3|FK_MERG_L4|FK_SORT_L2|FK_SORT_L3|FK_SORT_L4|NFK_JOIN_L2|NFK_JOIN_L3|"
           "OPTIMIZER_CHOOSES]. FK_MERG_*, FK_SORT_* and NFK_JOIN_* are the oblivious joins; all require an inner single-key "
           "equi-join over a tumbling window. The FK families additionally require unique keys on the right (PK) input; NFK_JOIN "
           "is the generic Krastnikov-based join allowing duplicates on both sides (no L4: its worst case is the Cartesian "
           "product). MERG maintains key-sorted windows (OAppend + bitonic merge); SORT keeps nothing sorted and pays a full "
           "bitonic sort per join (the Opaque-style baseline). L4 pads its output with dummy tuples; L3 leaks only the per-window "
           "output cardinality; L2 processes tuple-at-a-time in event-time order (leaking each tuple's degree) and requires "
           "event-time characteristics."};

    QueryOptimizerNetworkConfiguration network = {"network", "Network configuration overrides for query decomposition"};

private:
    std::vector<BaseOption*> getOptions() override { return {&joinStrategy, &network}; }
};

}
