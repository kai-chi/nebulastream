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

#include <utility>
#include <LoweringRules/AbstractLoweringRule.hpp>
#include <Operators/LogicalOperator.hpp>
#include <QueryExecutionConfiguration.hpp>

namespace NES
{

/// Lowers a Join with the FK_MERG_L4 implementation trait to the oblivious
/// foreign-key merge join operators. Requirements (validated here, violations
/// throw): inner single-key equi-join, identical integer key types, tumbling
/// window, fixed-size schemas (no VARSIZED). The right input is treated as the
/// PK side; a duplicate PK key within a window fails the query at runtime.
struct LowerToPhysicalFKMergJoin : AbstractLoweringRule
{
    explicit LowerToPhysicalFKMergJoin(QueryExecutionConfiguration conf) : conf(std::move(conf)) { }

    LoweringRuleResultSubgraph apply(LogicalOperator logicalOperator) override;

private:
    QueryExecutionConfiguration conf;
};

}
