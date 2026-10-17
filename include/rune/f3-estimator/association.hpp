#pragma once

#include <Eigen/Core>
#include <vector>

namespace Rune {
// Minimum-cost one-to-one matching. Every row has its own dummy (rejection)
// column, so even square matrices can reject arbitrary observations.
[[nodiscard]] std::vector<int> hungarianAssign(const Eigen::MatrixXd& costs,
                                              double gate_squared);
} // namespace Rune
