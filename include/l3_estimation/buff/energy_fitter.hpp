#pragma once

#include <deque>
#include <limits>
#include <optional>

namespace L3Estimation::Buff {

class EnergyFitter {
public:
  struct FitResult {
    double C{0.0};
    double v{0.0};
    double a{0.0};
    double omega{0.0};
    double phi{0.0};
    double cost{std::numeric_limits<double>::max()};
  };

  struct LinearResult {
    double C{0.0};
    double speed{0.0};
    double cost{std::numeric_limits<double>::max()};
  };

  void push(double t, double theta);
  void reset() noexcept;
  [[nodiscard]] std::optional<LinearResult> fitLinear() const;
  [[nodiscard]] std::optional<FitResult> fitSine() const;

  static constexpr double kWindowSeconds{6.0};
  static constexpr double kMinFitSeconds{1.5};
  static constexpr double kWeightHalfLifeSeconds{3.0};

  struct Point { double t; double theta; };

private:
  std::deque<Point> buffer_;
};

}  // namespace L3Estimation::Buff
