#pragma once

#include "l4_planning/types.hpp"

#include <memory>
#include <numbers>
#include <optional>
#include <string_view>

namespace L4Planning {

struct Impact {
  double z{0.0};         // 指定水平距离处的落点高度，m
  double fly_time{0.0};  // s
};

struct Launch {
  double pitch{0.0};     // 命中指定高度所需的发射仰角，rad
  double fly_time{0.0};  // s
};

class IBallisticModel {
public:
  virtual ~IBallisticModel() = default;

  IBallisticModel() = default;
  IBallisticModel(const IBallisticModel&) = delete;
  IBallisticModel& operator=(const IBallisticModel&) = delete;
  IBallisticModel(IBallisticModel&&) = delete;
  IBallisticModel& operator=(IBallisticModel&&) = delete;

  [[nodiscard]] virtual std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept = 0;
  [[nodiscard]] virtual std::optional<Launch> launch(
    double range, double height, double v0) const noexcept;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

class VacuumModel final : public IBallisticModel {
public:
  explicit VacuumModel(double gravity = 9.7833) noexcept;

  [[nodiscard]] std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept override;
  [[nodiscard]] std::optional<Launch> launch(
    double range, double height, double v0) const noexcept override;
  [[nodiscard]] std::string_view name() const noexcept override;

private:
  double gravity_;
};

class QuadraticDragModel final : public IBallisticModel {
public:
  QuadraticDragModel(double gravity, double drag_coefficient) noexcept;

  [[nodiscard]] std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept override;
  [[nodiscard]] std::optional<Launch> launch(
    double range, double height, double v0) const noexcept override;
  [[nodiscard]] std::string_view name() const noexcept override;

private:
  [[nodiscard]] double effectiveRange(double range) const noexcept;

  double gravity_;
  double drag_;
};

struct BallisticConfig {
  double gravity{9.7833};
  double drag_coefficient{0.0};
  int max_iterations{20};
  double height_tolerance{5e-3};
  double max_pitch{std::numbers::pi / 2.5};
};

class BallisticSolver {
public:
  explicit BallisticSolver(BallisticConfig config = {});

  // d 为水平距离、h 为目标相对枪口高度、bullet_speed 为枪口速度。
  [[nodiscard]] Ballistic solve(double d, double h, double bullet_speed) const;
  [[nodiscard]] const IBallisticModel& model() const noexcept { return *model_; }

private:
  [[nodiscard]] Ballistic solveByHeightCompensation(
    double d, double h, double bullet_speed) const;

  BallisticConfig config_;
  std::shared_ptr<const IBallisticModel> model_;
};

}  // namespace L4Planning
