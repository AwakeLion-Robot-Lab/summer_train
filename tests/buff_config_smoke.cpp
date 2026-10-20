#include "runtime/buff/buff_config.hpp"

#include <cmath>
#include <iostream>

int main()
{
  const auto config = runtime::buff::loadConfig("config/buff.yaml");
  if (std::abs(config.detector.min_distance - 1.7) > 1e-12 ||
      std::abs(config.detector.max_distance - 5.0) > 1e-12 ||
      std::abs(config.planner.fallback_bullet_speed - 22.5) > 1e-12 ||
      std::abs(config.planner.min_valid_bullet_speed - 14.0) > 1e-12 ||
      config.planner.trust_fallback_bullet_speed ||
      config.control.shoot_enable) {
    std::cerr << "Buff config smoke test failed\n";
    return 1;
  }

  const auto measured = config.planner.resolveBulletSpeed(25.0);
  const auto fallback = config.planner.resolveBulletSpeed(0.0);
  auto trusted_config = config.planner;
  trusted_config.trust_fallback_bullet_speed = true;
  const auto trusted_fallback = trusted_config.resolveBulletSpeed(13.9);
  if (std::abs(measured.value - 25.0) > 1e-12 || !measured.measured ||
      !measured.trusted ||
      std::abs(fallback.value - 22.5) > 1e-12 || fallback.measured ||
      fallback.trusted || trusted_fallback.measured ||
      !trusted_fallback.trusted) {
    std::cerr << "Buff bullet-speed policy smoke test failed\n";
    return 1;
  }
  std::cout << "Buff config smoke test passed\n";
  return 0;
}
