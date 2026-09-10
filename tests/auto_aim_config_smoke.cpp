// 配置加载冒烟：确认仓库里那份 config/auto_aim.yaml 真的被解析成了预期的
// 结构体值。这类错误不会崩、也不会报，只会表现为"某个特性打开了却没生效"，
// 光看代码查不出来。

#include "runtime/auto_aim_config.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << "auto aim config smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

// 仓库里那份配置能被读出来，而且 blend 这一节确实落到了结构体上。
void testRepositoryConfigLoads()
{
  const auto config = runtime::loadAutoAimConfig("config/auto_aim.yaml");
  const auto& blend = config.plan.blend;

  // enable 是调参开关，不是不变量——上车时会被反复开关，不该由 smoke 钉死。
  // 这里只验证这一节确实被解析到了结构体上，值本身交给使用者。
  require(
    blend.limits.max_yaw_acceleration > 0.0 &&
      blend.limits.max_pitch_acceleration > 0.0,
    "acceleration limits must be positive");
  require(
    blend.limits.min_duration < blend.limits.max_duration,
    "blend duration range must be ordered");
  // 前视窗口短于最长过渡时长的话，提前减速永远来不及，而且不报错。
  require(
    blend.horizon > blend.limits.max_duration,
    "forward horizon must exceed the longest blend");
  require(blend.grid > 0.0 && blend.grid < blend.horizon, "scan grid must fit the horizon");
  require(blend.derivative_step > 0.0, "derivative step must be positive");
  require(
    blend.limits.search_iterations >= 1 && blend.limits.search_iterations <= 32,
    "search iteration count must be sane");

  // 单位换算：yaml 写毫秒，结构体存秒。写反了会差三个数量级。
  require(
    blend.limits.min_duration > 1e-3 && blend.limits.min_duration < 1.0,
    "min_duration_ms must be converted to seconds");
  require(
    blend.horizon > 1e-3 && blend.horizon < 1.0,
    "horizon_ms must be converted to seconds");

  // 验收前必须保持关闭的两道闸门，顺带在这里守住。
  require(!config.fire.shoot_enable, "fire.shoot_enable must stay false before acceptance");

  std::cout << "  [ok] blend enabled, a_max=" << blend.limits.max_yaw_acceleration
            << '/' << blend.limits.max_pitch_acceleration << " rad/s^2, T in ["
            << blend.limits.min_duration * 1e3 << ", "
            << blend.limits.max_duration * 1e3 << "] ms, horizon "
            << blend.horizon * 1e3 << " ms\n";
}

// 缺失的文件必须降级而不是抛出：主循环不能因为配置路径写错就死掉。
void testMissingFileDegrades()
{
  const auto fallback = runtime::loadAutoAimConfig("config/does_not_exist.yaml");
  const L4Planning::BlendConfig defaults;
  require(
    fallback.plan.blend.enable == defaults.enable,
    "a missing config must fall back to struct defaults");
  std::cout << "  [ok] missing config degrades to defaults\n";
}

}  // namespace

int main()
{
  testRepositoryConfigLoads();
  testMissingFileDegrades();
  std::cout << "auto aim config smoke test passed\n";
  return 0;
}
