// 灯条配对几何门限的 smoke：不需要模型，逐条确认 rm_auto_aim 那几道门限都在起作用。
#include "l2_perception/armor/light_matcher.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using L2Perception::ArmorColor;
using L2Perception::Light;
using L2Perception::LightMatcherConfig;
using L2Perception::matchLights;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// 以中心 (x, y)、长度 length、偏离竖直 tilt_deg 构造一根灯条。
Light makeLight(float x, float y, float length, ArmorColor color, float tilt_deg = 0.0F)
{
  const float tilt = tilt_deg * static_cast<float>(CV_PI) / 180.0F;
  const cv::Point2f half(std::sin(tilt) * length * 0.5F, std::cos(tilt) * length * 0.5F);
  Light light;
  light.center = {x, y};
  light.top = light.center - half;
  light.bottom = light.center + half;
  light.length = length;
  light.tilt_angle_deg = tilt_deg;
  light.color = color;
  return light;
}

}  // namespace

int main()
{
  try {
    const LightMatcherConfig config;

    // 小板：中心相距 2 倍灯条长。输入顺序故意右在前，输出必须按 x 排好左右。
    {
      const std::vector<Light> lights{
        makeLight(160.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue)};
      const auto pairs = matchLights(lights, ArmorColor::Blue, config);
      require(pairs.size() == 1, "small armor should pair once");
      require(pairs.front().left == 1 && pairs.front().right == 0, "left/right must follow image x");
      require(!pairs.front().large, "center distance 2.0 is a small armor");
      require(std::abs(pairs.front().center_distance - 2.0F) < 1e-4F, "center distance in light lengths");
    }

    // 大板：中心相距 4 倍灯条长。
    {
      const std::vector<Light> lights{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Red),
        makeLight(220.0F, 100.0F, 30.0F, ArmorColor::Red)};
      const auto pairs = matchLights(lights, ArmorColor::Red, config);
      require(pairs.size() == 1 && pairs.front().large, "center distance 4.0 is a large armor");
    }

    // 颜色：只配指定颜色；Unknown 时要求两根同色且已知。
    {
      const std::vector<Light> mixed{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Red),
        makeLight(160.0F, 100.0F, 30.0F, ArmorColor::Blue)};
      require(matchLights(mixed, ArmorColor::Blue, config).empty(), "red light must not pair for blue");
      require(matchLights(mixed, ArmorColor::Unknown, config).empty(), "mixed colors must not pair");

      const std::vector<Light> unknown{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Unknown),
        makeLight(160.0F, 100.0F, 30.0F, ArmorColor::Unknown)};
      require(matchLights(unknown, ArmorColor::Unknown, config).empty(), "unknown-color lights must not pair");

      const std::vector<Light> red{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Red),
        makeLight(160.0F, 100.0F, 30.0F, ArmorColor::Red)};
      require(matchLights(red, ArmorColor::Unknown, config).size() == 1, "same known color pairs under Unknown");
    }

    // 中间夹着一根灯条：外侧两根不配，只配相邻的两对。
    {
      const std::vector<Light> lights{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(145.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(190.0F, 100.0F, 30.0F, ArmorColor::Blue)};
      const auto pairs = matchLights(lights, ArmorColor::Blue, config);
      require(pairs.size() == 2, "contained light must block the outer pair");
      for (const auto& pair : pairs) {
        require(!(pair.left == 0 && pair.right == 2), "outer pair must be rejected");
      }
    }

    // 长度比、间距、连线角度三道门限。
    {
      const std::vector<Light> length_ratio{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(160.0F, 100.0F, 18.0F, ArmorColor::Blue)};
      require(matchLights(length_ratio, ArmorColor::Blue, config).empty(), "length ratio 0.6 must fail");

      const std::vector<Light> too_close{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(115.0F, 100.0F, 30.0F, ArmorColor::Blue)};
      require(matchLights(too_close, ArmorColor::Blue, config).empty(), "center distance 0.5 must fail");

      const std::vector<Light> too_far{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(280.0F, 100.0F, 30.0F, ArmorColor::Blue)};
      require(matchLights(too_far, ArmorColor::Blue, config).empty(), "center distance 6.0 must fail");

      const std::vector<Light> steep{
        makeLight(100.0F, 100.0F, 30.0F, ArmorColor::Blue),
        makeLight(140.0F, 140.0F, 30.0F, ArmorColor::Blue)};
      require(matchLights(steep, ArmorColor::Blue, config).empty(), "45 degree connection must fail");
    }

    std::cout << "light matcher smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "light matcher smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
