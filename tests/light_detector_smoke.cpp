// 侧边灯条相关的 smoke：在合成图上画几根灯条，逐条确认端点、形状门限、颜色
// 判定、ROI 限制，以及剔除已检出装甲板自己的灯条。
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/armor/light_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using L2Perception::ArmorColor;
using L2Perception::findLights;
using L2Perception::Light;
using L2Perception::LightFinderConfig;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// 画一根灯条：先填一圈有色光晕，再填发白的核心。和实拍一样，只有核心能过
// 灰度阈值，颜色只留在光晕里。
void drawLight(
  cv::Mat& image, cv::Point2f center, float length, float width, float tilt_deg,
  const cv::Scalar& halo)
{
  const auto fill = [&](float grow, const cv::Scalar& color) {
    const cv::RotatedRect rect(center, {width + grow, length + grow}, tilt_deg);
    cv::Point2f corners[4];
    rect.points(corners);
    std::vector<cv::Point> polygon;
    for (const cv::Point2f& corner : corners) {
      polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::fillConvexPoly(image, polygon, color, cv::LINE_8);
  };
  fill(8.0F, halo);
  fill(0.0F, {255, 255, 255});
}

const Light* nearest(const std::vector<Light>& lights, cv::Point2f point)
{
  const Light* best = nullptr;
  for (const Light& light : lights) {
    if (best == nullptr || cv::norm(light.center - point) < cv::norm(best->center - point)) {
      best = &light;
    }
  }
  return best;
}

}  // namespace

int main()
{
  try {
    // 门限写死，不用默认值：这里要确认的是门限本身在起作用，默认值会随录像
    // 的曝光重新调。
    LightFinderConfig config;
    config.binary_threshold = 140;
    config.max_ratio = 0.4F;
    const cv::Scalar blue{255, 90, 20};
    const cv::Scalar red{20, 60, 255};

    cv::Mat image(400, 600, CV_8UC3, cv::Scalar{30, 30, 30});
    drawLight(image, {100.0F, 100.0F}, 40.0F, 6.0F, 0.0F, blue);
    drawLight(image, {300.0F, 100.0F}, 40.0F, 6.0F, 20.0F, red);
    // 近似正方形的亮斑（弹丸、反光）和太宽的亮块（数字）都不是灯条。
    cv::rectangle(image, cv::Rect(450, 80, 20, 20), {255, 255, 255}, cv::FILLED);
    cv::rectangle(image, cv::Rect(100, 250, 30, 50), {255, 255, 255}, cv::FILLED);
    // 横躺的细条：长宽比像灯条，但倾角不对。
    drawLight(image, {400.0F, 300.0F}, 40.0F, 6.0F, 90.0F, blue);

    const std::vector<Light> lights =
      findLights(image, cv::Rect(0, 0, image.cols, image.rows), config);
    require(lights.size() == 2, "expected exactly the two upright lights, got " +
                                  std::to_string(lights.size()));

    const Light* upright = nearest(lights, {100.0F, 100.0F});
    require(upright != nullptr && cv::norm(upright->center - cv::Point2f{100.0F, 100.0F}) < 1.0,
            "upright light center");
    require(std::abs(upright->length - 40.0) < 2.0, "upright light length");
    require(upright->top.y < upright->bottom.y, "top must be above bottom");
    require(upright->tilt_angle_deg < 2.0F, "upright light tilt");
    require(upright->color == ArmorColor::Blue, "blue halo must classify as blue");

    const Light* tilted = nearest(lights, {300.0F, 100.0F});
    require(std::abs(tilted->tilt_angle_deg - 20.0F) < 2.0F, "tilted light angle");
    require(tilted->color == ArmorColor::Red, "red halo must classify as red");

    // ROI 之外的灯条不找；返回坐标仍在原图上。
    {
      const std::vector<Light> in_roi =
        findLights(image, cv::Rect(250, 50, 100, 100), config);
      require(in_roi.size() == 1, "roi must limit the search");
      require(cv::norm(in_roi.front().center - cv::Point2f{300.0F, 100.0F}) < 1.0,
              "roi result must be in full-image coordinates");
    }

    // insideArmor：板自己的灯条算板的，相邻板的灯条（3 m 处约 4 倍灯长开外）不算。
    {
      L2Perception::Armor armor;
      armor.corners = {
        cv::Point2f{400.0F, 300.0F}, cv::Point2f{500.0F, 300.0F},
        cv::Point2f{500.0F, 340.0F}, cv::Point2f{400.0F, 340.0F}};
      const auto lightAt = [](float x, float y, float length) {
        Light light;
        light.center = {x, y};
        light.top = {x, y - length * 0.5F};
        light.bottom = {x, y + length * 0.5F};
        light.length = length;
        return light;
      };
      constexpr float kMargin = 0.5F;
      require(L2Perception::insideArmor(lightAt(401.0F, 320.0F, 40.0F), armor, kMargin),
              "the armor's own left light must count as inside");
      require(L2Perception::insideArmor(lightAt(510.0F, 320.0F, 40.0F), armor, kMargin),
              "a light within half a length of the box must count as inside");
      require(!L2Perception::insideArmor(lightAt(660.0F, 320.0F, 40.0F), armor, kMargin),
              "a neighbour-plate light four lengths away must stay a side light");
    }

    std::cout << "light detector smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "light detector smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
