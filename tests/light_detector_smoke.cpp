// 侧边灯条相关的 smoke：不需要模型，在合成图上画几根灯条，逐条确认端点、形状
// 门限、颜色判定、ROI 限制、两路合并的取舍，以及剔除已检出装甲板自己的灯条。
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
using L2Perception::LightSource;
using L2Perception::mergeLights;

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

// 构造一根竖直灯条，source 保持默认的 Model。
Light modelLight(float x, float y, float length)
{
  Light light;
  light.center = {x, y};
  light.top = {x, y - length * 0.5F};
  light.bottom = {x, y + length * 0.5F};
  light.length = length;
  light.color = ArmorColor::Blue;
  return light;
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
      findLights(image, cv::Rect(0, 0, image.cols, image.rows), config, 1.10);
    require(lights.size() == 2, "expected exactly the two upright lights, got " +
                                  std::to_string(lights.size()));

    const Light* upright = nearest(lights, {100.0F, 100.0F});
    require(upright != nullptr && cv::norm(upright->center - cv::Point2f{100.0F, 100.0F}) < 1.0,
            "upright light center");
    require(std::abs(upright->length - 40.0) < 2.0, "upright light length");
    require(upright->top.y < upright->bottom.y, "top must be above bottom");
    require(upright->tilt_angle_deg < 2.0F, "upright light tilt");
    require(upright->color == ArmorColor::Blue, "blue halo must classify as blue");
    require(upright->source == LightSource::Classic, "finder lights are classic");

    const Light* tilted = nearest(lights, {300.0F, 100.0F});
    require(std::abs(tilted->tilt_angle_deg - 20.0F) < 2.0F, "tilted light angle");
    require(tilted->color == ArmorColor::Red, "red halo must classify as red");

    // ROI 之外的灯条不找；返回坐标仍在原图上。
    {
      const std::vector<Light> in_roi =
        findLights(image, cv::Rect(250, 50, 100, 100), config, 1.10);
      require(in_roi.size() == 1, "roi must limit the search");
      require(cv::norm(in_roi.front().center - cv::Point2f{300.0F, 100.0F}) < 1.0,
              "roi result must be in full-image coordinates");
    }

    // 合并：离传统灯条近的模型灯条丢掉，远的补进来，传统的排在前面。
    {
      std::vector<Light> classic{modelLight(100.0F, 100.0F, 40.0F)};
      classic[0].source = LightSource::Classic;
      const std::vector<Light> model{
        modelLight(104.0F, 101.0F, 42.0F), modelLight(180.0F, 100.0F, 40.0F)};
      const std::vector<Light> merged =
        mergeLights(classic, model, config.merge_radius, config.length_agree);
      require(merged.size() == 2, "duplicate model light must be dropped");
      require(merged[0].source == LightSource::Classic, "agreeing classic light must be kept");
      require(cv::norm(merged[1].center - cv::Point2f{180.0F, 100.0F}) < 1e-3,
              "missing light must come from the model");

      // 同一块板的两根灯条中心至少相距 0.8 倍灯长，不能被判成同一根。
      const std::vector<Light> close{modelLight(133.0F, 100.0F, 40.0F)};
      require(mergeLights(classic, close, config.merge_radius, config.length_agree).size() == 2,
              "a light 0.8 lengths away is a different light");

      // 传统灯条只剩一半长（二值化断开），换成模型的。
      std::vector<Light> broken{modelLight(100.0F, 95.0F, 20.0F)};
      broken[0].source = LightSource::Classic;
      const std::vector<Light> fixed =
        mergeLights(broken, {model[0]}, config.merge_radius, config.length_agree);
      require(fixed.size() == 1 && fixed[0].source == LightSource::Model,
              "a broken classic light must be replaced by the model light");
      require(mergeLights(broken, {model[0]}, config.merge_radius, 0.0F)[0].source ==
                LightSource::Classic,
              "length_agree 0 always keeps the classic light");
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
