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

    // 剖面搜索：光晕比核心暗得多（实拍如此），半高门槛切在核心边缘上，端点应
    // 落回核心两端。hint 故意横向偏 3 px、纵向偏 2 px、角度偏 5°。
    {
      const cv::Scalar dim_blue{120, 50, 20};
      const cv::Scalar dim_red{20, 50, 120};
      cv::Mat scene(300, 500, CV_8UC3, cv::Scalar{30, 30, 30});
      drawLight(scene, {100.0F, 100.0F}, 40.0F, 6.0F, 0.0F, dim_blue);
      drawLight(scene, {300.0F, 100.0F}, 40.0F, 6.0F, 20.0F, dim_red);
      // 斜看的侧边灯条：核心宽 2 px、长 30 px。
      drawLight(scene, {100.0F, 220.0F}, 30.0F, 2.0F, 0.0F, dim_blue);

      LightFinderConfig profile;
      profile.search = L2Perception::LightSearch::Profile;
      const auto hintAt = [](cv::Point2f center, float length, float tilt_deg) {
        const float rad = tilt_deg * static_cast<float>(CV_PI) / 180.0F;
        const cv::Point2f half(-std::sin(rad) * length * 0.5F, std::cos(rad) * length * 0.5F);
        return L2Perception::LightHint{center - half, center + half};
      };

      const std::vector<Light> blue_found = L2Perception::searchLights(
        scene, {hintAt({103.0F, 102.0F}, 40.0F, 5.0F)}, profile, ArmorColor::Blue);
      require(blue_found.size() == 1, "profile must find the upright light");
      const Light& found = blue_found.front();
      // fillConvexPoly 填满第 80~120 行，按像素中心约定边缘在 79.5 与 120.5。
      // 合成光晕是一块平台（B=120），半高门槛按峰值定，所以端点会被光晕往外拉
      // 最多约 1 px；实拍光晕是平滑衰减的，这个偏差由回放上的长度偏差去量。
      require(cv::norm(found.top - cv::Point2f{100.0F, 79.5F}) < 1.0,
              "profile top endpoint, got " + std::to_string(found.top.x) + "," +
                std::to_string(found.top.y));
      require(cv::norm(found.bottom - cv::Point2f{100.0F, 120.5F}) < 1.0,
              "profile bottom endpoint, got " + std::to_string(found.bottom.x) + "," +
                std::to_string(found.bottom.y));
      require(found.color == ArmorColor::Blue, "profile light colour");

      const std::vector<Light> red_found = L2Perception::searchLights(
        scene, {hintAt({302.0F, 99.0F}, 38.0F, 15.0F)}, profile, ArmorColor::Red);
      require(red_found.size() == 1, "profile must find the tilted light");
      require(std::abs(red_found.front().tilt_angle_deg - 20.0F) < 2.0F,
              "profile tilted light angle");
      require(cv::norm(red_found.front().center - cv::Point2f{300.0F, 100.0F}) < 1.0,
              "profile tilted light center");

      const std::vector<Light> thin = L2Perception::searchLights(
        scene, {hintAt({102.0F, 221.0F}, 30.0F, 0.0F)}, profile, ArmorColor::Blue);
      require(thin.size() == 1, "profile must find a thin oblique light");
      require(std::abs(thin.front().length - 30.0) < 2.0, "thin light length");

      // 车身倾斜时的侧面板：灯条在图像里斜 55°，超过 max_angle_deg。轮廓法按
      // 图像竖直方向判会丢掉它，剖面搜索按预测方向判（预测偏 5°）要留下。
      cv::Mat tilted_scene(200, 200, CV_8UC3, cv::Scalar{30, 30, 30});
      drawLight(tilted_scene, {100.0F, 100.0F}, 40.0F, 6.0F, 55.0F, dim_blue);
      const std::vector<Light> steep = L2Perception::searchLights(
        tilted_scene, {hintAt({101.0F, 99.0F}, 40.0F, 50.0F)}, profile, ArmorColor::Blue);
      require(steep.size() == 1, "profile must keep a light tilted past max_angle_deg");
      require(std::abs(steep.front().tilt_angle_deg - 55.0F) < 2.0F, "steep light angle");

      require(
        L2Perception::searchLights(
          scene, {hintAt({400.0F, 220.0F}, 40.0F, 0.0F)}, profile, ArmorColor::Blue)
          .empty(),
        "a hint on empty background must find nothing");

      require(
        L2Perception::searchLights(
          scene,
          {hintAt({103.0F, 102.0F}, 40.0F, 5.0F), hintAt({98.0F, 99.0F}, 40.0F, -3.0F)},
          profile, ArmorColor::Blue)
            .size() == 1,
        "two hints on one light must yield one light");

      // 灯条伸出搜索范围（贴着画面边）时不收：端点没收住就不给端点观测。
      cv::Mat edge(120, 120, CV_8UC3, cv::Scalar{30, 30, 30});
      drawLight(edge, {60.0F, 20.0F}, 60.0F, 6.0F, 0.0F, dim_blue);
      require(
        L2Perception::searchLights(
          edge, {hintAt({60.0F, 20.0F}, 60.0F, 0.0F)}, profile, ArmorColor::Blue)
          .empty(),
        "a light cut by the image border must be rejected");
    }

    std::cout << "light detector smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "light detector smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
