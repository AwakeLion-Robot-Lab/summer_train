// 验证 ArmorRefiner 与 SP-Vision Detector::detect(Armor&, image)
// 相同的单目标接口。
#include "l2_perception/armor/armor_refiner.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;
// BGR。蓝灯条，B−R = 80，取自 records/3m_high 实测（灯条核心窗口 B/G/R 中位
// 数 181/123/102，B−R 的 p90 为 83）。原来的 {255, 230, 230} 是一根几乎发白
// 的灯条，B−R 只有 25，在 color_channel_diff 的默认阈值 50 之下——见下面的
// testNearWhiteBarIsNotRefined，那是刻意保留的已知边界。
const cv::Scalar kBrightBar{255, 200, 175};
// 白到几乎没有色差的灯条，B−R = 15，低于 color_diff_threshold。
const cv::Scalar kNearWhiteBar{255, 245, 240};

void drawLightbar(cv::Mat& image, float center_x, float center_y, float length, float width,
                  const cv::Scalar& color = kBrightBar)
{
  const cv::Rect bar(static_cast<int>(center_x - width * 0.5F),
                     static_cast<int>(center_y - length * 0.5F), static_cast<int>(width),
                     static_cast<int>(length));
  cv::rectangle(image, bar & cv::Rect(0, 0, image.cols, image.rows), color, cv::FILLED);
}

L2Perception::Armor makeArmor(float left_x, float right_x, float center_y, float length,
                              float corner_bias = 1.0F)
{
  L2Perception::Armor armor;
  const float half = length * 0.5F;
  armor.corners = {cv::Point2f(left_x + corner_bias, center_y - half + corner_bias),
                   cv::Point2f(right_x - corner_bias, center_y - half + corner_bias),
                   cv::Point2f(right_x - corner_bias, center_y + half - corner_bias),
                   cv::Point2f(left_x + corner_bias, center_y + half - corner_bias)};
  armor.color = L2Perception::ArmorColor::Blue;
  armor.class_id = 3;
  armor.confidence = 0.95F;
  for (const cv::Point2f& corner : armor.corners) {
    armor.center += corner;
  }
  armor.center *= 0.25F;
  return armor;
}

void testSingleArmorInterfaceRefines()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F);

  L2Perception::Armor armor = makeArmor(300.0F, 380.0F, 240.0F, 60.0F);
  const auto network_corners = armor.corners;
  const L2Perception::ArmorRefiner refiner;

  require(refiner.detect(armor, image), "two matching lightbars must refine the armor");
  require(armor.corner_source == L2Perception::CornerSource::Refined,
          "successful traditional detection must mark refined corners");
  require(armor.network_corners == network_corners,
          "successful refinement must preserve the network corners");
  require(armor.corner_shift > 0.0F, "successful refinement must report a corner shift");
  require(std::abs(armor.corners[0].y - 210.0F) < 2.0F,
          "top corner must land on the lightbar endpoint");
  require(std::abs(armor.corners[3].y - 269.0F) < 2.0F,
          "bottom corner must land on the lightbar endpoint");
}

void testFailureKeepsNetworkResult()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F);

  L2Perception::Armor armor = makeArmor(300.0F, 380.0F, 240.0F, 60.0F);
  const L2Perception::Armor before = armor;
  const L2Perception::ArmorRefiner refiner;

  require(!refiner.detect(armor, image), "a single lightbar must not be used for refinement");
  require(armor.corners == before.corners && armor.center == before.center &&
              armor.network_corners == before.network_corners &&
              armor.corner_source == before.corner_source &&
              armor.corner_shift == before.corner_shift,
          "a failed SP-style refinement must keep the network result unchanged");
}

void testBatchCompatibilityInterface()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F);

  std::vector<L2Perception::Armor> armors;
  armors.push_back(makeArmor(300.0F, 380.0F, 240.0F, 60.0F));
  armors.push_back(makeArmor(80.0F, 160.0F, 240.0F, 60.0F));

  std::vector<L2Perception::RefineRecord> records;
  const L2Perception::ArmorRefiner refiner;
  const L2Perception::RefineStats stats = refiner.refine(image, armors, &records);

  require(armors.size() == 2, "SP-style refinement must never erase network detections");
  require(stats.refined == 1 && stats.network_kept == 1,
          "batch statistics must distinguish refined and preserved network "
          "results");
  require(records.size() == 2, "batch compatibility interface must emit one record per armor");
}

void testDisabledKeepsInput()
{
  L2Perception::ArmorRefinerConfig config;
  config.enable = false;
  const L2Perception::ArmorRefiner refiner(config);

  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F);
  L2Perception::Armor armor = makeArmor(300.0F, 380.0F, 240.0F, 60.0F);
  const auto before = armor.corners;

  require(!refiner.detect(armor, image), "a disabled refiner must report no refinement");
  require(armor.corners == before, "a disabled refiner must keep the input unchanged");
}

// color_channel_diff 的已知边界，写成契约而不是留给现场去撞：灯条一旦白到
// B−R 低于 color_diff_threshold，差分图里就没有轮廓，精修静默退回网络角点。
// 不崩、不劣化，但也不再有修正——换相机或把曝光调高时要留意这一条。
//
// 同一张图在灰度底图下是能修的，所以这个用例同时证明差异确实来自底图选择。
void testNearWhiteBarIsNotRefined()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F, kNearWhiteBar);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F, kNearWhiteBar);

  L2Perception::ArmorRefinerConfig diff_config;
  diff_config.color_channel_diff = true;
  L2Perception::Armor armor = makeArmor(300.0F, 380.0F, 240.0F, 60.0F);
  const auto before = armor.corners;
  require(
    !L2Perception::ArmorRefiner(diff_config).detect(armor, image),
    "a near-white bar falls below color_diff_threshold and must not refine");
  require(
    armor.corners == before && armor.corner_source == L2Perception::CornerSource::Network,
    "a failed color-diff refinement must leave the network corners untouched");

  L2Perception::ArmorRefinerConfig gray_config;
  gray_config.color_channel_diff = false;
  L2Perception::Armor gray_armor = makeArmor(300.0F, 380.0F, 240.0F, 60.0F);
  require(
    L2Perception::ArmorRefiner(gray_config).detect(gray_armor, image),
    "the same near-white bar must still refine on the grayscale base image");
}

}  // namespace

int main()
{
  try {
    testSingleArmorInterfaceRefines();
    testFailureKeepsNetworkResult();
    testBatchCompatibilityInterface();
    testDisabledKeepsInput();
    testNearWhiteBarIsNotRefined();
  } catch (const std::exception& error) {
    std::printf("armor refiner smoke failed: %s\n", error.what());
    return 1;
  }

  std::printf("armor refiner smoke passed\n");
  return 0;
}
