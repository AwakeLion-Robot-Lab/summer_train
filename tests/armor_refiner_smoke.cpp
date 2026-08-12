// 验证 ArmorRefiner 的单目标接口：矫正成功就替换角点，失败则原样保留网络结果。
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
const cv::Scalar kBrightBar{255, 230, 230};

void drawLightbar(cv::Mat& image, float center_x, float center_y, float length, float width)
{
  const cv::Rect bar(static_cast<int>(center_x - width * 0.5F),
                     static_cast<int>(center_y - length * 0.5F), static_cast<int>(width),
                     static_cast<int>(length));
  cv::rectangle(image, bar & cv::Rect(0, 0, image.cols, image.rows), kBrightBar, cv::FILLED);
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
          "a failed refinement must keep the network result unchanged");
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

  require(armors.size() == 2, "refinement must never erase network detections");
  require(stats.refined == 1 && stats.network_kept == 1 && stats.rejected == 0,
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

}  // namespace

int main()
{
  try {
    testSingleArmorInterfaceRefines();
    testFailureKeepsNetworkResult();
    testBatchCompatibilityInterface();
    testDisabledKeepsInput();
  } catch (const std::exception& error) {
    std::printf("armor refiner smoke failed: %s\n", error.what());
    return 1;
  }

  std::printf("armor refiner smoke passed\n");
  return 0;
}
