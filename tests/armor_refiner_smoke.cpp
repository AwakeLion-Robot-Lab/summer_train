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

void testIndependentLightDetection()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  // 用细长椭圆模拟真实灯条的圆角光斑。
  cv::ellipse(
    image, cv::Point(320, 240), cv::Size(5, 34), 8.0, 0.0, 360.0,
    cv::Scalar(255, 20, 20), cv::FILLED);

  L2Perception::ArmorRefinerConfig config;
  config.independent_light_threshold_tolerance = 0.0;
  config.independent_light_binary_threshold = 30.0;
  const L2Perception::ArmorRefiner refiner(config);
  const auto lights = refiner.detectLights(
    image, cv::Rect(250, 160, 140, 160), {},
    L2Perception::ArmorColor::Blue);

  require(lights.size() == 1, "independent detector must return the isolated light");
  require(
    lights.front().color == L2Perception::ArmorColor::Blue,
    "independent detector must preserve BGR color classification");
  require(
    cv::norm(lights.front().top - lights.front().bottom) > 50.0,
    "independent light endpoints must span the observed bar");
  require(
    std::abs(lights.front().center.x - 320.0F) < 3.0F &&
      std::abs(lights.front().center.y - 240.0F) < 3.0F,
    "independent light coordinates must be restored from ROI to the full image");
}

void testIndependentLightRejectsBackgroundArtifacts()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  // 白色反光：灰度阈值会接受，但 HSV 饱和度门限必须拒绝。
  cv::ellipse(
    image, cv::Point(150, 240), cv::Size(5, 34), 4.0, 0.0, 360.0,
    cv::Scalar(255, 255, 255), cv::FILLED);
  // 绿色场地灯：亮度和形状都像灯条，但色相不是敌方蓝色。
  cv::ellipse(
    image, cv::Point(260, 240), cv::Size(5, 34), 4.0, 0.0, 360.0,
    cv::Scalar(20, 255, 20), cv::FILLED);
  // 蓝色横纹：颜色正确，但与竖直方向夹角应被几何门限拒绝。
  cv::ellipse(
    image, cv::Point(370, 240), cv::Size(34, 5), 0.0, 0.0, 360.0,
    cv::Scalar(255, 20, 20), cv::FILLED);
  // 稀疏的蓝色 L 形背景纹理：外接框很大、填充率很低。
  const std::vector<cv::Point> sparse_background{
    {470, 205}, {474, 205}, {474, 270}, {510, 270}, {510, 274},
    {470, 274}};
  cv::fillPoly(
    image, std::vector<std::vector<cv::Point>>{sparse_background},
    cv::Scalar(255, 20, 20));
  // 极细的蓝色背景线：倾角、长度都像灯条，但长宽比超过 dx
  // 代码默认的上限 17。
  cv::rectangle(
    image, cv::Rect(535, 202, 3, 72), cv::Scalar(255, 20, 20), cv::FILLED);
  // 近方形蓝色色块：填充率高，但长宽比不到 dx 默认的 2.4。
  cv::rectangle(
    image, cv::Rect(570, 218, 20, 28), cv::Scalar(255, 20, 20), cv::FILLED);

  L2Perception::ArmorRefinerConfig config;
  config.independent_light_threshold_tolerance = 0.0;
  config.independent_light_binary_threshold = 30.0;
  const L2Perception::ArmorRefiner refiner(config);
  const auto lights = refiner.detectLights(
    image, cv::Rect(80, 150, 530, 180), {},
    L2Perception::ArmorColor::Blue);

  require(
    lights.empty(),
    "white glare, wrong hue, horizontal stripe, sparse texture, thin line and "
    "square patch must be rejected");
}

void testIndependentLightUsesRequestedEnemyColor()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  cv::ellipse(
    image, cv::Point(280, 240), cv::Size(5, 34), 5.0, 0.0, 360.0,
    cv::Scalar(255, 20, 20), cv::FILLED);
  cv::ellipse(
    image, cv::Point(360, 240), cv::Size(5, 34), -5.0, 0.0, 360.0,
    cv::Scalar(20, 20, 255), cv::FILLED);

  L2Perception::ArmorRefinerConfig config;
  config.independent_light_threshold_tolerance = 0.0;
  config.independent_light_binary_threshold = 30.0;
  const L2Perception::ArmorRefiner refiner(config);
  const cv::Rect roi(220, 150, 200, 180);

  const auto blue = refiner.detectLights(
    image, roi, {}, L2Perception::ArmorColor::Blue);
  const auto red = refiner.detectLights(
    image, roi, {}, L2Perception::ArmorColor::Red);
  const auto any = refiner.detectLights(
    image, roi, {}, L2Perception::ArmorColor::Unknown);

  require(
    blue.size() == 1 && blue.front().color == L2Perception::ArmorColor::Blue,
    "blue request must not return the red light");
  require(
    red.size() == 1 && red.front().color == L2Perception::ArmorColor::Red,
    "red request must not return the blue light");
  require(any.size() == 2, "unknown color request must detect both color bands");
}

}  // namespace

int main()
{
  try {
    testSingleArmorInterfaceRefines();
    testFailureKeepsNetworkResult();
    testBatchCompatibilityInterface();
    testDisabledKeepsInput();
    testIndependentLightDetection();
    testIndependentLightRejectsBackgroundArtifacts();
    testIndependentLightUsesRequestedEnemyColor();
  } catch (const std::exception& error) {
    std::printf("armor refiner smoke failed: %s\n", error.what());
    return 1;
  }

  std::printf("armor refiner smoke passed\n");
  return 0;
}
