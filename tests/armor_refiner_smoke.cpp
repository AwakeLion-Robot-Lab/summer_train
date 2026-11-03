// 验证 ArmorRefiner 与 SP-Vision Detector::detect(Armor&, image)
// 相同的单目标接口。
#include "l1_perception/armor/armor_refiner.hpp"
#include "l1_perception/armor/armor_detector.hpp"

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
  require(armor.corners != network_corners, "successful refinement must update corners");
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
              armor.class_id == before.class_id && armor.color == before.color &&
              armor.confidence == before.confidence,
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

class Backend final : public L2Perception::IInferenceBackend {
public:
  bool fail{false};
  void load(const L2Perception::InferenceModelConfig&) override {}
  bool ready() const noexcept override { return true; }
  const L2Perception::InferenceInputSpec& inputSpec() const override { return spec_; }
  L2Perception::InferenceResult infer(const L2Perception::InferenceInput&) override {
    if (fail) throw std::runtime_error("synthetic inference failure");
    // 原图中的板位于 (300, 240) 到 (380, 240)；模型仅看到裁剪 ROI。
    std::vector<float> row{101, 71, 101, 129, 179, 129, 179, 71, 10};
    row.resize(22, -5);
    row[9] = 10;
    row[16] = 10;
    L2Perception::InferenceTensor tensor;
    tensor.name = "output";
    tensor.shape = {1, 1, 22};
    tensor.setOwnedData(std::move(row));
    return {{std::move(tensor)}};
  }
private:
  L2Perception::InferenceInputSpec spec_{"images", {1, 200, 200, 3}};
};

void testDetectorRefinesInOriginalCoordinates()
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F);
  const cv::Rect roi(200, 140, 200, 200);

  L2Perception::ArmorDetectorConfig plain_config;
  plain_config.refiner.enable = false;
  L2Perception::ArmorDetector plain(std::make_unique<Backend>(), plain_config);
  const auto network = plain.detectFrame(image, roi);
  auto backend = std::make_unique<Backend>();
  auto* observed = backend.get();
  L2Perception::ArmorDetector detector(std::move(backend), {});
  const auto result = detector.detectFrame(image, roi);
  require(network.armors.size() == 1 && result.armors.size() == 1 &&
          detector.lastRefineStats().refined == 1,
          "detector must refine a decoded armor after restoring the ROI offset");
  const auto& before = network.armors[0];
  const auto& armor = result.armors[0];
  require(armor.corners != before.corners &&
          cv::norm(armor.corners[0] - cv::Point2f(299.5F, 210.0F)) < 1.0,
          "refined corners must use original-image coordinates");
  require(armor.center == before.center && armor.class_id == before.class_id &&
          armor.network_class_id == before.network_class_id && armor.color == before.color &&
          armor.confidence == before.confidence && result.lights.empty(),
          "refinement must preserve classification, network center and observation interface");

  observed->fail = true;
  const auto failed = detector.detectFrame(image, roi);
  require(failed.armors.empty() && detector.lastRefineStats().refined == 0 &&
          detector.lastRefineStats().network_kept == 0 && detector.lastTiming().refine == 0,
          "inference failure must clear refinement diagnostics");
}

}  // namespace

int main()
{
  try {
    testSingleArmorInterfaceRefines();
    testFailureKeepsNetworkResult();
    testBatchCompatibilityInterface();
    testDisabledKeepsInput();
    testDetectorRefinesInOriginalCoordinates();
  } catch (const std::exception& error) {
    std::printf("armor refiner smoke failed: %s\n", error.what());
    return 1;
  }

  std::printf("armor refiner smoke passed\n");
  return 0;
}
