#include "l2_perception/armor/armor_decoder.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t kFieldCount = 22;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

float logit(float probability)
{
  return std::log(probability / (1.0F - probability));
}

bool near(float actual, float expected, float tolerance = 1e-4F)
{
  return std::abs(actual - expected) <= tolerance;
}

void setCandidate(std::vector<float>& values, std::size_t candidate,
                  const std::array<cv::Point2f, 4>& raw_corners, float confidence,
                  std::size_t color, std::size_t armor_class, float best_class_score)
{
  const std::size_t row = candidate * kFieldCount;
  for (std::size_t corner = 0; corner < raw_corners.size(); ++corner) {
    values[row + corner * 2] = raw_corners[corner].x;
    values[row + corner * 2 + 1] = raw_corners[corner].y;
  }
  values[row + 8] = logit(confidence);

  for (std::size_t index = 0; index < 4; ++index) {
    values[row + 9 + index] = -5.0F;
  }
  values[row + 9 + color] = 5.0F;

  for (std::size_t index = 0; index < 9; ++index) {
    values[row + 13 + index] = -5.0F;
  }
  values[row + 13 + armor_class] = best_class_score;
}

// YOLOV8-21 契约：[1, 21, 6300] channels-first，行 0~3 颜色、4~12 九类、
// 13~20 四角点，没有 objectness 通道，类别分数已过 sigmoid。
constexpr std::size_t kV8FieldCount = 21;

void setV8Candidate(std::vector<float>& values, std::size_t candidate_count, std::size_t candidate,
                    const std::array<cv::Point2f, 4>& raw_corners, float class_score,
                    std::size_t color, std::size_t armor_class)
{
  const auto at = [&](std::size_t field) -> float& {
    return values[field * candidate_count + candidate];
  };
  for (std::size_t corner = 0; corner < raw_corners.size(); ++corner) {
    at(13 + corner * 2) = raw_corners[corner].x;
    at(13 + corner * 2 + 1) = raw_corners[corner].y;
  }
  for (std::size_t index = 0; index < 4; ++index) {
    at(index) = 0.01F;
  }
  at(color) = 0.9F;
  for (std::size_t index = 0; index < 9; ++index) {
    at(4 + index) = 0.01F;
  }
  at(4 + armor_class) = class_score;
}

void checkYolov8Layout()
{
  constexpr std::size_t candidate_count = 3;
  std::vector<float> values(kV8FieldCount * candidate_count, 0.0F);

  // 置信度只能来自类别分支：0.93 通过 0.5 门限。
  setV8Candidate(values, candidate_count, 0,
                 {{{100.0F, 100.0F}, {100.0F, 200.0F}, {300.0F, 200.0F}, {300.0F, 100.0F}}}, 0.93F,
                 1, 6);
  // 低于 0.5 门限，必须被初筛丢掉。
  setV8Candidate(values, candidate_count, 1,
                 {{{600.0F, 100.0F}, {600.0F, 200.0F}, {700.0F, 200.0F}, {700.0F, 100.0F}}}, 0.30F,
                 0, 2);
  // 与候选 0 不重叠，应当保留。
  setV8Candidate(values, candidate_count, 2,
                 {{{400.0F, 300.0F}, {400.0F, 350.0F}, {500.0F, 350.0F}, {500.0F, 300.0F}}}, 0.80F,
                 0, 1);

  L2Perception::InferenceTensor tensor;
  tensor.name = "output0";
  tensor.shape = {1, kV8FieldCount, candidate_count};
  tensor.setOwnedData(std::move(values));
  L2Perception::InferenceResult result;
  result.outputs.push_back(std::move(tensor));

  const L2Perception::ImageTransform transform{.source_size = {1280, 960},
                                               .model_size = {640, 480},
                                               .source_to_model_scale = 0.5F,
                                               .pad_left = 0,
                                               .pad_top = 0,
                                               .pad_right = 0,
                                               .pad_bottom = 0};

  const L2Perception::ArmorDecoder decoder(L2Perception::yolov8_21DecoderConfig());
  const auto detections = decoder.decode(result, transform);
  require(detections.size() == 2, "YOLOV8-21 layout did not retain exactly two candidates");

  const auto& outpost = detections[0];
  require(outpost.class_id == 6, "YOLOV8-21 class argmax is wrong");
  require(outpost.color == L2Perception::ArmorColor::Red, "YOLOV8-21 color index 1 must be red");
  // 没有 objectness 通道，confidence 必须等于类别分支最大值本身，且不再过 sigmoid。
  require(near(outpost.confidence, 0.93F), "YOLOV8-21 confidence must be the max class score");
  require(near(outpost.corners[0].x, 200.0F) && near(outpost.corners[0].y, 200.0F),
          "YOLOV8-21 corner offset 13 or inverse transform is wrong");
  require(near(outpost.corners[1].x, 600.0F) && near(outpost.corners[1].y, 200.0F),
          "YOLOV8-21 model point 3 must become Armor right-top");

  require(detections[1].class_id == 1, "YOLOV8-21 second class argmax is wrong");
  require(detections[1].color == L2Perception::ArmorColor::Blue,
          "YOLOV8-21 color index 0 must be blue");
}

}  // namespace

int main()
{
  try {
    constexpr std::size_t candidate_count = 4;
    std::vector<float> values(candidate_count * kFieldCount, 0.0F);

    // 模型原始角点顺序为：左上、左下、右下、右上。
    setCandidate(values, 0,
                 {{{100.0F, 100.0F}, {100.0F, 200.0F}, {300.0F, 200.0F}, {300.0F, 100.0F}}}, 0.95F,
                 0, 3, -4.0F);
    // 与候选 0 高度重叠；虽然类别分数更高，但 SP 规则应按 objectness 抑制它。
    setCandidate(values, 1,
                 {{{105.0F, 105.0F}, {105.0F, 205.0F}, {305.0F, 205.0F}, {305.0F, 105.0F}}}, 0.90F,
                 1, 4, 10.0F);
    // 独立的红色英雄装甲。
    setCandidate(values, 2,
                 {{{400.0F, 300.0F}, {400.0F, 350.0F}, {500.0F, 350.0F}, {500.0F, 300.0F}}}, 0.88F,
                 1, 1, 3.0F);
    // 通过 0.7 初筛但没有严格超过 SP demo 的 0.8 最终门限。
    setCandidate(values, 3, {{{20.0F, 20.0F}, {20.0F, 40.0F}, {60.0F, 40.0F}, {60.0F, 20.0F}}},
                 0.75F, 0, 0, 4.0F);

    L2Perception::InferenceTensor tensor;
    tensor.name = "output";
    tensor.shape = {1, candidate_count, kFieldCount};
    tensor.setOwnedData(std::move(values));
    L2Perception::InferenceResult result;
    result.outputs.push_back(std::move(tensor));

    const L2Perception::ImageTransform transform{.source_size = {1280, 960},
                                                 .model_size = {640, 640},
                                                 .source_to_model_scale = 0.5F,
                                                 .pad_left = 10,
                                                 .pad_top = 20,
                                                 .pad_right = 0,
                                                 .pad_bottom = 0};

    const L2Perception::ArmorDecoder decoder;
    const auto detections = decoder.decode(result, transform);
    require(detections.size() == 2, "SP thresholds/NMS did not retain exactly two candidates");

    const auto& blue = detections[0];
    require(blue.color == L2Perception::ArmorColor::Blue, "SP color index 0 must map to blue");
    require(blue.class_id == 3, "class argmax was not preserved");
    require(near(blue.confidence, 0.95F), "objectness sigmoid was not preserved");
    require(near(blue.corners[0].x, 180.0F) && near(blue.corners[0].y, 160.0F),
            "left-top point or inverse letterbox transform is wrong");
    require(near(blue.corners[1].x, 580.0F) && near(blue.corners[1].y, 160.0F),
            "model point 3 must become Armor right-top");
    require(near(blue.corners[2].x, 580.0F) && near(blue.corners[2].y, 360.0F),
            "model point 2 must become Armor right-bottom");
    require(near(blue.corners[3].x, 180.0F) && near(blue.corners[3].y, 360.0F),
            "model point 1 must become Armor left-bottom");
    require(near(blue.center.x, 380.0F) && near(blue.center.y, 260.0F),
            "Armor center was not derived from source-image corners");

    const auto& red = detections[1];
    require(red.color == L2Perception::ArmorColor::Red, "SP color index 1 must map to red");
    require(red.class_id == 1, "second class argmax is wrong");

    bool rejected_invalid_order = false;
    try {
      L2Perception::ArmorDecoderConfig bad_config;
      bad_config.contract.corner_order = {0, 0, 2, 3};
      const L2Perception::ArmorDecoder bad_decoder(bad_config);
      (void)bad_decoder;
    } catch (const std::invalid_argument&) {
      rejected_invalid_order = true;
    }
    require(rejected_invalid_order, "invalid corner-order configuration was accepted");

    // 第二套契约：深大 Infantry-v8n。与上面共用同一个 Decoder，只换配置。
    checkYolov8Layout();

    require(!L2Perception::armorDecoderPreset("yolov5"),
            "unknown decoder preset name was accepted");

    // 离线工具靠输出名认契约，两条都要认得出来。
    require(L2Perception::armorDecoderConfigFor({{"output", {1, 25200, 22}}}).contract
              == L2Perception::yolov5_22DecoderConfig().contract,
            "probe did not resolve the yolov5_22 contract");
    require(L2Perception::armorDecoderConfigFor({{"output0", {1, 21, 6300}}}).contract
              == L2Perception::yolov8_21DecoderConfig().contract,
            "probe did not resolve the yolov8_21 contract");

    std::cout << "SP-Vision armor decoder smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SP-Vision armor decoder smoke failed: " << error.what() << '\n';
    return 1;
  }
}
