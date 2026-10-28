#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l2_perception/buff/types.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <array>
#include <vector>

namespace L2Perception::Buff {

struct DecoderConfig {
  float confidence_threshold{0.5F};
  float nms_iou_threshold{0.45F};
};

// buff.onnx: [1,16,N] = xywh + BR/BW/RR/RW scores + 4*(x,y).
class Decoder {
public:
  explicit Decoder(DecoderConfig config = {});
  [[nodiscard]] std::vector<ModelDetection> decode(
    const InferenceResult& result, const ImageTransform& transform,
    L1Sensor::EnemyColor enemy) const;

private:
  DecoderConfig config_;
};

} // namespace L2Perception::Buff
