#include "rune/f1-observation/keypoint_detector.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class FakeBackend final : public L2Perception::IInferenceBackend {
public:
  explicit FakeBackend(bool channels_first) : channels_first(channels_first) {}
  bool channels_first;
  std::array<std::array<float, 18>, 6> rows{};
  std::vector<std::uint8_t> last_input;
  L2Perception::InferenceInputSpec spec{"images", {1,640,640,3}};
  void load(const L2Perception::InferenceModelConfig&) override {}
  bool ready() const noexcept override { return true; }
  const L2Perception::InferenceInputSpec& inputSpec() const override { return spec; }
  L2Perception::InferenceResult infer(const L2Perception::InferenceInput& input) override {
    check(input.isConsistent() && input.shape == spec.shape, "bad input contract");
    last_input.assign(input.values().begin(), input.values().end());
    L2Perception::InferenceTensor tensor;
    tensor.name = "output0";
    tensor.shape = channels_first ? std::vector<std::size_t>{1,18,rows.size()}
                                 : std::vector<std::size_t>{1,rows.size(),18};
    std::vector<float> values(18 * rows.size());
    for (std::size_t row = 0; row < rows.size(); ++row)
      for (std::size_t channel = 0; channel < 18; ++channel)
        values[channels_first ? channel * rows.size() + row : row * 18 + channel] = rows[row][channel];
    tensor.setOwnedData(std::move(values));
    return {{std::move(tensor)}};
  }
};

void setRow(std::array<float,18>& row, int cls, float confidence, float offset) {
  row[cls] = confidence;
  // Source is 1280x720: scale=0.5, vertical padding=140.
  for (int point = 0; point < 5; ++point) {
    row[3 + point * 3] = (100 + offset + point * 10) * 0.5F;
    row[4 + point * 3] = (200 + point * 20) * 0.5F + 140;
    row[5 + point * 3] = 0.9F;
  }
}

void mockChecks(bool channels_first) {
  auto fake = std::make_unique<FakeBackend>(channels_first);
  auto* backend = fake.get();
  setRow(fake->rows[0], 0, 0.9F, 0);
  setRow(fake->rows[1], 0, 0.8F, 1);  // same leaf, lower quality
  setRow(fake->rows[2], 1, 0.9F, 200);
  setRow(fake->rows[3], 2, 0.9F, 400);
  setRow(fake->rows[4], 0, 0.9F, 600);
  fake->rows[4][5] = 0.1F;  // missing a required point
  setRow(fake->rows[5], 0, 0.9F, 800);
  fake->rows[5][3] = std::numeric_limits<float>::quiet_NaN();
  Rune::OpenVinoKeypointDetector detector(std::move(fake));
  // Non-contiguous image input must also work.
  const cv::Mat canvas(722,1282,CV_8UC3,cv::Scalar(10,20,30));
  const cv::Mat image = canvas(cv::Rect(1,1,1280,720));
  const auto small = detector.detect(image);
  check(small.size() == 2, "small mode, keypoint filtering or NMS failed");
  const auto& leaf = small[0];
  check(leaf.class_id == 0 && leaf.top == cv::Point2f(100,200) &&
        leaf.left == cv::Point2f(110,220) && leaf.right == cv::Point2f(130,260) &&
        leaf.bottom == cv::Point2f(140,280) && leaf.center_r == cv::Point2f(120,240),
        "keypoint order/letterbox inversion failed");
  check(small[1].class_id == 1 && small[1].top.x == 300, "small activation mapping failed");
  check(backend->last_input[0] == 114 && backend->last_input[(140*640)*3] == 10 &&
        backend->last_input[(140*640)*3+2] == 30, "centered BGR letterbox failed");
  const auto big = detector.detect(image, Rune::Mode::Big);
  check(big.size() == 2 && big[1].class_id == 1 && big[1].top.x == 500,
        "big activation mapping failed");
  // Same-location class conflict: upstream NMS runs before mode filtering.
  setRow(backend->rows[0], 2, 0.99F, 0);
  check(detector.detect(image, Rune::Mode::Small).size() == 1,
        "class-independent suppression must precede mode filtering");
  bool rejected = false;
  try { (void)detector.detect(cv::Mat{}); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "empty input accepted");
}
}  // namespace

int main() {
  try {
    mockChecks(true);
    mockChecks(false);
    auto config = Rune::KeypointDetectorConfig::load("config/rune_detector.yaml");
    Rune::OpenVinoKeypointDetector detector(config);
    const cv::Mat blank(720,1280,CV_8UC3,cv::Scalar(0,0,0));
    check(detector.detect(blank, Rune::Mode::Small).empty(), "false detection on blank image");
    check(detector.detect(blank, Rune::Mode::Big).empty(), "repeated inference/lease failed");
    std::cout << "rune_detector_smoke: decoder, mode, NMS, letterbox and real-model inference passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rune_detector_smoke: " << error.what() << '\n';
    return 1;
  }
}
