// 数字二次分类器的检查：加载核对、抠图几何、以及一块真实装甲板的判定。
//
// 真样本取自仓库里的 tests/data/sp_auto_aim/demo.avi：第 520 帧有一块 4 号板，
// 角点是整板网络在这一帧给出的值（写死在下面），所以这个测试不需要推理后端。

#include "l2_perception/armor/number_classifier.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

// 以灯条长度为单位造一块板的四角点：左上、右上、右下、左下。
std::array<cv::Point2f, 4> makeCorners(float light_length, float center_distance_ratio)
{
  const float half = light_length * 0.5F;
  const float width = light_length * center_distance_ratio;
  return {
    cv::Point2f{100.0F, 100.0F - half}, cv::Point2f{100.0F + width, 100.0F - half},
    cv::Point2f{100.0F + width, 100.0F + half}, cv::Point2f{100.0F, 100.0F + half}};
}

cv::Mat frameOf(const std::string& path, int index)
{
  cv::VideoCapture video(path);
  cv::Mat image;
  for (int i = 0; i <= index; ++i) {
    if (!video.read(image)) {
      return {};
    }
  }
  return image;
}

}  // namespace

int main()
{
  // --- 1. 加载与标签 --------------------------------------------------
  L2Perception::NumberClassifier classifier;
  const L2Perception::NumberClassifierConfig config;
  classifier.load(config);
  expect(classifier.ready(), "默认配置应当能加载 mlp.onnx 和 label.txt");
  expect(classifier.label(0) == "1", "第一行标签应当是 1");
  expect(classifier.label(8) == "negative", "最后一行标签应当是 negative");
  expect(classifier.label(99) == "?", "越界下标应当返回 ?");

  {
    L2Perception::NumberClassifierConfig broken = config;
    broken.model_path = "model/light_model/does_not_exist.onnx";
    bool threw = false;
    try {
      L2Perception::NumberClassifier{}.load(broken);
    } catch (const std::exception&) {
      threw = true;
    }
    expect(threw, "模型文件缺失时 load 必须抛异常");
  }

  {
    // 标签数与模型输出维数对不上时，argmax 的下标会整体错位成另一辆车，
    // 而且不会有任何报错，所以 load 必须挡住。
    const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "newvision_number_labels.txt";
    std::ofstream(path) << "1\n2\n3\n";
    L2Perception::NumberClassifierConfig broken = config;
    broken.label_path = path;
    bool threw = false;
    try {
      L2Perception::NumberClassifier{}.load(broken);
    } catch (const std::exception&) {
      threw = true;
    }
    expect(threw, "标签数与模型输出不一致时 load 必须抛异常");
    std::filesystem::remove(path);
  }

  // --- 2. 板型按灯条间距判 --------------------------------------------
  {
    const cv::Mat blank(300, 400, CV_8UC3, cv::Scalar{20, 20, 20});
    const auto small = classifier.classify(blank, makeCorners(30.0F, 2.4F));
    const auto large = classifier.classify(blank, makeCorners(30.0F, 4.5F));
    expect(!small.large, "中心距 2.4 倍灯长应当判小板");
    expect(large.large, "中心距 4.5 倍灯长应当判大板");
    // 纯色图里没有数字，必须拒绝，否则每个背景框都会被当成一辆车。
    expect(
      small.verdict != L2Perception::NumberVerdict::Accepted, "纯色图不应当被采信");

    // 退化的四边形（灯条长度为 0）直接返回默认结果，不能除零。
    const std::array<cv::Point2f, 4> degenerate{
      cv::Point2f{10.0F, 10.0F}, cv::Point2f{20.0F, 10.0F}, cv::Point2f{20.0F, 10.0F},
      cv::Point2f{10.0F, 10.0F}};
    const auto broken = classifier.classify(blank, degenerate);
    expect(
      broken.verdict != L2Perception::NumberVerdict::Accepted && broken.number_image.empty(),
      "退化四边形应当直接拒绝");
  }

  // --- 3. 真实装甲板 ---------------------------------------------------
  {
    const cv::Mat image = frameOf("tests/data/sp_auto_aim/demo.avi", 520);
    expect(!image.empty(), "读不到 tests/data/sp_auto_aim/demo.avi 的第 520 帧");
    if (!image.empty()) {
      const std::array<cv::Point2f, 4> corners{
        cv::Point2f{726.06F, 499.054F}, cv::Point2f{864.309F, 516.811F},
        cv::Point2f{853.123F, 581.92F}, cv::Point2f{714.579F, 562.68F}};
      const auto result = classifier.classify(image, corners);
      expect(
        result.verdict == L2Perception::NumberVerdict::Accepted,
        "第 520 帧那块 4 号板应当被采信");
      expect(
        result.armor_class == L2Perception::ArmorClass::Infantry4,
        "第 520 帧那块板应当判成 4 号");
      expect(result.confidence > 0.7, "真板的置信度应当高于默认门限");
      expect(
        result.number_image.size() == cv::Size(20, 28) &&
          result.number_image.type() == CV_8UC1,
        "送进 MLP 的应当是 20x28 的单通道二值图");

      // 角点左右对调等于把数字镜像过去，不该还判成同一个数字。
      const std::array<cv::Point2f, 4> mirrored{
        corners[1], corners[0], corners[3], corners[2]};
      const auto flipped = classifier.classify(image, mirrored);
      expect(
        flipped.armor_class != L2Perception::ArmorClass::Infantry4 ||
          flipped.verdict != L2Perception::NumberVerdict::Accepted,
        "角点顺序反了还判成同一个数字，说明抠图没有用到角点顺序");
    }
  }

  if (failure_count != 0) {
    std::cerr << "number classifier smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "number classifier smoke test passed\n";
  return 0;
}
