// ArmorRefiner 的四种判定分支各自验证一次。用合成图而不是真实录像，
// 是为了让"两根灯条 / 单灯条 / 粘连 / 太暗"这四种工况互相独立、可精确构造。
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

// 画一根竖直灯条。颜色按 BGR 给出，蓝板灯条只点亮 B 通道。
void drawLightbar(cv::Mat& image, float center_x, float center_y, float length, float width,
                  const cv::Scalar& color)
{
  const cv::Rect bar(static_cast<int>(center_x - width * 0.5F),
                     static_cast<int>(center_y - length * 0.5F), static_cast<int>(width),
                     static_cast<int>(length));
  cv::rectangle(image, bar & cv::Rect(0, 0, image.cols, image.rows), color, cv::FILLED);
}

// 按左右灯条的几何位置生成对应的网络角点，顺序为左上、右上、右下、左下。
// 故意加入偏移，模拟网络角点内缩到灯条内侧的真实行为。
L2Perception::Armor makeArmor(float left_x, float right_x, float center_y, float length,
                              float corner_bias)
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
  armor.center = armor.center * 0.25F;
  return armor;
}

// 真实灯条在相机里是过曝的，核心接近白色，灰度远高于 SP-Vision 的 150 阈值。
// 这里刻意用高亮度而不是纯饱和蓝，否则灰度化后过不了阈值，测不到真实通路。
const cv::Scalar kBlueBar{255, 230, 230};

// 两根灯条都在：角点应被替换，检出保留。
void testRefined(const L2Perception::ArmorRefiner& refiner)
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F, kBlueBar);
  drawLightbar(image, 380.0F, 240.0F, 60.0F, 6.0F, kBlueBar);

  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 380.0F, 240.0F, 60.0F, 4.0F)};
  const auto before = armors.front().corners;
  const auto stats = refiner.refine(image, armors);

  require(stats.refined == 1 && stats.rejected == 0, "two lightbars must produce a refined armor");
  require(armors.size() == 1, "a refined armor must be kept");

  const auto& armor = armors.front();
  require(armor.corner_source == L2Perception::CornerSource::Refined, "corner_source must be Refined");
  require(armor.corner_shift > 0.0F, "refined corners must differ from the network corners");
  // 原始角点必须留档，否则无法离线验证精修是否真的有收益。
  require(armor.network_corners == before, "network_corners must preserve the pre-refine corners");

  // 精修后的角点应当贴到灯条实际端点（±half=210/270），比内缩的网络角点更外。
  const float top_error = std::abs(armor.corners[0].y - 210.0F);
  const float bottom_error = std::abs(armor.corners[3].y - 270.0F);
  require(top_error < 3.0F && bottom_error < 3.0F,
          "refined corners must land on the real lightbar endpoints");
}

// 只有一根灯条，且明显侧对：必须拒绝整块检出。
void testRejectedSingleLightbar(const L2Perception::ArmorRefiner& refiner)
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F, kBlueBar);

  // 宽高比 40/60≈0.67，低于 edge_on_aspect_ratio=1.2，属于侧对工况。
  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 340.0F, 240.0F, 60.0F, 2.0F)};
  const auto stats = refiner.refine(image, armors);

  require(stats.rejected == 1, "a single lightbar on an edge-on armor must be rejected");
  require(armors.empty(), "a rejected armor must be erased from the results");
}

// 只有一根灯条但装甲板正对：更可能是曝光问题而不是遮挡，必须保留。
void testFrontFacingSingleLightbarKept(const L2Perception::ArmorRefiner& refiner)
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F, kBlueBar);

  // 宽高比 150/60=2.5，远高于门限，不允许据此判定遮挡。
  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 450.0F, 240.0F, 60.0F, 2.0F)};
  const auto stats = refiner.refine(image, armors);

  require(stats.rejected == 0, "a front-facing armor must not be rejected for one lightbar");
  require(armors.size() == 1 && stats.network_kept == 1, "the armor must survive with network corners");
  require(armors.front().corner_source == L2Perception::CornerSource::Network,
          "an unrefined armor must keep CornerSource::Network");
}

// 过曝把两根灯条粘成一块：属于检查失效，不能当成单灯条证据。
void testMergedBlobKept(const L2Perception::ArmorRefiner& refiner)
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  // 一整块横跨左右中轴的亮区，模拟近距离过曝。
  drawLightbar(image, 340.0F, 240.0F, 60.0F, 84.0F, kBlueBar);

  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 380.0F, 240.0F, 60.0F, 2.0F)};
  const auto stats = refiner.refine(image, armors);

  require(stats.rejected == 0, "a merged blob must not be treated as a single lightbar");
  require(armors.size() == 1, "a merged blob must keep the detection");
}

// 灯条太短：传统检测的结论是噪声，必须跳过检查而不是拒绝。
void testTooSmallSkipped(const L2Perception::ArmorRefiner& refiner)
{
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 5.0F, 2.0F, kBlueBar);

  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 304.0F, 240.0F, 5.0F, 0.0F)};
  const auto stats = refiner.refine(image, armors);

  require(stats.rejected == 0, "a far-range armor must never be rejected by the lightbar check");
  require(armors.size() == 1 && stats.network_kept == 1, "a far-range armor must be kept as-is");
}

// 关闭开关后必须完全不改动输入，保证可以随时回退对照。
void testDisabled()
{
  L2Perception::ArmorRefinerConfig config;
  config.enable = false;
  const L2Perception::ArmorRefiner refiner(config);

  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar::all(0));
  drawLightbar(image, 300.0F, 240.0F, 60.0F, 6.0F, kBlueBar);

  std::vector<L2Perception::Armor> armors{makeArmor(300.0F, 340.0F, 240.0F, 60.0F, 2.0F)};
  const auto before = armors.front().corners;
  const auto stats = refiner.refine(image, armors);

  require(stats.refined == 0 && stats.rejected == 0 && stats.network_kept == 0,
          "a disabled refiner must not report any verdict");
  require(armors.size() == 1 && armors.front().corners == before,
          "a disabled refiner must leave the detections untouched");
}

}  // namespace

int main()
{
  try {
    const L2Perception::ArmorRefiner refiner{};
    testRefined(refiner);
    testRejectedSingleLightbar(refiner);
    testFrontFacingSingleLightbarKept(refiner);
    testMergedBlobKept(refiner);
    testTooSmallSkipped(refiner);
    testDisabled();
  } catch (const std::exception& error) {
    std::printf("armor refiner smoke failed: %s\n", error.what());
    return 1;
  }

  std::printf("armor refiner smoke passed\n");
  return 0;
}
