#include "l2_perception/armor/light_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

// 端点连线与图像竖直方向的夹角，单位为度。分母取绝对值下限避免灯条水平时除零。
float tiltDegrees(const cv::Point2f& top, const cv::Point2f& bottom)
{
  return static_cast<float>(
    std::atan2(std::abs(top.x - bottom.x), std::max(1e-6F, std::abs(top.y - bottom.y))) *
    180.0 / CV_PI);
}

// 剖面搜索的底图取值：target 为目标颜色通道、opponent 为对方颜色通道（蓝灯
// 是 B 与 R），target < 0 时取灰度。
//   目标通道与 G 都过 saturation → 255。这是过曝白核，差分在这里是 0。G 是
//     区分白核与白色背景的那个通道：3m_high 上灯条白核 G 在 249~255，旁边白
//     海报 (255,232,235) 的 G 只有 229~240，而它的 B 与白核一样饱和。
//   其余 → gain ×（目标 − 对方）。白背景和灰背景接近 0。gain < 1 让白核比
//     光晕高出一档：贴着车身一侧的光晕 B−R 能到 180~215，与白核同档的话半高
//     区会一路铺到剖面边上。没有白核（低曝光）时整条剖面同比缩放，门限都是
//     相对峰值定的，不受影响。
struct Feature
{
  int target{-1};
  int opponent{-1};
  float saturation{245.0F};
  float gain{0.5F};

  float operator()(const cv::Vec3b& pixel) const
  {
    if (target < 0) {
      return (pixel[0] + pixel[1] + pixel[2]) / 3.0F;
    }
    if (pixel[target] >= saturation && pixel[1] >= saturation) {
      return 255.0F;
    }
    return gain * std::max(0.0F, static_cast<float>(pixel[target]) - pixel[opponent]);
  }
};

// 双线性取底图值。落在图外返回 -1，调用方据此判这一行无效。调用方保证图像
// 至少 2×2。
float sample(const cv::Mat& image, float x, float y, const Feature& feature)
{
  if (!(x >= 0.0F && y >= 0.0F && x <= image.cols - 1.0F && y <= image.rows - 1.0F)) {
    return -1.0F;
  }
  const int x0 = std::min(static_cast<int>(x), image.cols - 2);
  const int y0 = std::min(static_cast<int>(y), image.rows - 2);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const auto value = [&](int col, int row) { return feature(image.at<cv::Vec3b>(row, col)); };
  const float top = value(x0, y0) * (1.0F - fx) + value(x0 + 1, y0) * fx;
  const float bottom = value(x0, y0 + 1) * (1.0F - fx) + value(x0 + 1, y0 + 1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

// 沿灯条方向第 s 个像素处的一条垂直剖面。
struct ProfileRow
{
  float s{0.0F};
  // 脊线中心相对预测轴线的横向偏移，pixel。
  float center{0.0F};
  // 峰值 − 剖面最小值。
  float contrast{0.0F};
  // 整条剖面都在图内。
  bool inside{false};
  // 在图内，且峰上半高区没碰到剖面两端（碰到说明更亮的东西在窗外，这一行的峰
  // 不是灯条）。
  bool valid{false};
};

struct Found
{
  Light light;
  float contrast{0.0F};
};

std::optional<Found> searchOne(
  const cv::Mat& image, const LightHint& hint, const LightFinderConfig& config,
  const Feature& feature)
{
  const cv::Point2f delta = hint.bottom - hint.top;
  const float length = static_cast<float>(cv::norm(delta));
  if (!(length >= 1.0F)) {
    return std::nullopt;
  }
  const cv::Point2f along = delta / length;
  const cv::Point2f across(-along.y, along.x);
  const int half = static_cast<int>(std::ceil(
    std::max(config.profile_half_width_min_px, config.profile_half_width_ratio * length)));
  const float extend = std::max(2.0F, config.profile_extend_ratio * length);
  const int first = -static_cast<int>(std::ceil(extend));
  const int last = static_cast<int>(std::ceil(length + extend));

  std::vector<ProfileRow> rows;
  rows.reserve(static_cast<std::size_t>(last - first + 1));
  std::vector<float> profile(static_cast<std::size_t>(2 * half + 1));
  std::vector<float> smooth(profile.size());
  for (int s = first; s <= last; ++s) {
    ProfileRow row;
    row.s = static_cast<float>(s);
    bool inside = true;
    for (int k = -half; k <= half && inside; ++k) {
      const cv::Point2f point =
        hint.top + along * static_cast<float>(s) + across * static_cast<float>(k);
      const float value = sample(image, point.x, point.y, feature);
      inside = value >= 0.0F;
      profile[static_cast<std::size_t>(k + half)] = value;
    }
    row.inside = inside;
    if (inside) {
      // 先做 [1 2 1] 平滑：过曝白核在底图上会碎成几段（个别像素没过饱和门槛，
      // 掉到差分值），不平滑的话半高区在白核中间断开。
      for (std::size_t i = 0; i < profile.size(); ++i) {
        const float left = profile[i > 0 ? i - 1 : i];
        const float right = profile[i + 1 < profile.size() ? i + 1 : i];
        smooth[i] = 0.25F * left + 0.5F * profile[i] + 0.25F * right;
      }
      const float low = *std::min_element(smooth.begin(), smooth.end());
      const float level = low + 0.5F * (*std::max_element(smooth.begin(), smooth.end()) - low);
      // 半高以上可能有好几段（灯条、旁边的白海报、另一根灯条），取离预测轴线
      // 最近的那一段，而不是全局最亮的——预测已经告诉了灯条在哪。
      int left = -1;
      int right = -1;
      int best_gap = std::numeric_limits<int>::max();
      for (int i = 0; i < static_cast<int>(smooth.size());) {
        if (smooth[i] < level) {
          ++i;
          continue;
        }
        int j = i;
        while (j + 1 < static_cast<int>(smooth.size()) && smooth[j + 1] >= level) {
          ++j;
        }
        const int gap = (i <= half && half <= j) ? 0 : std::min(std::abs(i - half), std::abs(j - half));
        if (gap < best_gap) {
          best_gap = gap;
          left = i;
          right = j;
        }
        i = j + 1;
      }
      // 半高以上部分按超出量加权取质心；平顶（过曝）时就是平台中点。
      double weight = 0.0;
      double moment = 0.0;
      float peak = level;
      for (int i = left; i <= right; ++i) {
        const double w = smooth[i] - level + 1e-3;
        weight += w;
        moment += w * (i - half);
        peak = std::max(peak, smooth[i]);
      }
      row.center = static_cast<float>(moment / weight);
      row.contrast = peak - low;
      row.valid = left > 0 && right + 1 < static_cast<int>(smooth.size());
    }
    rows.push_back(row);
  }

  // 最长的一段连续亮行，允许中间断一行（灯条上偶有反光或坏点）。
  const auto lit = [&](const ProfileRow& row) {
    return row.valid && row.contrast >= config.profile_min_contrast;
  };
  int best_begin = -1;
  int best_end = -1;
  int begin = -1;
  int last_lit = -1;
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    if (!lit(rows[i])) {
      continue;
    }
    if (begin < 0 || i - last_lit > 2) {
      begin = i;
    }
    last_lit = i;
    if (best_begin < 0 || i - begin > best_end - best_begin) {
      best_begin = begin;
      best_end = i;
    }
  }
  if (best_begin < 0 || best_end - best_begin + 1 < 3) {
    return std::nullopt;
  }

  // 行峰值的半高门槛取这段亮行对比度的中位数的一半。只够 min_contrast、不够
  // 半高的多是灯条两头的光晕，脊线中心不可信，不参与拟合。
  std::vector<float> contrasts;
  for (int i = best_begin; i <= best_end; ++i) {
    if (lit(rows[i])) {
      contrasts.push_back(rows[i].contrast);
    }
  }
  std::nth_element(
    contrasts.begin(), contrasts.begin() + contrasts.size() / 2, contrasts.end());
  const float reference = contrasts[contrasts.size() / 2];
  const float threshold = 0.5F * reference;
  const auto core = [&](const ProfileRow& row) { return lit(row) && row.contrast >= threshold; };

  // 脊线拟合成直线 center = a + b·s。残差 RMS 超限时逐个剔掉残差最大的行再拟
  // 合：灯条端点外侧有时是另一块亮物（白海报）的行，对比度也过半高，会把直线
  // 拽歪。剔到不足原来的六成还压不下去，就不是一根直灯条。
  std::vector<cv::Point2d> ridge;
  for (int i = best_begin; i <= best_end; ++i) {
    if (core(rows[i])) {
      ridge.emplace_back(rows[i].s, rows[i].center);
    }
  }
  const std::size_t keep_at_least = std::max<std::size_t>(3, (ridge.size() * 6 + 9) / 10);
  double slope = 0.0;
  double offset = 0.0;
  while (true) {
    if (ridge.size() < keep_at_least) {
      return std::nullopt;
    }
    double sum_s = 0.0;
    double sum_c = 0.0;
    double sum_ss = 0.0;
    double sum_sc = 0.0;
    for (const cv::Point2d& point : ridge) {
      sum_s += point.x;
      sum_c += point.y;
      sum_ss += point.x * point.x;
      sum_sc += point.x * point.y;
    }
    const double n = static_cast<double>(ridge.size());
    const double denominator = n * sum_ss - sum_s * sum_s;
    if (!(std::abs(denominator) > 1e-9)) {
      return std::nullopt;
    }
    slope = (n * sum_sc - sum_s * sum_c) / denominator;
    offset = (sum_c - slope * sum_s) / n;
    double squared = 0.0;
    std::size_t worst = 0;
    double worst_residual = -1.0;
    for (std::size_t i = 0; i < ridge.size(); ++i) {
      const double residual = std::abs(ridge[i].y - (offset + slope * ridge[i].x));
      squared += residual * residual;
      if (residual > worst_residual) {
        worst_residual = residual;
        worst = i;
      }
    }
    if (std::sqrt(squared / n) <= config.profile_max_residual_px) {
      break;
    }
    ridge.erase(ridge.begin() + static_cast<std::ptrdiff_t>(worst));
  }
  const auto lineAt = [&](double s) { return offset + slope * s; };

  // 端点：从这段两头往外走，只要行峰值不低于端点门槛、且脊线还在直线上，就算
  // 灯条的一部分；越过门槛的那一格与前一格线性插值。以下几种说明灯条没有自然收尾，
  // 整根不要，而不是交出一截短灯条让 L3 的长度门去拦：
  //   走到搜索范围边上，或下一格已出图（被画面截断的端点是画面边缘）；
  //   下一格仍然够亮，只是脊线偏了或峰贴到剖面边上——灯条和旁边的亮物粘在
  //   一起了（白海报、另一根灯条）。
  //
  // 端点门槛单独给（profile_end_level × 中位对比度），不跟拟合用的半高走：过曝
  // 灯条的光晕沿灯条方向也铺出去，半高处的端点比灯条实际更靠外。
  const float end_threshold = config.profile_end_level * reference;
  const double tolerance = std::max(1.5, 3.0 * config.profile_max_residual_px);
  const auto onLight = [&](const ProfileRow& row) {
    return row.valid && row.contrast >= end_threshold &&
           std::abs(row.center - lineAt(row.s)) <= tolerance;
  };
  const auto crossing = [&](const ProfileRow& inside, const ProfileRow& outside) {
    const float span = inside.contrast - outside.contrast;
    const float fraction =
      span > 1e-3F ? std::clamp((inside.contrast - end_threshold) / span, 0.0F, 1.0F) : 0.0F;
    return inside.s + (outside.s - inside.s) * fraction;
  };
  // 亮行段是按 profile_min_contrast 划的，两头可能是只有光晕的行（对比度够
  // min_contrast、不够端点门槛），先往里收到第一格过门槛的行，再往外走。
  int top = best_begin;
  while (top < best_end && !onLight(rows[top])) {
    ++top;
  }
  int bottom = best_end;
  while (bottom > top && !onLight(rows[bottom])) {
    --bottom;
  }
  while (top > 0 && onLight(rows[top - 1])) {
    --top;
  }
  while (bottom + 1 < static_cast<int>(rows.size()) && onLight(rows[bottom + 1])) {
    ++bottom;
  }
  if (
    top == 0 || bottom + 1 == static_cast<int>(rows.size()) || bottom - top < 2 ||
    !rows[top - 1].inside || !rows[bottom + 1].inside ||
    rows[top - 1].contrast >= end_threshold || rows[bottom + 1].contrast >= end_threshold) {
    return std::nullopt;
  }
  const double s_top = crossing(rows[top], rows[top - 1]);
  const double s_bottom = crossing(rows[bottom], rows[bottom + 1]);
  const auto pointAt = [&](double s) {
    return hint.top + along * static_cast<float>(s) +
           across * static_cast<float>(lineAt(s));
  };

  Found found;
  found.contrast = reference;
  Light& light = found.light;
  light.top = pointAt(s_top);
  light.bottom = pointAt(s_bottom);
  if (light.top.y > light.bottom.y) {
    std::swap(light.top, light.bottom);
  }
  light.length = cv::norm(light.top - light.bottom);
  if (!(light.length >= config.min_length)) {
    return std::nullopt;
  }
  light.tilt_angle_deg = tiltDegrees(light.top, light.bottom);
  // 方向门以预测灯条为参考，不以图像竖直方向为参考：车身倾斜、斜看侧面板时
  // 灯条在图像里可以斜过 40°，预测也跟着斜。拟合直线在预测轴系里的斜率就是
  // 两者夹角的正切。
  const double off_axis_deg = std::atan(std::abs(slope)) * 180.0 / CV_PI;
  if (!(off_axis_deg < config.max_angle_deg)) {
    return std::nullopt;
  }
  light.center = (light.top + light.bottom) * 0.5F;
  light.color = lightColor(image, light.top, light.bottom, config.color_ratio_threshold);
  return found;
}

}  // namespace

std::vector<Light> searchLights(
  const cv::Mat& image, const std::vector<LightHint>& hints, const LightFinderConfig& config,
  ArmorColor color)
{
  if (image.empty() || image.type() != CV_8UC3 || image.cols < 2 || image.rows < 2) {
    return {};
  }
  Feature feature;
  feature.saturation = config.profile_saturation;
  feature.gain = config.profile_diff_gain;
  if (color == ArmorColor::Blue) {
    feature.target = 0;
    feature.opponent = 2;
  } else if (color == ArmorColor::Red) {
    feature.target = 2;
    feature.opponent = 0;
  }

  std::vector<Found> found;
  for (const LightHint& hint : hints) {
    if (auto result = searchOne(image, hint, config, feature)) {
      found.push_back(std::move(*result));
    }
  }

  // 两个 hint 收到同一根灯条：中心距小于较长那根的 0.3 倍就算同一根，留对比度
  // 高的。同一块板的两根灯条哪怕斜到 70° 也相距约 0.8 倍灯长，不会被并掉。
  std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) {
    return a.contrast > b.contrast;
  });
  std::vector<Light> lights;
  for (const Found& candidate : found) {
    const bool duplicate = std::any_of(lights.begin(), lights.end(), [&](const Light& kept) {
      return cv::norm(kept.center - candidate.light.center) <
             0.3 * std::max(kept.length, candidate.light.length);
    });
    if (!duplicate) {
      lights.push_back(candidate.light);
      lights.back().id = lights.size() - 1;
    }
  }
  return lights;
}

ArmorColor lightColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold)
{
  // 取样框：横向按灯条长度的 0.45 倍外扩，灯条宽约为长的 1/6，这样能吃到两侧光晕。
  const cv::Point2f center = (top + bottom) * 0.5F;
  const float length = static_cast<float>(cv::norm(top - bottom));
  const float half_width = std::max(3.0F, length * 0.45F);
  const float half_height = std::max(3.0F, length * 0.60F);
  cv::Rect roi(
    static_cast<int>(std::lround(center.x - half_width)),
    static_cast<int>(std::lround(center.y - half_height)),
    static_cast<int>(std::lround(half_width * 2.0F)),
    static_cast<int>(std::lround(half_height * 2.0F)));
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (roi.width < 3 || roi.height < 3) {
    return ArmorColor::Unknown;
  }

  double red = 0.0;
  double blue = 0.0;
  long long counted = 0;
  const cv::Mat patch = image(roi);
  for (int y = 0; y < patch.rows; ++y) {
    const cv::Vec3b* row = patch.ptr<cv::Vec3b>(y);
    for (int x = 0; x < patch.cols; ++x) {
      const cv::Vec3b& pixel = row[x];
      // 三通道都接近饱和的像素没有颜色信息，留着只会把比值往 1 拉。
      if (pixel[0] >= 245 && pixel[1] >= 245 && pixel[2] >= 245) {
        continue;
      }
      // 同理丢掉过暗的背景像素，它们数量远多于灯条本身。
      if (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40) {
        continue;
      }
      blue += pixel[0];
      red += pixel[2];
      ++counted;
    }
  }
  // 样本太少时不猜颜色，交给上层当未知灯条丢弃。
  if (counted < 8) {
    return ArmorColor::Unknown;
  }

  const double ratio = red / std::max(1.0, blue);
  if (ratio > ratio_threshold) {
    return ArmorColor::Red;
  }
  if (ratio < 1.0 / ratio_threshold) {
    return ArmorColor::Blue;
  }
  return ArmorColor::Unknown;
}

std::vector<Light> findLights(
  const cv::Mat& image, const cv::Rect& roi, const LightFinderConfig& config,
  ArmorColor color)
{
  const cv::Rect area = roi & cv::Rect(0, 0, image.cols, image.rows);
  if (image.empty() || image.type() != CV_8UC3 || area.area() <= 0) {
    return {};
  }

  // 底图与阈值必须配套，见 LightFinderConfig 的两个键。
  const bool color_diff = config.color_channel_diff && color != ArmorColor::Unknown;
  cv::Mat base;
  if (color_diff) {
    cv::Mat blue;
    cv::Mat red;
    cv::extractChannel(image(area), blue, 0);
    cv::extractChannel(image(area), red, 2);
    // 饱和减法：白色背景和过曝白核都被压到 0。
    if (color == ArmorColor::Blue) {
      cv::subtract(blue, red, base);
    } else {
      cv::subtract(red, blue, base);
    }
  } else {
    cv::cvtColor(image(area), base, cv::COLOR_BGR2GRAY);
  }
  cv::Mat binary;
  cv::threshold(
    base, binary, color_diff ? config.color_diff_threshold : config.binary_threshold, 255,
    cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  const cv::Point2f offset(static_cast<float>(area.x), static_cast<float>(area.y));
  std::vector<Light> lights;
  for (const std::vector<cv::Point>& contour : contours) {
    // 点数太少的轮廓拟合不出可信的方向。
    if (contour.size() < 6) {
      continue;
    }
    // 角点按 y 排序后，前两个是上边、后两个是下边，取中点作端点并补 roi 偏移。
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    std::array<cv::Point2f, 4> corners;
    rect.points(corners.data());
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
      return a.y < b.y;
    });
    const cv::Point2f top = (corners[0] + corners[1]) * 0.5F + offset;
    const cv::Point2f bottom = (corners[2] + corners[3]) * 0.5F + offset;
    const float length = static_cast<float>(cv::norm(top - bottom));
    if (!(length >= config.min_length)) {
      continue;
    }
    // 宽也按排序后的角点取，不用外接矩形的尺寸：横躺的细条按 y 排序后「宽」
    // 是它的长边，比值远大于 1 会被挡掉；用尺寸算则会把它当成竖直的短灯条。
    const float ratio = static_cast<float>(cv::norm(corners[0] - corners[1])) / length;
    if (!(config.min_ratio < ratio && ratio < config.max_ratio)) {
      continue;
    }
    const float tilt_deg = tiltDegrees(top, bottom);
    if (!(tilt_deg < config.max_angle_deg)) {
      continue;
    }

    Light light;
    light.top = top;
    light.bottom = bottom;
    light.center = (top + bottom) * 0.5F;
    light.length = length;
    light.tilt_angle_deg = tilt_deg;
    light.color = lightColor(image, top, bottom, config.color_ratio_threshold);
    light.id = lights.size();
    lights.push_back(light);
  }
  return lights;
}

}  // namespace L2Perception
