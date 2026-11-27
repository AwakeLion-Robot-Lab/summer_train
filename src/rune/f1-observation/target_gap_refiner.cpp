#include "rune/f1-observation/target_gap_refiner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace Rune {
namespace {

constexpr double kGapRadiusRatio = 0.7037;
constexpr int kOuterRays = 48;
constexpr int kAzimuthSamples = 120;

bool finite(cv::Point2f p) { return std::isfinite(p.x) && std::isfinite(p.y); }
bool finite(cv::Point2d p) { return std::isfinite(p.x) && std::isfinite(p.y); }

bool inBilinearBounds(cv::Point2f p, cv::Size size) {
  return finite(p) && p.x >= 0.0F && p.y >= 0.0F &&
         p.x < float(size.width - 1) && p.y < float(size.height - 1);
}

bool inImage(cv::Point2f p, cv::Size size) {
  return finite(p) && p.x >= 0.0F && p.y >= 0.0F &&
         p.x < float(size.width) && p.y < float(size.height);
}

double wrapAngle(double a) {
  return std::remainder(a, 2.0 * std::numbers::pi);
}

float sampleChannel(const cv::Mat& bgr, cv::Point2f p, int index) {
  const int x = int(p.x);
  const int y = int(p.y);
  const float u = p.x - float(x);
  const float v = p.y - float(y);
  const auto* top = bgr.ptr<cv::Vec3b>(y) + x;
  const auto* bottom = bgr.ptr<cv::Vec3b>(y + 1) + x;
  return (1.0F - v) * ((1.0F - u) * float(top[0][index]) + u * float(top[1][index])) +
         v * ((1.0F - u) * float(bottom[0][index]) + u * float(bottom[1][index]));
}

float sampleDiff(const cv::Mat& bgr, cv::Point2f p, int primary, int opposite) {
  const int x = int(p.x);
  const int y = int(p.y);
  const float u = p.x - float(x);
  const float v = p.y - float(y);
  const auto* top = bgr.ptr<cv::Vec3b>(y) + x;
  const auto* bottom = bgr.ptr<cv::Vec3b>(y + 1) + x;
  const auto diff = [&](const cv::Vec3b& px) {
    return float(int(px[primary]) - int(px[opposite]));
  };
  return (1.0F - v) * ((1.0F - u) * diff(top[0]) + u * diff(top[1])) +
         v * ((1.0F - u) * diff(bottom[0]) + u * diff(bottom[1]));
}

struct TargetMapper {
  bool use_homography{false};
  cv::Matx33d homography{cv::Matx33d::eye()};
  cv::Matx33d inv_homography{cv::Matx33d::eye()};
  cv::Point2d center{};
  double a{1.0}, b{1.0};
  double ca{1.0}, sa{0.0};
  cv::Point2d out_n{0.0, -1.0};
  cv::Point2d right_n{1.0, 0.0};
  double minor_px{1.0};

  static std::optional<TargetMapper> create(
      const cv::RotatedRect& ellipse, cv::Point2f outward_direction,
      const std::optional<cv::Matx33d>& H) {
    TargetMapper m;
    m.a = 0.5 * double(ellipse.size.width);
    m.b = 0.5 * double(ellipse.size.height);
    m.minor_px = std::min(m.a, m.b);
    if (H) {
      cv::Matx33d inv;
      if (cv::invert(*H, inv, cv::DECOMP_LU) != 0.0 && cv::checkRange(cv::Mat(inv, false))) {
        m.use_homography = true;
        m.homography = *H;
        m.inv_homography = inv;
        const auto c = m.toImage(0.0, 0.0);
        const auto r1 = m.toImage(1.0, 0.0);
        const auto r2 = m.toImage(0.0, 1.0);
        if (c && r1 && r2) {
          m.minor_px = std::min(cv::norm(*r1 - *c), cv::norm(*r2 - *c));
          if (std::isfinite(m.minor_px) && m.minor_px >= 6.0) return m;
        }
      }
    }
    if (!std::isfinite(m.a) || !std::isfinite(m.b) || m.minor_px < 6.0 ||
        !finite(ellipse.center) || !finite(outward_direction)) {
      return std::nullopt;
    }
    m.center = cv::Point2d(ellipse.center);
    const double rad = double(ellipse.angle) * std::numbers::pi / 180.0;
    m.ca = std::cos(rad);
    m.sa = std::sin(rad);
    const cv::Point2d raw_out(
        (double(outward_direction.x) * m.ca + double(outward_direction.y) * m.sa) / m.a,
        (-double(outward_direction.x) * m.sa + double(outward_direction.y) * m.ca) / m.b);
    const double len = cv::norm(raw_out);
    if (!std::isfinite(len) || len < 1e-6) return std::nullopt;
    m.out_n = raw_out * (1.0 / len);
    // 图像坐标下顺时针90度映射为模型 +x（右侧）。
    m.right_n = {-m.out_n.y, m.out_n.x};
    return m;
  }

  std::optional<cv::Point2f> toImage(double u, double v) const {
    if (use_homography) {
      const cv::Vec3d p = homography * cv::Vec3d(u, v, 1.0);
      if (!std::isfinite(p[2]) || std::abs(p[2]) < 1e-6) return std::nullopt;
      const cv::Point2f px(float(p[0] / p[2]), float(p[1] / p[2]));
      return finite(px) ? std::optional{px} : std::nullopt;
    }
    const cv::Point2d n = u * right_n + v * out_n;
    const double lx = n.x * a;
    const double ly = n.y * b;
    const cv::Point2f px(float(center.x + ca * lx - sa * ly),
                         float(center.y + sa * lx + ca * ly));
    return finite(px) ? std::optional{px} : std::nullopt;
  }

  std::optional<cv::Point2d> toCanonical(cv::Point2f p) const {
    if (!finite(p)) return std::nullopt;
    if (use_homography) {
      const cv::Vec3d q = inv_homography * cv::Vec3d(p.x, p.y, 1.0);
      if (!std::isfinite(q[2]) || std::abs(q[2]) < 1e-6) return std::nullopt;
      const cv::Point2d uv(q[0] / q[2], q[1] / q[2]);
      return finite(uv) ? std::optional{uv} : std::nullopt;
    }
    const cv::Point2d d = cv::Point2d(p) - center;
    const cv::Point2d n((d.x * ca + d.y * sa) / a, (-d.x * sa + d.y * ca) / b);
    return cv::Point2d(n.dot(right_n), n.dot(out_n));
  }

  std::optional<cv::Point2f> polarToImage(double r, double phi) const {
    return toImage(r * std::sin(phi), r * std::cos(phi));
  }
};

struct CircleFit {
  cv::Point2d center;
  double radius;
};

std::optional<CircleFit> fitWeightedCircle(const std::vector<cv::Point2d>& pts,
                                           const std::vector<double>& weights) {
  cv::Matx33d lhs = cv::Matx33d::zeros();
  cv::Vec3d rhs{};
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const double w = weights[i];
    const auto& p = pts[i];
    const cv::Vec3d row(2.0 * p.x, 2.0 * p.y, 1.0);
    lhs += (row * row.t()) * w;
    rhs += row * (p.dot(p) * w);
  }
  cv::Vec3d sol{};
  if (!cv::solve(lhs, rhs, sol, cv::DECOMP_CHOLESKY)) return std::nullopt;
  const cv::Point2d c(sol[0], sol[1]);
  const double r2 = c.dot(c) + sol[2];
  if (!std::isfinite(r2) || r2 <= 0.0) return std::nullopt;
  return CircleFit{c, std::sqrt(r2)};
}

std::optional<cv::RotatedRect> refineOuterEllipse(
    const cv::Mat& bgr, int primary, int opposite, const cv::RotatedRect& initial) {
  const float a = initial.size.width * 0.5F;
  const float b = initial.size.height * 0.5F;
  const float minor = std::min(a, b);
  if (minor < 8.0F || !std::isfinite(a) || !std::isfinite(b)) return std::nullopt;
  const float ang = initial.angle * float(std::numbers::pi / 180.0);
  const float c = std::cos(ang), s = std::sin(ang);
  const auto toPixel = [&](cv::Point2f p) {
    return initial.center + cv::Point2f(c * a * p.x - s * b * p.y, s * a * p.x + c * b * p.y);
  };

  std::vector<cv::Point2d> samples;
  std::vector<int> rays;
  samples.reserve(kOuterRays);
  rays.reserve(kOuterRays);
  for (int i = 0; i < kOuterRays; ++i) {
    const double phi = 2.0 * std::numbers::pi * i / kOuterRays;
    const cv::Point2f unit(float(std::cos(phi)), float(std::sin(phi)));
    const float pixel_r = float(cv::norm(toPixel(unit) - initial.center));
    if (pixel_r < 6.0F) continue;
    const float step = std::clamp(1.0F / pixel_r, 0.008F, 0.05F);
    const int n_steps = int(std::floor((1.16F - 0.84F) / step)) + 1;
    if (n_steps < 5 || n_steps > 96) continue;
    std::array<float, 98> vals{};
    bool ok = true;
    for (int k = 0; k < n_steps; ++k) {
      const float r = 0.84F + float(k) * step;
      const auto p = toPixel(r * unit);
      if (!inBilinearBounds(p, bgr.size())) {
        ok = false;
        break;
      }
      vals[k] = sampleDiff(bgr, p, primary, opposite);
    }
    if (!ok) continue;
    float best_score = 0.0F;
    float best_r = 0.0F;
    for (int k = 1; k + 1 < n_steps; ++k) {
      const float g = vals[k - 1] - vals[k + 1];
      if (g < 14.0F || vals[k - 1] < 14.0F) continue;
      const float g_prev = k >= 2 ? (vals[k - 2] - vals[k]) : g;
      const float g_next = k + 2 < n_steps ? (vals[k] - vals[k + 2]) : g;
      if (g < g_prev || g < g_next) continue;
      const float r_k = 0.84F + float(k) * step;
      const float score = g / (1.0F + 2.5F * std::abs(r_k - 1.0F));
      if (score <= best_score) continue;
      const float denom = g_prev - 2.0F * g + g_next;
      const float sub = denom < -1e-3F ? std::clamp(0.5F * (g_prev - g_next) / denom, -0.5F, 0.5F) : 0.0F;
      best_score = score;
      best_r = r_k + sub * step;
    }
    if (best_score > 0.0F) {
      samples.emplace_back(double(unit.x) * best_r, double(unit.y) * best_r);
      rays.push_back(i);
    }
  }
  if (samples.size() < 20) return std::nullopt;

  std::vector<double> weights(samples.size(), 1.0);
  auto fit = fitWeightedCircle(samples, weights);
  if (!fit) return std::nullopt;
  const double huber = std::max(1.25 / double(minor), 0.025);
  for (int iter = 0; iter < 3; ++iter) {
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const double err = std::abs(cv::norm(samples[i] - fit->center) - fit->radius);
      weights[i] = err <= huber ? 1.0 : huber / err;
    }
    auto next = fitWeightedCircle(samples, weights);
    if (!next) break;
    fit = next;
  }
  if (cv::norm(fit->center) > 0.14 || fit->radius < 0.88 || fit->radius > 1.12)
    return std::nullopt;
  std::array<int, 4> quadrants{};
  int inliers = 0;
  double err_sq = 0.0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double err = std::abs(cv::norm(samples[i] - fit->center) - fit->radius);
    if (err > 2.0 * huber) continue;
    ++inliers;
    err_sq += err * err;
    ++quadrants[rays[i] / (kOuterRays / 4)];
  }
  if (inliers < 18 ||
      std::count_if(quadrants.begin(), quadrants.end(), [](int n) { return n >= 3; }) < 3 ||
      std::sqrt(err_sq / double(inliers)) > huber * 1.35) {
    return std::nullopt;
  }
  const auto new_center = toPixel(cv::Point2f(float(fit->center.x), float(fit->center.y)));
  if (!inImage(new_center, bgr.size())) return std::nullopt;
  return cv::RotatedRect(new_center, initial.size * float(fit->radius), initial.angle);
}

double percentile(std::vector<float> values, double p) {
  if (values.empty()) return 0.0;
  const auto idx = std::clamp(std::size_t(std::llround(p * double(values.size() - 1))),
                              std::size_t(0), values.size() - 1);
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return double(values[idx]);
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double m = values[mid];
  if (values.size() % 2 == 0) {
    const double lo = *std::max_element(values.begin(), values.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

}  // namespace

TargetGapRefinementResult refineTargetGapsFromImage(
    const cv::Mat& bgr, Color color, const cv::RotatedRect& initial_ellipse,
    cv::Point2f outward_direction,
    const std::optional<cv::Matx33d>& homography_target_to_image,
    const std::array<double, 8>* prior_sigma_px,
    bool refine_outer_ellipse) {
  TargetGapRefinementResult result;
  result.refined_ellipse = initial_ellipse;
  if (bgr.empty() || bgr.type() != CV_8UC3) return result;
  const int primary = color == Color::Blue ? 0 : 2;
  const int opposite = 2 - primary;

  if (refine_outer_ellipse && !homography_target_to_image) {
    if (auto refined = refineOuterEllipse(bgr, primary, opposite, initial_ellipse)) {
      result.refined_ellipse = *refined;
      result.ellipse_refined = true;
    }
  }

  const auto mapper = TargetMapper::create(
      result.refined_ellipse, outward_direction, homography_target_to_image);
  if (!mapper) return result;

  // 1. 沿环带 r ∈ [0.58, 0.81] 做极坐标采样，用径向中值抑制单点噪点。
  constexpr std::array<double, 5> kProfileRadii{0.58, 0.64, kGapRadiusRatio, 0.76, 0.81};
  std::array<float, kAzimuthSamples> profile{};
  std::vector<float> valid_samples;
  valid_samples.reserve(kAzimuthSamples);
  int valid_azimuths = 0;
  for (int k = 0; k < kAzimuthSamples; ++k) {
    const double phi = 2.0 * std::numbers::pi * double(k) / double(kAzimuthSamples);
    std::array<float, 5> shell{};
    int count = 0;
    for (double r : kProfileRadii) {
      const auto px = mapper->polarToImage(r, phi);
      if (px && inBilinearBounds(*px, bgr.size()))
        shell[count++] = sampleDiff(bgr, *px, primary, opposite);
    }
    if (count < 3) {
      profile[k] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    std::sort(shell.begin(), shell.begin() + count);
    profile[k] = shell[count / 2];
    valid_samples.push_back(profile[k]);
    ++valid_azimuths;
  }
  if (valid_azimuths < kAzimuthSamples / 2) return result;

  const double bg = percentile(valid_samples, 0.25);
  const double pk = percentile(valid_samples, 0.96);
  const double contrast = pk - bg;
  if (!std::isfinite(contrast) || contrast < 16.0) return result;

  // 2. 四重周期匹配滤波估计公共相位（4条亮辐条对应 0, 90, 180, 270 度）。
  double c4 = 0.0, s4 = 0.0, total_pos = 0.0;
  for (int k = 0; k < kAzimuthSamples; ++k) {
    if (!std::isfinite(profile[k])) continue;
    const double val = std::max(0.0, double(profile[k]) - bg);
    const double phi = 2.0 * std::numbers::pi * double(k) / double(kAzimuthSamples);
    c4 += val * std::cos(4.0 * phi);
    s4 += val * std::sin(4.0 * phi);
    total_pos += val;
  }
  if (total_pos <= 1e-6) return result;
  const double modulation4 = std::hypot(c4, s4) / total_pos;
  // 排除均匀亮环（如已激活靶圈）或无四重对称结构的区域。
  if (modulation4 < 0.18) return result;
  const double global_phase = 0.25 * std::atan2(s4, c4);
  if (std::abs(global_phase) > 22.5 * std::numbers::pi / 180.0) return result;
  result.phase_offset_rad = global_phase;

  // 3. 对 4 条亮辐条做多径向层切向亚像素左右边界精修。
  struct SpokeMeasurement {
    double center_phi{0.0};
    double left_phi{0.0};
    double right_phi{0.0};
    double sigma_phi{0.0};
    double contrast{0.0};
  };
  std::array<std::optional<SpokeMeasurement>, 4> spokes;
  const double arc_radius_px = kGapRadiusRatio * mapper->minor_px;
  const double dphi = std::clamp(0.75 / std::max(8.0, arc_radius_px),
                                 0.5 * std::numbers::pi / 180.0,
                                 2.0 * std::numbers::pi / 180.0);
  const double half_window = 22.0 * std::numbers::pi / 180.0;
  const int half_steps = std::clamp(int(std::ceil(half_window / dphi)), 10, 48);
  const int n_scan = 2 * half_steps + 1;
  constexpr std::array<double, 5> kSpokeRadii{0.60, 0.65, kGapRadiusRatio, 0.75, 0.80};

  for (int s = 0; s < 4; ++s) {
    const double nominal_phi = s * (0.5 * std::numbers::pi) + global_phase;
    std::vector<double> shell_mid, shell_left, shell_right, shell_contrast;
    shell_mid.reserve(kSpokeRadii.size());
    shell_left.reserve(kSpokeRadii.size());
    shell_right.reserve(kSpokeRadii.size());
    shell_contrast.reserve(kSpokeRadii.size());

    for (double r : kSpokeRadii) {
      std::array<float, 98> vals{};
      bool valid = true;
      for (int i = 0; i < n_scan; ++i) {
        const double phi = nominal_phi + double(i - half_steps) * dphi;
        const auto px = mapper->polarToImage(r, phi);
        if (!px || !inBilinearBounds(*px, bgr.size())) {
          valid = false;
          break;
        }
        vals[i] = sampleDiff(bgr, *px, primary, opposite);
      }
      if (!valid) continue;

      // 在中心 ±12° 内寻找亮辐条峰值。
      const int peak_margin = std::min(half_steps - 2,
                                       std::max(3, int(std::ceil((12.0 * std::numbers::pi / 180.0) / dphi))));
      int peak_idx = half_steps;
      float peak_val = -1e9F;
      for (int i = half_steps - peak_margin; i <= half_steps + peak_margin; ++i) {
        if (vals[i] > peak_val) {
          peak_val = vals[i];
          peak_idx = i;
        }
      }
      const double local_contrast = double(peak_val) - bg;
      if (local_contrast < std::max(14.0, 0.25 * contrast)) continue;

      // 左侧上升沿（暗到亮）亚像素梯度峰值。
      int left_idx = -1;
      float left_grad = 0.0F;
      for (int i = std::max(1, peak_idx - half_steps + 1); i < peak_idx; ++i) {
        const float g = vals[i + 1] - vals[i - 1];
        if (g > left_grad) {
          left_grad = g;
          left_idx = i;
        }
      }
      // 右侧下降沿（亮到暗）亚像素梯度峰值。
      int right_idx = -1;
      float right_grad = 0.0F;
      for (int i = peak_idx + 1; i <= std::min(n_scan - 2, peak_idx + half_steps - 1); ++i) {
        const float g = vals[i - 1] - vals[i + 1];
        if (g > right_grad) {
          right_grad = g;
          right_idx = i;
        }
      }
      const float min_edge_grad = float(std::max(10.0, 0.20 * local_contrast));
      if (left_idx < 1 || right_idx < 1 || left_grad < min_edge_grad || right_grad < min_edge_grad)
        continue;

      const auto subpixelOffset = [&](int idx, bool rising) {
        const auto gradAt = [&](int k) {
          return rising ? (vals[k + 1] - vals[k - 1]) : (vals[k - 1] - vals[k + 1]);
        };
        if (idx <= 1 || idx >= n_scan - 2) return 0.0;
        const float g0 = gradAt(idx - 1), g1 = gradAt(idx), g2 = gradAt(idx + 1);
        const float denom = g0 - 2.0F * g1 + g2;
        if (denom >= -1e-3F) return 0.0;
        return double(std::clamp(0.5F * (g0 - g2) / denom, -0.5F, 0.5F));
      };
      const double left_phi = nominal_phi + (double(left_idx - half_steps) + subpixelOffset(left_idx, true)) * dphi;
      const double right_phi = nominal_phi + (double(right_idx - half_steps) + subpixelOffset(right_idx, false)) * dphi;
      const double width_deg = (right_phi - left_phi) * 180.0 / std::numbers::pi;
      if (width_deg < 2.5 || width_deg > 34.0) continue;

      // 校验辐条外侧确为暗缺口，排除大面积反光或亮块干扰。
      const int flank_shift = std::max(2, int(std::round((6.0 * std::numbers::pi / 180.0) / dphi)));
      const int l_flank = std::max(0, left_idx - flank_shift);
      const int r_flank = std::min(n_scan - 1, right_idx + flank_shift);
      const double dark_limit = bg + 0.45 * local_contrast;
      if (double(vals[l_flank]) > dark_limit || double(vals[r_flank]) > dark_limit) continue;

      shell_left.push_back(left_phi);
      shell_right.push_back(right_phi);
      shell_mid.push_back(0.5 * (left_phi + right_phi));
      shell_contrast.push_back(local_contrast);
    }

    if (shell_mid.size() < 2) continue;
    const double mid_med = median(shell_mid);
    const double left_med = median(shell_left);
    const double right_med = median(shell_right);
    double disp_sq = 0.0;
    for (double m : shell_mid) disp_sq += (m - mid_med) * (m - mid_med);
    const double shell_sigma = std::sqrt(disp_sq / double(shell_mid.size()));
    if (shell_sigma > 6.0 * std::numbers::pi / 180.0) continue;
    const double c_med = median(shell_contrast);
    const double sigma_px = std::clamp(
        std::hypot(arc_radius_px * shell_sigma, 16.0 / std::max(12.0, c_med) + 0.45),
        0.6, 5.0);
    spokes[s] = SpokeMeasurement{mid_med, left_med, right_med,
                                 sigma_px / std::max(8.0, arc_radius_px), c_med};
  }

  // 4. 校验并精修 4 个暗缺口（编号 0, 2, 4, 6）及 4 个亮辐条（编号 1, 3, 5, 7）。
  std::array<std::optional<double>, 8> raw_angles{};
  std::array<double, 8> raw_sigma_phi{};
  for (int s = 0; s < 4; ++s) {
    if (!spokes[s]) continue;
    const int slot = 2 * s + 1;
    raw_angles[slot] = spokes[s]->center_phi;
    raw_sigma_phi[slot] = spokes[s]->sigma_phi;
  }

  for (int g = 0; g < 4; ++g) {
    const int s_prev = (g + 3) % 4;
    const int s_next = g;
    const double nominal_gap = (2 * g - 1) * (0.25 * std::numbers::pi) + global_phase;
    // 检验暗缺口内部确为暗区，且外环处存在发光边缘（防止在被遮挡区域凭空产出点）。
    std::vector<float> interior;
    interior.reserve(15);
    for (int ia = -2; ia <= 2; ++ia) {
      const double phi = nominal_gap + double(ia) * (8.0 * std::numbers::pi / 180.0);
      for (double r : {0.60, kGapRadiusRatio, 0.78}) {
        const auto px = mapper->polarToImage(r, phi);
        if (px && inBilinearBounds(*px, bgr.size()))
          interior.push_back(sampleDiff(bgr, *px, primary, opposite));
      }
    }
    if (interior.size() < 9) continue;
    const double gap_med = percentile(interior, 0.5);
    if (gap_med - bg > std::max(14.0, 0.28 * contrast)) continue;

    int bright_outer = 0;
    for (int ia = -2; ia <= 2; ++ia) {
      const double phi = nominal_gap + double(ia) * (10.0 * std::numbers::pi / 180.0);
      float best_rim = -1e9F;
      for (double r : {0.86, 0.91, 0.96, 1.01}) {
        const auto px = mapper->polarToImage(r, phi);
        if (px && inBilinearBounds(*px, bgr.size()))
          best_rim = std::max(best_rim, sampleDiff(bgr, *px, primary, opposite));
      }
      if (double(best_rim) - bg >= std::max(12.0, 0.16 * contrast)) ++bright_outer;
    }
    if (bright_outer < 2) continue;

    const int slot = 2 * g;
    if (spokes[s_prev] && spokes[s_next]) {
      const double a0 = spokes[s_prev]->right_phi;
      const double span = std::remainder(spokes[s_next]->left_phi - a0, 2.0 * std::numbers::pi);
      const double positive_span = span < 0.0 ? span + 2.0 * std::numbers::pi : span;
      if (positive_span >= 45.0 * std::numbers::pi / 180.0 &&
          positive_span <= 110.0 * std::numbers::pi / 180.0) {
        raw_angles[slot] = a0 + 0.5 * positive_span;
        raw_sigma_phi[slot] = 0.75 * std::hypot(spokes[s_prev]->sigma_phi, spokes[s_next]->sigma_phi);
      }
    } else if (spokes[s_prev] || spokes[s_next]) {
      const auto& known = spokes[s_prev] ? *spokes[s_prev] : *spokes[s_next];
      const double offset = spokes[s_prev] ? (0.25 * std::numbers::pi) : (-0.25 * std::numbers::pi);
      raw_angles[slot] = known.center_phi + offset;
      raw_sigma_phi[slot] = 1.5 * known.sigma_phi;
    }
  }

  // 5. 靶面刚体八点相位 Huber 联合约束：剔除单点离群角并做方差加权融合。
  int observed = 0;
  for (const auto& a : raw_angles) observed += a.has_value();
  if (observed < 2) return result;

  double rigid_phase = global_phase;
  const double huber_phi = 3.0 * std::numbers::pi / 180.0;
  for (int iter = 0; iter < 4; ++iter) {
    double num = 0.0, den = 0.0;
    for (int k = 0; k < 8; ++k) {
      if (!raw_angles[k]) continue;
      const double base = double(k - 1) * (0.25 * std::numbers::pi);
      const double delta = wrapAngle(*raw_angles[k] - base - rigid_phase);
      const double inv_var = 1.0 / std::max(1e-6, raw_sigma_phi[k] * raw_sigma_phi[k]);
      const double w = inv_var * (std::abs(delta) <= huber_phi ? 1.0 : huber_phi / std::abs(delta));
      num += w * delta;
      den += w;
    }
    if (den <= 1e-12) break;
    rigid_phase = wrapAngle(rigid_phase + num / den);
  }
  result.phase_offset_rad = rigid_phase;

  // 计算刚体公共相位的后验方差，用于与单点观测融合。
  double info_sum = 0.0;
  for (int k = 0; k < 8; ++k) {
    if (!raw_angles[k]) continue;
    const double base = double(k - 1) * (0.25 * std::numbers::pi);
    const double delta = std::abs(wrapAngle(*raw_angles[k] - base - rigid_phase));
    const double gate_phi = std::max(8.0 * std::numbers::pi / 180.0, 3.0 * raw_sigma_phi[k]);
    if (delta > gate_phi) continue;
    info_sum += 1.0 / std::max(1e-6, raw_sigma_phi[k] * raw_sigma_phi[k]);
  }
  const double rigid_var_phi = info_sum > 1e-9 ? (1.0 / info_sum) : 1e-2;

  for (int k = 0; k < 8; ++k) {
    if (!raw_angles[k]) continue;
    const double base = double(k - 1) * (0.25 * std::numbers::pi);
    const double delta = wrapAngle(*raw_angles[k] - base - rigid_phase);
    const double gate_phi = std::max(8.0 * std::numbers::pi / 180.0, 3.0 * raw_sigma_phi[k]);
    if (std::abs(delta) > gate_phi) continue;
    // 在单点实测与刚体结构约束之间按精度加权融合（保留局部自适应能力同时抑制抖动）。
    const double obs_var = raw_sigma_phi[k] * raw_sigma_phi[k];
    const double prior_var = std::max(rigid_var_phi * 2.5, (1.2 * std::numbers::pi / 180.0) * (1.2 * std::numbers::pi / 180.0));
    const double gain = prior_var / (prior_var + obs_var);
    const double fused_phi = base + rigid_phase + gain * delta;
    const auto px = mapper->polarToImage(kGapRadiusRatio, fused_phi);
    if (!px || !inImage(*px, bgr.size())) continue;
    double sigma_px = std::clamp(std::sqrt(obs_var) * arc_radius_px, 0.7, 5.5);
    if (prior_sigma_px && std::isfinite((*prior_sigma_px)[k]) && (*prior_sigma_px)[k] > 0.0) {
      // 当存在三维先验搜索椭圆时，结合先验偏差对边缘点方差做合理下界约束。
      sigma_px = std::max(sigma_px, 0.5);
    }
    result.gap_points[k] = *px;
    result.gap_point_variances[k] = float(sigma_px * sigma_px);
    if (k % 2 == 1) ++result.measured_spokes;
    else ++result.measured_gaps;
  }
  return result;
}

void refineGapPointsRigidHuber(
    std::array<std::optional<cv::Point2f>, 8>& current_points,
    std::array<float, 8>& current_variances,
    const std::array<std::optional<cv::Point2f>, 8>& predicted_points,
    const std::array<float, 8>& predicted_variances,
    const cv::RotatedRect& target_ellipse, cv::Point2f outward_direction,
    cv::Size image_size, float alpha, std::size_t& rejected_count) {
  const auto mapper = TargetMapper::create(target_ellipse, outward_direction, std::nullopt);
  if (!mapper) return;
  const double radius_px = std::max(6.0, mapper->minor_px);
  const double default_var = std::pow(std::max(1.2, 0.025 * 2.0 * radius_px), 2);
  const double gate_px = std::max(3.0, 0.12 * 2.0 * radius_px);

  struct Pair {
    int id;
    cv::Point2d pred_uv;
    cv::Point2d obs_uv;
    double var_uv;
  };
  std::vector<Pair> pairs;
  pairs.reserve(8);
  for (int i = 0; i < 8; ++i) {
    if (!current_points[i] || !predicted_points[i]) continue;
    const auto p_uv = mapper->toCanonical(*predicted_points[i]);
    const auto o_uv = mapper->toCanonical(*current_points[i]);
    if (!p_uv || !o_uv) continue;
    const double obs_var = current_variances[i] > 0.0F ? double(current_variances[i]) : default_var;
    const double pred_var = predicted_variances[i] > 0.0F ? double(predicted_variances[i]) : default_var;
    const double var_uv = (obs_var + pred_var) / (radius_px * radius_px);
    pairs.push_back({i, *p_uv, *o_uv, std::max(1e-6, var_uv)});
  }
  if (pairs.empty()) return;

  // 在规范化靶面坐标系中，对 (dx, dy, dtheta, ds) 做 Huber-IRLS 平面相似刚体拟合。
  cv::Vec4d params(0.0, 0.0, 0.0, 0.0);  // tx, ty, dtheta, ds
  if (pairs.size() >= 2) {
    const double huber_uv = std::max(2.0 / radius_px, 0.035);
    const cv::Matx44d prior = cv::Matx44d::diag({1.0 / (0.12 * 0.12), 1.0 / (0.12 * 0.12),
                                                 1.0 / (0.12 * 0.12), 1.0 / (0.08 * 0.08)});
    for (int iter = 0; iter < 4; ++iter) {
      cv::Matx44d H = prior;
      cv::Vec4d g = -(prior * params);
      for (const auto& pair : pairs) {
        const double x = pair.pred_uv.x, y = pair.pred_uv.y;
        const double pred_x = (1.0 + params[3]) * x - params[2] * y + params[0];
        const double pred_y = params[2] * x + (1.0 + params[3]) * y + params[1];
        const double rx = pair.obs_uv.x - pred_x;
        const double ry = pair.obs_uv.y - pred_y;
        const double err = std::hypot(rx, ry);
        const double robust = err <= huber_uv ? 1.0 : huber_uv / err;
        const double w = robust / pair.var_uv;
        const cv::Matx<double, 2, 4> J(1.0, 0.0, -y, x,
                                       0.0, 1.0,  x, y);
        H += (J.t() * J) * w;
        g += J.t() * cv::Vec2d(rx, ry) * w;
      }
      cv::Vec4d step{};
      if (!cv::solve(H, g, step, cv::DECOMP_CHOLESKY)) break;
      params += step;
      params[0] = std::clamp(params[0], -0.15, 0.15);
      params[1] = std::clamp(params[1], -0.15, 0.15);
      params[2] = std::clamp(params[2], -0.15, 0.15);
      params[3] = std::clamp(params[3], -0.10, 0.10);
      if (cv::norm(step) < 1e-6) break;
    }
  }

  for (const auto& pair : pairs) {
    const int i = pair.id;
    const double x = pair.pred_uv.x, y = pair.pred_uv.y;
    const cv::Point2d aligned_uv(
        (1.0 + params[3]) * x - params[2] * y + params[0],
        params[2] * x + (1.0 + params[3]) * y + params[1]);
    const auto aligned_px = mapper->toImage(aligned_uv.x, aligned_uv.y);
    const auto pred_px = *predicted_points[i];
    if (!aligned_px || !inImage(*aligned_px, image_size)) {
      current_points[i].reset();
      current_variances[i] = 0.0F;
      ++rejected_count;
      continue;
    }
    const double obs_var = current_variances[i] > 0.0F ? double(current_variances[i]) : default_var;
    const double pred_var = predicted_variances[i] > 0.0F ? double(predicted_variances[i]) : default_var;
    const double raw_dist = cv::norm(*current_points[i] - pred_px);
    const double rigid_dist = cv::norm(*current_points[i] - *aligned_px);
    const double chi2_gate = std::sqrt(9.21 * (obs_var + pred_var));
    if (raw_dist > gate_px || rigid_dist > std::max(3.0, chi2_gate)) {
      current_points[i].reset();
      current_variances[i] = 0.0F;
      ++rejected_count;
      continue;
    }
    // 使用方差自适应卡尔曼增益融合预测与本帧亚像素观测，抑制高方差噪点。
    const double effective_pred_var = pred_var / std::clamp(double(alpha), 0.08, 0.95);
    const double gain = std::clamp(effective_pred_var / (effective_pred_var + obs_var), 0.15, 0.85);
    const cv::Point2f fused = float(gain) * (*current_points[i]) + float(1.0 - gain) * pred_px;
    if (!inImage(fused, image_size)) {
      current_points[i].reset();
      current_variances[i] = 0.0F;
      ++rejected_count;
      continue;
    }
    current_points[i] = fused;
    current_variances[i] = float(std::max(0.36, (1.0 - gain) * effective_pred_var));
  }
}

}  // namespace Rune
