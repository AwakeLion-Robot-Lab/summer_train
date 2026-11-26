#include "ring_reobservation.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace Rune::detail {
namespace {
constexpr int ray_count = 48;

struct RimSample {
  cv::Point2d normalized;
  int ray;
};

float channel(const cv::Mat& bgr, cv::Point2f p, int index) {
  const int x = int(p.x), y = int(p.y);
  const float u = p.x - x, v = p.y - y;
  const auto* top = bgr.ptr<cv::Vec3b>(y) + x;
  const auto* bottom = bgr.ptr<cv::Vec3b>(y + 1) + x;
  return (1-v)*((1-u)*top[0][index]+u*top[1][index]) +
         v*((1-u)*bottom[0][index]+u*bottom[1][index]);
}

struct Circle {
  cv::Point2d center;
  double radius;
};

std::optional<Circle> fitCircle(const std::vector<RimSample>& samples) {
  cv::Matx33d lhs = cv::Matx33d::zeros();
  cv::Vec3d rhs{};
  for (const auto& sample : samples) {
    const auto p = sample.normalized;
    const cv::Vec3d row(2*p.x, 2*p.y, 1);
    lhs += row * row.t();
    rhs += row * p.dot(p);
  }
  cv::Vec3d solved{};
  if (!cv::solve(lhs, rhs, solved, cv::DECOMP_CHOLESKY)) return {};
  const cv::Point2d center(solved[0], solved[1]);
  const double radius = std::sqrt(center.dot(center) + solved[2]);
  if (!std::isfinite(radius)) return {};
  return Circle{center, radius};
}

double residual(const RimSample& sample, const Circle& fit) {
  return std::abs(cv::norm(sample.normalized - fit.center) - fit.radius);
}
} // namespace

std::optional<ReobservedTargetRing> reobserveRing(
    const cv::Mat& bgr, Color color, const cv::RotatedRect& search, bool from_shared_motion) {
  const float a = search.size.width*.5F, b = search.size.height*.5F;
  const float minor = std::min(a,b);
  if (minor<7 || !std::isfinite(a) || !std::isfinite(b)) return {};
  const float angle = search.angle*float(CV_PI/180), c = std::cos(angle), s = std::sin(angle);
  const auto toPixel = [&](cv::Point2f p) {
    return search.center + cv::Point2f(c*a*p.x-s*b*p.y, s*a*p.x+c*b*p.y);
  };
  const auto inImage = [&](cv::Point2f p) {
    return p.x>=0 && p.y>=0 && p.x<bgr.cols-1 && p.y<bgr.rows-1;
  };
  const int primary = color==Color::Blue ? 0 : 2;
  const int opposite = 2-primary;
  std::vector<RimSample> samples;
  samples.reserve(ray_count);
  for (int i=0; i<ray_count; ++i) {
    const double phi = 2*CV_PI*i/ray_count;
    const cv::Point2f unit(float(std::cos(phi)),float(std::sin(phi)));
    const float pixel_radius = float(cv::norm(toPixel(unit)-search.center));
    const float step = std::clamp(1.F/pixel_radius,.006F,.06F);
    float best_score = 0, best_radius = 0;
    for (float r=.78F; r<=1.22F; r+=step) {
      const auto inner = toPixel((r-step)*unit), outer = toPixel((r+step)*unit);
      if (!inImage(inner) || !inImage(outer)) continue;
      const float intensity = channel(bgr,inner,primary);
      const float brightness_drop = intensity-channel(bgr,outer,primary);
      const float chroma = intensity-channel(bgr,inner,opposite);
      const float outer_chroma = channel(bgr,outer,primary)-channel(bgr,outer,opposite);
      if (chroma<std::max(14.F,intensity*.125F) || brightness_drop<14 ||
          chroma-outer_chroma<8) continue;
      const float score = brightness_drop+chroma-outer_chroma;
      if (score>best_score) { best_score=score; best_radius=r; }
    }
    if (best_score>0) samples.push_back({cv::Point2d(unit)*best_radius,i});
  }
  constexpr std::size_t min_samples = 16;
  if (samples.size()<min_samples) return {};

  // 在椭圆归一化坐标中拟合。搜索框提供尺度和方向，中心偏移完全由本帧边缘求出。
  const auto first = fitCircle(samples);
  if (!first) return {};
  const double tolerance = std::max(1.25/minor,.035);
  std::erase_if(samples,[&](const auto& sample) { return residual(sample,*first)>2*tolerance; });
  if (samples.size()<min_samples) return {};
  const auto fit = fitCircle(samples);
  if (!fit || cv::norm(fit->center)>.18 || fit->radius<.82 || fit->radius>1.18) return {};

  std::array<int,4> quadrants{};
  int largest_gap = 0, previous = samples.back().ray-ray_count;
  double squared_error = 0;
  for (const auto& sample : samples) {
    const double error = residual(sample,*fit);
    squared_error += error*error;
    ++quadrants[sample.ray/(ray_count/4)];
    largest_gap = std::max(largest_gap,sample.ray-previous);
    previous=sample.ray;
  }
  const double rms = std::sqrt(squared_error/samples.size());
  if (std::count_if(quadrants.begin(),quadrants.end(),[](int n) { return n>=3; })<3 ||
      largest_gap>ray_count/2-2 || rms>tolerance) return {};
  const auto center = toPixel(cv::Point2f(fit->center));
  if (!inImage(center)) return {};
  int dark_interior = 0;
  for (const auto& sample : samples) {
    const cv::Point2d inward=fit->center+.55*(sample.normalized-fit->center);
    const auto p=toPixel(cv::Point2f(inward));
    if (!inImage(p)) continue;
    const float intensity=channel(bgr,p,primary);
    if (intensity-channel(bgr,p,opposite)<std::max(14.F,intensity*.125F)) ++dark_interior;
  }
  if (dark_interior*3<int(samples.size())) return {};
  return ReobservedTargetRing{
      {center,search.size*float(fit->radius),search.angle},
      double(samples.size())/ray_count,rms*minor,from_shared_motion};
}
} // namespace Rune::detail
