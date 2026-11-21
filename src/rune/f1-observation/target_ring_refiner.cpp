#include "rune/f1-observation/target_ring_refiner.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace Rune {
namespace {
bool finite(cv::Point2f p) { return std::isfinite(p.x) && std::isfinite(p.y); }

struct Ring {
  cv::RotatedRect ellipse;
  cv::Matx22f to_axes;
  bool subpixel{false};
  cv::Point2f normalized(cv::Point2f point) const {
    const auto p=to_axes*cv::Vec2f(point.x-ellipse.center.x,point.y-ellipse.center.y);
    return {p[0]/(ellipse.size.width*.5F),p[1]/(ellipse.size.height*.5F)};
  }
  float radius(cv::Point2f direction) const {
    const auto p=to_axes*cv::Vec2f(direction.x,direction.y);
    return 1.0F/std::hypot(p[0]/(ellipse.size.width*.5F),p[1]/(ellipse.size.height*.5F));
  }
  float residual(cv::Point2f point) const {
    const auto q=normalized(point);
    const float r=std::hypot(q.x,q.y);
    const float gradient=std::hypot(q.x/(ellipse.size.width*.5F),q.y/(ellipse.size.height*.5F));
    return gradient>1e-6F ? (r-1)*r/gradient : 0;
  }
};

Ring ringOf(const cv::RotatedRect& ellipse) {
  const float angle=ellipse.angle*float(CV_PI/180);
  return {ellipse,{std::cos(angle),std::sin(angle),-std::sin(angle),std::cos(angle)}};
}

bool finiteRing(const Ring& ring) {
  return finite(ring.ellipse.center) && std::isfinite(ring.ellipse.angle) &&
      std::isfinite(ring.ellipse.size.width) && std::isfinite(ring.ellipse.size.height) &&
      ring.ellipse.size.width>2 && ring.ellipse.size.height>2;
}

float sample(const cv::Mat& image,cv::Point2f p) {
  if (!finite(p) || p.x<0 || p.y<0 || p.x>=image.cols-1 || p.y>=image.rows-1) return 0;
  const int x=int(p.x),y=int(p.y);
  const float dx=p.x-x,dy=p.y-y;
  const float* a=image.ptr<float>(y),*b=image.ptr<float>(y+1);
  return (1-dy)*((1-dx)*a[x]+dx*a[x+1])+dy*((1-dx)*b[x]+dx*b[x+1]);
}

float median(std::vector<float> values) {
  const auto middle=values.begin()+values.size()/2;
  std::nth_element(values.begin(),middle,values.end());
  return *middle;
}

bool supportedAngles(const Ring& ring,const std::vector<cv::Point2f>& points,int minimum=24) {
  std::array<bool,32> supported{};
  for (const auto& p : points) {
    const auto q=ring.normalized(p);
    const int bin=std::clamp(int((std::atan2(q.y,q.x)+CV_PI)*32/(2*CV_PI)),0,31);
    supported[bin]=true;
  }
  return std::count(supported.begin(),supported.end(),true)>=minimum;
}

bool compatibleRim(const Ring& fit,const Ring& coarse) {
  if (!finiteRing(fit)) return false;
  const float a=coarse.ellipse.size.width,b=coarse.ellipse.size.height;
  const float aa=fit.ellipse.size.width,bb=fit.ellipse.size.height;
  const float scale=std::sqrt(aa*bb/(a*b));
  return aa>=.82F*a && aa<=1.08F*a && bb>=.82F*b && bb<=1.08F*b &&
      scale>=.88F && scale<=1.04F &&
      cv::norm(fit.ellipse.center-coarse.ellipse.center)<=std::max(1.0F,.05F*std::min(a,b));
}

std::optional<Ring> fitRidges(const std::vector<cv::Point2f>& points,const Ring& coarse) {
  // Select the outside support of measured LED centers. Interior LED rows and
  // spokes may yield several ridges per normal; their coordinates must not be
  // averaged into the outer boundary. These are photometric centers, not the
  // threshold-dependent extreme pixels used by the coarse verifier.
  std::vector<cv::Point2f> hull;
  cv::convexHull(points,hull);
  if (hull.size()<12) return {};
  const int minimum_bins=std::min(16,int(hull.size())-2);
  auto fit=ringOf(cv::fitEllipse(hull));
  if (!finiteRing(fit)) return {};
  const float radius=std::min(coarse.ellipse.size.width,coarse.ellipse.size.height)*.5F;
  std::vector<cv::Point2f> inliers;
  for (int iteration=0;iteration<4;++iteration) {
    std::vector<float> errors;
    for (const auto& p : hull) errors.push_back(fit.residual(p));
    const float center=median(errors);
    std::vector<float> deviations;
    for (const auto error : errors) deviations.push_back(std::abs(error-center));
    const float gate=std::clamp(2.5F*1.4826F*median(deviations),.75F,std::max(.75F,.04F*radius));
    inliers.clear();
    for (std::size_t i=0;i<hull.size();++i)
      if (std::abs(errors[i]-center)<=gate) inliers.push_back(hull[i]);
    if (inliers.size()<10 || inliers.size()<.7*hull.size() ||
        !supportedAngles(fit,inliers,minimum_bins)) return {};
    fit=ringOf(cv::fitEllipse(inliers));
    if (!finiteRing(fit)) return {};
  }
  if (!compatibleRim(fit,coarse)) return {};
  double sum=0;
  for (const auto& p : inliers) {
    const double error=fit.residual(p);sum+=error*error;
  }
  const float max_rms=std::max(.8F,.025F*radius);
  if (!std::isfinite(sum) || sum/inliers.size()>max_rms*max_rms ||
      !supportedAngles(fit,inliers,minimum_bins)) return {};
  fit.subpixel=true;
  return fit;
}

std::optional<Ring> polishRing(const cv::Mat& signal,const cv::Mat& chroma,
                              const Ring& coarse,double color_threshold) {
  const float a=coarse.ellipse.size.width*.5F,b=coarse.ellipse.size.height*.5F;
  const float inward=std::clamp(.10F*std::min(a,b)+3,4.0F,12.0F);
  const float outward=std::clamp(.05F*std::min(a,b),2.0F,5.0F);
  constexpr float step=.5F;
  const int samples=int(std::ceil((inward+outward)/step))+1;
  std::vector<cv::Point2f> points;
  points.reserve(128);
  for (int i=0;i<128;++i) {
    const float theta=float(2*CV_PI*(i+.5)/128),c=std::cos(theta),s=std::sin(theta);
    const auto offset=coarse.to_axes.t()*cv::Vec2f(a*c,b*s);
    const cv::Point2f origin=coarse.ellipse.center+cv::Point2f(offset[0],offset[1]);
    auto normal=coarse.to_axes.t()*cv::Vec2f(c/a,s/b);
    normal*=1.0F/std::hypot(normal[0],normal[1]);
    const cv::Point2f n(normal[0],normal[1]),t(-normal[1],normal[0]);
    std::vector<float> profile(samples),colors(samples);
    for (int j=0;j<samples;++j) {
      const auto p=origin+n*(-inward+j*step);
      profile[j]=.5F*sample(signal,p)+.25F*(sample(signal,p-t)+sample(signal,p+t));
      colors[j]=std::max({sample(chroma,p),sample(chroma,p-t),sample(chroma,p+t)});
    }
    auto sorted=profile;
    const auto lower=sorted.begin()+samples/5;
    std::nth_element(sorted.begin(),lower,sorted.end());
    const float background=*lower,peak=*std::max_element(profile.begin(),profile.end());
    if (peak-background<20) continue;
    auto smooth=profile;
    for (int j=1;j<samples-1;++j)
      smooth[j]=.25F*profile[j-1]+.5F*profile[j]+.25F*profile[j+1];
    // Retain locally separated bright ridges in the verified outer band.
    // Outside support selection and the geometric fit resolve competing peaks.
    for (int top=samples-2;top>0;--top) {
      if (smooth[top]<smooth[top+1] || smooth[top]<=smooth[top-1]) continue;
      int left_valley=top-1,right_valley=top+1;
      while (left_valley>0 && smooth[left_valley-1]<=smooth[left_valley]) --left_valley;
      while (right_valley<samples-1 && smooth[right_valley+1]<=smooth[right_valley]) ++right_valley;
      const float prominence=smooth[top]-std::max(smooth[left_valley],smooth[right_valley]);
      if (prominence<std::max(12.0F,.08F*smooth[top])) continue;
      const float left_level=.5F*(smooth[top]+smooth[left_valley]);
      const float right_level=.5F*(smooth[top]+smooth[right_valley]);
      int left=top,right=top;
      while (left>left_valley && smooth[left-1]>=left_level) --left;
      while (right<right_valley && smooth[right+1]>=right_level) ++right;
      if (left==0 || right==samples-1) continue;
      const float lo=left-1+(left_level-smooth[left-1])/(smooth[left]-smooth[left-1]);
      const float hi=right+(smooth[right]-right_level)/(smooth[right]-smooth[right+1]);
      const float width=(hi-lo)*step;
      const float color=*std::max_element(colors.begin()+left,colors.begin()+right+1);
      if (width>=1 && width<=std::min(8.0F,.12F*std::min(a,b)+2) && color>=color_threshold) {
        // Interpolated half-height edge midpoint handles a flat/saturated LED
        // core without fitting to the threshold-dependent outer halo boundary.
        points.push_back(origin+n*(-inward+(lo+hi)*.5F*step));
      }
    }
  }
  if (points.size()<40 || !supportedAngles(coarse,points)) return {};
  return fitRidges(points,coarse);
}

std::optional<Ring> findRing(const cv::Mat& bgr,cv::Point2f center,
                             cv::Point2f radial,double expected_radius,bool subpixel,Ring& coarse,bool& reconnected) {
  // Bound all work in original pixels, including malformed/very near targets.
  if (expected_radius<6 || expected_radius>200 || !finite(center)) return {};
  const int half=cvCeil(1.6*expected_radius);
  if (center.x<half || center.y<half || center.x+half>=bgr.cols || center.y+half>=bgr.rows)
    return {}; // A clipped ring cannot establish its full physical diameter.
  const cv::Rect roi(cvRound(center.x)-half,cvRound(center.y)-half,2*half+1,2*half+1);
  if ((roi & cv::Rect(0,0,bgr.cols,bgr.rows))!=roi) return {};
  std::vector<cv::Mat> channels;
  cv::split(bgr(roi),channels);
  cv::Mat difference,mask,closed;
  // The detector has no enemy-color input. Both red and blue LED rings are
  // usable; neutral reflections do not provide chromatic ring evidence.
  cv::absdiff(channels[0],channels[2],difference);
  cv::GaussianBlur(difference,difference,{3,3},0);
  const double threshold=cv::threshold(difference,mask,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
  if (threshold<12) return {};
  std::vector<cv::Point> bright;
  cv::findNonZero(mask,bright);
  std::optional<Ring> best;
  double best_area=0;
  const int base_kernel=std::max(3,2*int(expected_radius*.06)+1);
  const int recovery_kernel=std::max(base_kernel+2,2*int(expected_radius*.10)+1);
  // Sparse LEDs and short dark gaps can split a complete rim into several
  // contours. Retry a bounded, larger closing only when the usual fit fails.
  // All coverage/fill checks still use unclosed pixels, never the bridge pixels.
  for (const int kernel : {base_kernel,recovery_kernel}) {
    cv::morphologyEx(mask,closed,cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE,{kernel,kernel}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      if (contour.size()<8) continue;
      std::vector<cv::Point> hull;
      cv::convexHull(contour,hull);
      if (hull.size()<8) continue;
      const auto ellipse=cv::fitEllipse(hull);
      const double a=ellipse.size.width*.5,b=ellipse.size.height*.5;
      if (!finite(ellipse.center) || !std::isfinite(a) || !std::isfinite(b) ||
          std::min(a,b)<.45*expected_radius || std::max(a,b)>1.4*expected_radius ||
          std::min(a,b)<.45*std::max(a,b) ||
          cv::norm(ellipse.center+cv::Point2f(roi.tl())-center)>.3*expected_radius) continue;
      auto ring=ringOf(ellipse);
      // Use the radial diameter, so camera foreshortening along the tangential
      // direction alone cannot make us select a smaller concentric ring.
      const double ratio=ring.radius(radial)/expected_radius;
      if (ratio<.85 || ratio>1.15) continue;
      double squared_error=0;
      for (const auto& p : hull) {
        const double error=cv::norm(ring.normalized(cv::Point2f(p)))-1;
        squared_error+=error*error;
      }
      if (squared_error/hull.size()>.01) continue;
      // Morphology/hulls are only fitting aids. Require actual, unclosed LED
      // pixels around at least 3/4 of the rim; an arc or solid blob is insufficient.
      std::array<bool,32> supported{};
      int inside=0,near=0;
      for (const auto& p : bright) {
        const auto q=ring.normalized(cv::Point2f(p));
        const double radius=cv::norm(q);
        if (radius<.8) ++inside;
        if (radius<.90 || radius>1.04) continue;
        ++near;
        const int bin=std::clamp(int((std::atan2(q.y,q.x)+CV_PI)*32/(2*CV_PI)),0,31);
        supported[bin]=true;
      }
      if (std::count(supported.begin(),supported.end(),true)<24 || near<24 ||
          inside>.75*CV_PI*a*b*.64) continue;
      if (a*b>best_area) {
        best=ring; best_area=a*b;
      }
    }
    if (best) {
      reconnected=kernel!=base_kernel;
      break;
    }
  }
  if (best) {
    coarse=*best;
    coarse.ellipse.center+=cv::Point2f(roi.tl());
  }
  if (best && subpixel) {
    cv::Mat signal,chroma;
    cv::max(channels[0],channels[2],signal);signal.convertTo(signal,CV_32F);
    cv::absdiff(channels[0],channels[2],chroma);chroma.convertTo(chroma,CV_32F);
    try {
      if (const auto fit=polishRing(signal,chroma,*best,std::max(12.0,.5*threshold))) {
        const double ratio=fit->radius(radial)/expected_radius;
        if (ratio>=.85 && ratio<=1.15) best=fit;
      }
    } catch (const cv::Exception&) { /* Retain the already verified coarse rim. */ }
  }
  if (best) best->ellipse.center+=cv::Point2f(roi.tl());
  return best;
}
} // namespace

TargetRingResult refineTargetRing(const cv::Mat& bgr,Keypoints& k,Mode mode,double radius_ratio,bool subpixel,
                                  TargetRingDiagnostics* diagnostics) {
  TargetRingDiagnostics detail;
  if (diagnostics) *diagnostics=detail;
  k.target_ring_subpixel=false;
  k.target_ring_coarse_points={};
  // Big-rune activated blades may have only an arm, with no target ring.
  if (mode==Mode::Big && k.class_id==1) {
    k.target_ring_status=TargetRingStatus::Unchecked;
    return TargetRingResult::Skipped;
  }
  const auto reject=[&](TargetRingRejectReason reason=TargetRingRejectReason::InvalidInput) {
    detail.rejection=reason;
    if (diagnostics) *diagnostics=detail;
    std::fill_n(k.point_valid.begin(),4,false);
    k.target_ring_status=TargetRingStatus::Rejected;
    return TargetRingResult::Rejected;
  };
  const auto center=k.armorCenter();
  if (bgr.empty() || bgr.type()!=CV_8UC3 || !center || !finite(*center) ||
      std::count(k.point_valid.begin(),k.point_valid.begin()+4,true)<3 ||
      !k.point_valid[4] || !finite(k.center_r) ||
      !std::isfinite(radius_ratio) || radius_ratio<=0 || radius_ratio>=.5) return reject();
  auto radial=*center-k.center_r;
  const double distance=cv::norm(radial);
  if (distance<1) return reject();
  radial*=float(1/distance);
  std::optional<Ring> ring;
  Ring coarse;
  try { ring=findRing(bgr,*center,radial,distance*radius_ratio,subpixel,coarse,detail.reconnected); }
  catch (const cv::Exception&) { return reject(TargetRingRejectReason::NoSupportedContour); }
  if (!ring) return reject(TargetRingRejectReason::NoSupportedContour);
  // Preserve projected semantic directions (they need not be orthogonal under
  // perspective). NN points choose directions, never which concentric radius.
  auto vertical=k.point_valid[0] && k.point_valid[3] ? k.top-k.bottom
      : k.point_valid[0] ? k.top-*center : *center-k.bottom;
  auto horizontal=k.point_valid[1] && k.point_valid[2] ? k.right-k.left
      : k.point_valid[2] ? k.right-*center : *center-k.left;
  if (!finite(vertical) || !finite(horizontal) || cv::norm(vertical)<1 || cv::norm(horizontal)<1)
    return reject(TargetRingRejectReason::InvalidDirections);
  vertical*=float(1/cv::norm(vertical)); horizontal*=float(1/cv::norm(horizontal));
  if (vertical.dot(radial)<.8 || std::abs(vertical.cross(horizontal))<.5) return reject(TargetRingRejectReason::InvalidDirections);
  const std::array<cv::Point2f,4> directions{vertical,-horizontal,horizontal,-vertical};
  std::array<cv::Point2f,4> corrected;
  for (std::size_t i=0;i<corrected.size();++i) {
    corrected[i]=ring->ellipse.center+directions[i]*ring->radius(directions[i]);
    if (!finite(corrected[i]) || corrected[i].x<0 || corrected[i].y<0 ||
        corrected[i].x>=bgr.cols || corrected[i].y>=bgr.rows) return reject(TargetRingRejectReason::OutOfImage);
  }
  if (subpixel) for (std::size_t i=0;i<corrected.size();++i)
    k.target_ring_coarse_points[i]=coarse.ellipse.center+directions[i]*coarse.radius(directions[i]);
  std::array<cv::Point2f*,4> points{&k.top,&k.left,&k.right,&k.bottom};
  for (std::size_t i=0;i<points.size();++i) if (k.point_valid[i]) *points[i]=corrected[i];
  k.target_ring_status=TargetRingStatus::Verified;
  k.target_ring_subpixel=ring->subpixel;
  if (diagnostics) *diagnostics=detail;
  return TargetRingResult::Refined;
}
} // namespace Rune
