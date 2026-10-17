// Algorithm reference: SZURPVision/RP-26Rune, PowerRunePlane.
// See README.md and RP-26Rune-LICENSE in this directory.
// R/inactive-seed layout selection also references Alliance-Algorithm/
// rmcs_auto_aim_v2 (MIT); see ../f3-estimator/ALLIANCE-LICENSE.
#include "rune/f2-building/builder_detail.hpp"

#include <opencv2/imgproc.hpp>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace Rune::building_detail {

std::optional<Field> makeField(const std::vector<cv::Point>& contour,
                              const std::vector<cv::Point3d>& model,
                              const Geometry& g) {
  auto image=tools::geometry::makeContourDistanceField(
    contour,g.distance_field_padding_px,g.max_distance_field_pixels);
  if (!image) return {};
  return Field{std::move(*image),model};
}
Evaluation evaluate(const Parameters& p, const std::vector<cv::Point3d>& anchors,
                    const std::vector<cv::Point2d>& image, const std::vector<Field>& fields,
                    const cv::Mat& camera, const cv::Mat& distortion, const Geometry& g,
                    const std::optional<Eigen::Vector3d>& vertical) {
  if (!inFront(anchors,p)) return {};
  std::size_t count = 2*image.size()+(vertical && g.normal_vertical_weight_px>0 ? 1 : 0);
  for (const auto& f : fields) {
    if (!inFront(f.model,p)) return {};
    count += f.model.size();
  }
  Evaluation e;
  e.residual.resize(count);
  e.cost = 0;
  int index = 0;
  const auto append = [&](double raw, double weight) {
    // Signed square root of the Huber loss: LM minimizes this robust objective.
    const double a = std::abs(raw), delta = g.huber_delta_px;
    const double robust = a <= delta ? raw : std::copysign(std::sqrt(2*delta*a-delta*delta), raw);
    e.residual[index++] = weight*robust;
  };
  const auto pixels = project(anchors,p,camera,distortion);
  double sum = 0;
  for (std::size_t i=0; i<image.size(); ++i) {
    const auto delta = pixels[i]-image[i];
    sum += delta.dot(delta);
    append(delta.x,1); append(delta.y,1);
  }
  e.anchor_rms = std::sqrt(sum/image.size());
  double contour_sum = 0;
  const double cap = std::max(10.0,4*g.max_chamfer_error_px);
  for (const auto& f : fields) {
    double field_sum = 0;
    // Normalize each semantic class so dense arm samples cannot drown out R/armor.
    const double weight = std::sqrt(image.size()*g.contour_weight/(fields.size()*f.model.size()));
    for (const auto& pixel : project(f.model,p,camera,distortion)) {
      const double distance = tools::geometry::sampleDistanceField(f.image,pixel,cap);
      field_sum += distance*distance;
      append(distance,weight);
    }
    contour_sum += field_sum/f.model.size();
  }
  e.chamfer_rms = fields.empty() ? 0 : std::sqrt(contour_sum/fields.size());
  if (vertical && g.normal_vertical_weight_px>0) {
    const auto r=rotation(p);
    const Eigen::Vector3d normal(r(0,2),r(1,2),r(2,2));
    e.residual[index++]=std::sqrt(image.size()/5.0)*g.normal_vertical_weight_px*normal.dot(*vertical);
  }
  if (!e.residual.allFinite()) return {};
  e.cost = e.residual.squaredNorm();
  return e;
}
void refine(Parameters& p, Evaluation& e, const std::vector<cv::Point3d>& anchors,
            const std::vector<cv::Point2d>& image, const std::vector<Field>& fields,
            const cv::Mat& k, const cv::Mat& d, const Geometry& g,
            const std::optional<Eigen::Vector3d>& vertical) {
  double lambda = 1e-3;
  for (int iteration=0; iteration<g.refinement_iterations; ++iteration) {
    Eigen::MatrixXd jacobian(e.residual.size(),6);
    for (int j=0; j<6; ++j) {
      const double step = 1e-5*std::max(1.0,std::abs(p[j]));
      Parameters plus=p, minus=p;
      plus[j]+=step; minus[j]-=step;
      const auto a=evaluate(plus,anchors,image,fields,k,d,g,vertical);
      const auto b=evaluate(minus,anchors,image,fields,k,d,g,vertical);
      if (a.residual.size()!=e.residual.size() || b.residual.size()!=e.residual.size()) return;
      jacobian.col(j)=(a.residual-b.residual)/(2*step);
    }
    Eigen::Matrix<double,6,6> h = jacobian.transpose()*jacobian;
    const Parameters gradient = jacobian.transpose()*e.residual;
    for (int j=0; j<6; ++j) h(j,j) += lambda*std::max(1.0,h(j,j));
    const Parameters delta = -h.ldlt().solve(gradient);
    if (!delta.allFinite() || delta.norm()<1e-8) break;
    const Parameters trial=p+delta;
    auto next=evaluate(trial,anchors,image,fields,k,d,g,vertical);
    if (next.cost < e.cost && next.anchor_rms <= g.max_reprojection_error_px) {
      const double improvement=e.cost-next.cost;
      p=trial; e=std::move(next);
      lambda=std::max(1e-9,lambda*0.3);
      if (improvement < 1e-7) break;
    } else {
      lambda*=10;
      if (lambda>1e9) break;
    }
  }
}
bool consistentFields(const std::vector<Field>& fields,const Parameters& p,
                      const cv::Mat& camera,const cv::Mat& distortion,const Geometry& g) {
  const double cap=std::max(10.0,4*g.max_chamfer_error_px);
  for (const auto& field : fields) {
    double sum=0;
    for (const auto& pixel : project(field.model,p,camera,distortion)) {
      const double distance=tools::geometry::sampleDistanceField(field.image,pixel,cap); sum+=distance*distance;
    }
    if (std::sqrt(sum/field.model.size())>g.max_chamfer_error_px) return false;
  }
  return true;
}

} // namespace Rune::building_detail
