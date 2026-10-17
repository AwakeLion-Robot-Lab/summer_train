// Private implementation shared by the building translation units.
#pragma once

#include "rune/types.hpp"
#include "geometry/pose_projection.hpp"
#include "geometry/contour.hpp"

#include <array>
#include <limits>
#include <optional>
#include <vector>

namespace Rune::building_detail {

using Parameters = tools::geometry::PoseParameters;
using tools::geometry::finite;
using tools::geometry::rigid;
using tools::geometry::rotation;
using tools::geometry::inFront;
using tools::geometry::project;
using tools::geometry::appendContourSamples;

struct Field {
  tools::geometry::ContourDistanceField image;
  std::vector<cv::Point3d> model;
};

struct Evaluation {
  Eigen::VectorXd residual;
  double cost{std::numeric_limits<double>::infinity()};
  double anchor_rms{0}, chamfer_rms{0};
};

struct Prepared {
  std::size_t index;
  std::vector<cv::Point3d> anchors;
  std::vector<cv::Point2d> pixels;
  std::vector<Field> fields;
  bool pca{false};
};

bool validGeometry(const Geometry& g);
bool validCamera(const Frame& frame, cv::Mat& camera, cv::Mat& distortion);
bool worldRotation(const Frame& frame, std::optional<Eigen::Matrix3d>& rotation);
bool validObservation(const Observation& observation, bool arm_only);
bool validPcaBox(const std::vector<cv::Point3d>& box);

std::vector<cv::Point2d> networkPoints(const Observation& observation);
cv::Point3d rotatePoint(const cv::Point3d& point, int sector);
std::vector<cv::Point3d> rotatePoints(const std::vector<cv::Point3d>& points, int sector);
void updatePhase(Pose& pose);

std::optional<std::vector<cv::Point2d>> pcaEnvelope(const Observation& observation,
    const cv::Mat& camera, const cv::Mat& distortion);
std::optional<Field> makeField(const std::vector<cv::Point>& contour,
    const std::vector<cv::Point3d>& model, const Geometry& geometry);
Evaluation evaluate(const Parameters& p, const std::vector<cv::Point3d>& anchors,
    const std::vector<cv::Point2d>& image, const std::vector<Field>& fields,
    const cv::Mat& camera, const cv::Mat& distortion, const Geometry& geometry,
    const std::optional<Eigen::Vector3d>& vertical);
void refine(Parameters& p, Evaluation& evaluation, const std::vector<cv::Point3d>& anchors,
    const std::vector<cv::Point2d>& image, const std::vector<Field>& fields,
    const cv::Mat& camera, const cv::Mat& distortion, const Geometry& geometry,
    const std::optional<Eigen::Vector3d>& vertical);
bool consistentFields(const std::vector<Field>& fields, const Parameters& p,
    const cv::Mat& camera, const cv::Mat& distortion, const Geometry& geometry);

} // namespace Rune::building_detail
