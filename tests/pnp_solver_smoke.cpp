#include "l3_estimation/pnp_solver.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace {

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {1280, 1024};
  calibration.camera_matrix = (cv::Mat_<double>(3, 3)
    << 1200.0, 0.0, 640.0,
       0.0, 1200.0, 512.0,
       0.0, 0.0, 1.0);
  calibration.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64FC1);
  return calibration;
}

std::array<cv::Point3f, 4> smallArmorPoints()
{
  return {{{0.0F,  0.0675F,  0.0275F},
           {0.0F, -0.0675F,  0.0275F},
           {0.0F, -0.0675F, -0.0275F},
           {0.0F,  0.0675F, -0.0275F}}};
}

}  // namespace

int main()
{
  const auto calibration = makeCalibration();
  L3Estimation::PnpSolver solver(calibration);

  // 把装甲板的 x 法向、y 向左、z 向上转换到相机的 x 右、y 下、z 向前。
  const cv::Matx33d rotation{
     0.0, -1.0,  0.0,
     0.0,  0.0, -1.0,
     1.0,  0.0,  0.0};
  cv::Vec3d rvec;
  cv::Rodrigues(rotation, rvec);
  const cv::Vec3d expected_tvec{0.12, -0.04, 3.2};

  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    smallArmorPoints(), rvec, expected_tvec,
    calibration.camera_matrix, calibration.distortion_coefficients, projected);

  L2Perception::ArmorDetection detection;
  std::copy(projected.begin(), projected.end(), detection.corners.begin());
  const auto pose = solver.solve(detection, L3Estimation::ArmorSize::Small);
  if (!pose) {
    std::cerr << "PnpSolver failed on synthetic armor points\n";
    return 1;
  }

  if (cv::norm(pose->tvec - expected_tvec) > 1e-4) {
    std::cerr << "PnpSolver recovered an inaccurate translation: "
              << pose->tvec << '\n';
    return 2;
  }

  L2Perception::ArmorDetection degenerate;
  degenerate.corners = {{{10.0F, 10.0F}, {20.0F, 10.0F},
                         {30.0F, 10.0F}, {40.0F, 10.0F}}};
  if (solver.solve(degenerate, L3Estimation::ArmorSize::Small)) {
    std::cerr << "PnpSolver accepted degenerate collinear corners\n";
    return 3;
  }

  std::cout << "PnpSolver smoke test passed\n";
  return 0;
}
