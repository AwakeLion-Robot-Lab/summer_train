#include "tools/camera_calibration/high_precision_calibrator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace tools {
namespace {

constexpr double kOutlierAbsoluteFloor = 0.25;
constexpr double kMaxFinalRms = 0.25;
constexpr double kMaxPerViewP95 = 0.35;
constexpr double kMaxCrossValidationRms = 0.30;
constexpr double kMaxFocalRelativeStdDev = 0.005;
constexpr double kMaxPrincipalPointStdDev = 2.0;
constexpr double kModelImprovementRatio = 0.02;
constexpr int kCalibrationIterations = 200;
constexpr double kCalibrationEpsilon = 1e-12;
constexpr char kPreviewWindow[] = "camera_calibrator";

struct Observation {
  std::filesystem::path path;
  std::vector<cv::Point2f> corners;
  std::vector<double> descriptor;
  cv::Point2d center_normalized{};
  double area_ratio{};
  double sharpness{};
  double contrast{};
  double reprojection_error{std::numeric_limits<double>::quiet_NaN()};
  bool detected{};
  bool quality_passed{};
  bool selected{};
  bool used{};
  std::string rejection_reason;
};

struct FitResult {
  double rms{std::numeric_limits<double>::quiet_NaN()};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  cv::Mat intrinsic_std_deviations;
  std::vector<cv::Mat> rotation_vectors;
  std::vector<cv::Mat> translation_vectors;
  std::vector<double> per_view_errors;
};

struct BootstrapResult {
  int requested{};
  int succeeded{};
  std::vector<double> parameter_std_deviations;
};

double finiteOrZero(double value)
{
  return std::isfinite(value) ? value : 0.0;
}

std::string lowerCase(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  return value;
}

bool naturalLess(
  const std::filesystem::path& left_path,
  const std::filesystem::path& right_path)
{
  const std::string left = lowerCase(left_path.filename().string());
  const std::string right = lowerCase(right_path.filename().string());
  std::size_t left_index = 0;
  std::size_t right_index = 0;

  while (left_index < left.size() && right_index < right.size()) {
    const bool left_digit =
      std::isdigit(static_cast<unsigned char>(left[left_index])) != 0;
    const bool right_digit =
      std::isdigit(static_cast<unsigned char>(right[right_index])) != 0;
    if (left_digit && right_digit) {
      std::size_t left_end = left_index;
      std::size_t right_end = right_index;
      while (left_end < left.size()
             && std::isdigit(
               static_cast<unsigned char>(left[left_end])) != 0) {
        ++left_end;
      }
      while (right_end < right.size()
             && std::isdigit(
               static_cast<unsigned char>(right[right_end])) != 0) {
        ++right_end;
      }

      std::size_t left_significant = left_index;
      std::size_t right_significant = right_index;
      while (left_significant + 1 < left_end
             && left[left_significant] == '0') {
        ++left_significant;
      }
      while (right_significant + 1 < right_end
             && right[right_significant] == '0') {
        ++right_significant;
      }

      const std::size_t left_digits = left_end - left_significant;
      const std::size_t right_digits = right_end - right_significant;
      if (left_digits != right_digits) {
        return left_digits < right_digits;
      }
      const int number_compare = left.compare(
        left_significant, left_digits,
        right, right_significant, right_digits);
      if (number_compare != 0) {
        return number_compare < 0;
      }
      left_index = left_end;
      right_index = right_end;
      continue;
    }

    if (left[left_index] != right[right_index]) {
      return left[left_index] < right[right_index];
    }
    ++left_index;
    ++right_index;
  }
  return left.size() < right.size();
}

std::vector<std::filesystem::path> discoverImages(
  const std::filesystem::path& input_directory)
{
  std::error_code error;
  if (!std::filesystem::is_directory(input_directory, error)) {
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to inspect calibration input directory",
        input_directory, error);
    }
    throw std::invalid_argument(
      "calibration input is not a directory: "
      + input_directory.string());
  }

  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator iterator{
         input_directory, error};
       !error && iterator != std::filesystem::directory_iterator{};
       iterator.increment(error)) {
    if (!iterator->is_regular_file()) {
      continue;
    }
    const std::string extension =
      lowerCase(iterator->path().extension().string());
    if (extension == ".png" || extension == ".jpg"
        || extension == ".jpeg") {
      paths.push_back(iterator->path());
    }
  }
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to enumerate calibration images", input_directory, error);
  }

  std::sort(paths.begin(), paths.end(), naturalLess);
  if (paths.empty()) {
    throw std::runtime_error(
      "no PNG or JPEG images found in: " + input_directory.string());
  }
  return paths;
}

void validateOptions(const CameraCalibratorOptions& options)
{
  if (options.input_directory.empty()) {
    throw std::invalid_argument("calibration input directory is empty");
  }
  if (options.output_root.empty()) {
    throw std::invalid_argument("calibration output directory is empty");
  }
  if (options.pattern_size.width < 2 || options.pattern_size.height < 2) {
    throw std::invalid_argument(
      "checkerboard inner-corner dimensions must both be at least 2");
  }
  if (!std::isfinite(options.square_size) || options.square_size <= 0.0) {
    throw std::invalid_argument(
      "checkerboard square size must be a finite positive number");
  }
  if (options.min_views < 10) {
    throw std::invalid_argument(
      "minimum calibration view count must be at least 10");
  }
  if (options.max_views < options.min_views) {
    throw std::invalid_argument(
      "maximum calibration view count must not be below minimum view count");
  }
  if (!std::isfinite(options.max_sharpness)
      || options.max_sharpness <= 0.0) {
    throw std::invalid_argument(
      "maximum checkerboard sharpness must be finite and positive");
  }
  if (!std::isfinite(options.min_contrast)
      || options.min_contrast < 0.0 || options.min_contrast > 255.0) {
    throw std::invalid_argument(
      "minimum checkerboard contrast must be between 0 and 255");
  }
  if (!std::isfinite(options.min_board_area_ratio)
      || options.min_board_area_ratio <= 0.0
      || options.min_board_area_ratio >= 1.0) {
    throw std::invalid_argument(
      "minimum board area ratio must be between 0 and 1");
  }
  if (options.bootstrap_iterations < 0
      || options.bootstrap_iterations > 1000) {
    throw std::invalid_argument(
      "bootstrap iteration count must be between 0 and 1000");
  }

  std::error_code error;
  std::filesystem::create_directories(options.output_root, error);
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to create calibration output root",
      options.output_root, error);
  }
  if (!std::filesystem::is_directory(options.output_root, error)) {
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to inspect calibration output root",
        options.output_root, error);
    }
    throw std::invalid_argument(
      "calibration output root is not a directory: "
      + options.output_root.string());
  }
}

std::vector<cv::Point3f> makeObjectPoints(
  cv::Size pattern_size,
  double square_size)
{
  std::vector<cv::Point3f> points;
  points.reserve(
    static_cast<std::size_t>(pattern_size.width * pattern_size.height));
  for (int row = 0; row < pattern_size.height; ++row) {
    for (int column = 0; column < pattern_size.width; ++column) {
      points.emplace_back(
        static_cast<float>(column * square_size),
        static_cast<float>(row * square_size),
        0.0F);
    }
  }
  return points;
}

double pointDistance(const cv::Point2f& left, const cv::Point2f& right)
{
  const cv::Point2f difference = left - right;
  return std::hypot(
    static_cast<double>(difference.x),
    static_cast<double>(difference.y));
}

void calculateGeometry(
  Observation& observation,
  cv::Size pattern_size,
  cv::Size image_size)
{
  cv::Point2d center{};
  for (const auto& corner : observation.corners) {
    center.x += corner.x;
    center.y += corner.y;
  }
  center.x /= static_cast<double>(observation.corners.size());
  center.y /= static_cast<double>(observation.corners.size());
  observation.center_normalized = {
    center.x / image_size.width,
    center.y / image_size.height};

  std::vector<cv::Point2f> hull;
  cv::convexHull(observation.corners, hull);
  observation.area_ratio =
    std::abs(cv::contourArea(hull))
    / static_cast<double>(image_size.area());

  const std::size_t top_left = 0;
  const std::size_t top_right =
    static_cast<std::size_t>(pattern_size.width - 1);
  const std::size_t bottom_left =
    static_cast<std::size_t>(
      (pattern_size.height - 1) * pattern_size.width);
  const std::size_t bottom_right =
    observation.corners.size() - 1;

  const auto& tl = observation.corners[top_left];
  const auto& tr = observation.corners[top_right];
  const auto& bl = observation.corners[bottom_left];
  const auto& br = observation.corners[bottom_right];
  const double top_length = pointDistance(tl, tr);
  const double bottom_length = pointDistance(bl, br);
  const double left_length = pointDistance(tl, bl);
  const double right_length = pointDistance(tr, br);
  const double roll = std::atan2(
    static_cast<double>(tr.y - tl.y),
    static_cast<double>(tr.x - tl.x));
  const double horizontal_perspective =
    (bottom_length - top_length)
    / std::max(bottom_length + top_length, 1e-9);
  const double vertical_perspective =
    (right_length - left_length)
    / std::max(right_length + left_length, 1e-9);

  observation.descriptor = {
    observation.center_normalized.x,
    observation.center_normalized.y,
    std::log(std::max(observation.area_ratio, 1e-12)),
    std::cos(roll),
    std::sin(roll),
    horizontal_perspective,
    vertical_perspective,
    tl.x / image_size.width,
    tl.y / image_size.height,
    tr.x / image_size.width,
    tr.y / image_size.height,
    br.x / image_size.width,
    br.y / image_size.height,
    bl.x / image_size.width,
    bl.y / image_size.height};
}

cv::Mat makePreview(
  const cv::Mat& image,
  const Observation& observation,
  cv::Size pattern_size)
{
  cv::Mat preview;
  if (image.channels() == 1) {
    cv::cvtColor(image, preview, cv::COLOR_GRAY2BGR);
  } else {
    preview = image.clone();
  }
  if (observation.detected) {
    cv::drawChessboardCorners(
      preview, pattern_size, observation.corners, true);
  }

  std::ostringstream line;
  line << observation.path.filename().string();
  if (observation.detected) {
    line << " sharpness=" << std::fixed << std::setprecision(2)
         << observation.sharpness
         << " contrast=" << std::setprecision(1)
         << observation.contrast
         << " area=" << std::setprecision(2)
         << observation.area_ratio * 100.0 << "%";
  } else {
    line << " corners not found";
  }
  cv::putText(
    preview, line.str(), {20, 35},
    cv::FONT_HERSHEY_SIMPLEX, 0.7,
    observation.quality_passed
      ? cv::Scalar{0, 255, 0}
      : cv::Scalar{0, 0, 255},
    2, cv::LINE_AA);
  return preview;
}

std::vector<Observation> detectObservations(
  const std::vector<std::filesystem::path>& paths,
  const CameraCalibratorOptions& options,
  cv::Size& image_size)
{
  std::vector<Observation> observations;
  observations.reserve(paths.size());
  const int detection_flags =
    cv::CALIB_CB_NORMALIZE_IMAGE
    | cv::CALIB_CB_EXHAUSTIVE
    | cv::CALIB_CB_ACCURACY;

  for (const auto& path : paths) {
    Observation observation;
    observation.path = path;
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      observation.rejection_reason = "unable to read image";
      observations.push_back(std::move(observation));
      continue;
    }
    if (image.depth() != CV_8U
        || (image.channels() != 1 && image.channels() != 3
            && image.channels() != 4)) {
      observation.rejection_reason =
        "unsupported image type; expected 8-bit gray, BGR, or BGRA";
      observations.push_back(std::move(observation));
      continue;
    }
    if (image_size.empty()) {
      image_size = image.size();
    }
    if (image.size() != image_size) {
      std::ostringstream reason;
      reason << "image size " << image.cols << 'x' << image.rows
             << " differs from calibration size "
             << image_size.width << 'x' << image_size.height;
      observation.rejection_reason = reason.str();
      observations.push_back(std::move(observation));
      continue;
    }

    cv::Mat gray;
    if (image.channels() == 1) {
      gray = image;
    } else if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }

    observation.detected = cv::findChessboardCornersSB(
      gray, options.pattern_size, observation.corners, detection_flags);
    if (!observation.detected) {
      std::ostringstream reason;
      reason << options.pattern_size.width << 'x'
             << options.pattern_size.height
             << " inner corners not found";
      observation.rejection_reason = reason.str();
    } else {
      const cv::Scalar horizontal = cv::estimateChessboardSharpness(
        gray, options.pattern_size, observation.corners, 0.8F, false);
      const cv::Scalar vertical = cv::estimateChessboardSharpness(
        gray, options.pattern_size, observation.corners, 0.8F, true);
      observation.sharpness = std::max(horizontal[0], vertical[0]);
      observation.contrast = std::min(
        horizontal[2] - horizontal[1],
        vertical[2] - vertical[1]);
      calculateGeometry(
        observation, options.pattern_size, image_size);

      if (!std::isfinite(observation.sharpness)
          || observation.sharpness > options.max_sharpness) {
        std::ostringstream reason;
        reason << "edge transition " << std::fixed << std::setprecision(3)
               << observation.sharpness << " px exceeds "
               << options.max_sharpness << " px";
        observation.rejection_reason = reason.str();
      } else if (!std::isfinite(observation.contrast)
                 || observation.contrast < options.min_contrast) {
        std::ostringstream reason;
        reason << "contrast " << std::fixed << std::setprecision(1)
               << observation.contrast << " is below "
               << options.min_contrast;
        observation.rejection_reason = reason.str();
      } else if (!std::isfinite(observation.area_ratio)
                 || observation.area_ratio
                      < options.min_board_area_ratio) {
        std::ostringstream reason;
        reason << "board area " << std::fixed << std::setprecision(3)
               << observation.area_ratio * 100.0
               << "% is below "
               << options.min_board_area_ratio * 100.0 << '%';
        observation.rejection_reason = reason.str();
      } else {
        observation.quality_passed = true;
      }
    }

    if (options.preview) {
      cv::Mat preview =
        makePreview(image, observation, options.pattern_size);
      if (preview.cols > 1280 || preview.rows > 900) {
        const double scale = std::min(
          1280.0 / preview.cols, 900.0 / preview.rows);
        cv::resize(
          preview, preview, {}, scale, scale, cv::INTER_AREA);
      }
      cv::imshow(kPreviewWindow, preview);
      const int key = cv::waitKey(20);
      if (key == 27 || key == 'q' || key == 'Q') {
        cv::destroyWindow(kPreviewWindow);
        throw std::runtime_error("calibration preview aborted by user");
      }
    }

    observations.push_back(std::move(observation));
  }

  if (options.preview) {
    cv::destroyWindow(kPreviewWindow);
  }
  if (image_size.empty()) {
    throw std::runtime_error("none of the input images could be decoded");
  }
  return observations;
}

std::vector<std::size_t> selectDiverseViews(
  std::vector<Observation>& observations,
  std::size_t max_views,
  double max_sharpness)
{
  std::vector<std::size_t> candidates;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (observations[index].quality_passed) {
      candidates.push_back(index);
    }
  }

  if (candidates.size() <= max_views) {
    for (const auto index : candidates) {
      observations[index].selected = true;
    }
    return candidates;
  }

  const std::size_t dimensions =
    observations[candidates.front()].descriptor.size();
  std::vector<double> minimum(
    dimensions, std::numeric_limits<double>::infinity());
  std::vector<double> maximum(
    dimensions, -std::numeric_limits<double>::infinity());
  for (const auto index : candidates) {
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
      const double value = observations[index].descriptor[dimension];
      minimum[dimension] = std::min(minimum[dimension], value);
      maximum[dimension] = std::max(maximum[dimension], value);
    }
  }

  std::vector<std::vector<double>> normalized(
    observations.size(), std::vector<double>(dimensions));
  for (const auto index : candidates) {
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
      const double range = maximum[dimension] - minimum[dimension];
      normalized[index][dimension] =
        range > 1e-12
        ? (observations[index].descriptor[dimension] - minimum[dimension])
            / range
        : 0.0;
    }
  }

  const auto seed = *std::max_element(
    candidates.begin(), candidates.end(),
    [&](std::size_t left, std::size_t right) {
      const double left_score =
        observations[left].area_ratio
        / std::max(observations[left].sharpness, 0.1);
      const double right_score =
        observations[right].area_ratio
        / std::max(observations[right].sharpness, 0.1);
      if (left_score == right_score) {
        return observations[left].path.string()
               > observations[right].path.string();
      }
      return left_score < right_score;
    });

  std::vector<std::size_t> selected{seed};
  std::vector<bool> is_selected(observations.size(), false);
  is_selected[seed] = true;
  std::vector<double> minimum_distance(
    observations.size(), std::numeric_limits<double>::infinity());

  while (selected.size() < max_views) {
    const std::size_t newest = selected.back();
    std::size_t best = observations.size();
    double best_score = -1.0;
    for (const auto candidate : candidates) {
      if (is_selected[candidate]) {
        continue;
      }

      double distance_squared = 0.0;
      for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        const double difference =
          normalized[candidate][dimension]
          - normalized[newest][dimension];
        distance_squared += difference * difference;
      }
      minimum_distance[candidate] =
        std::min(minimum_distance[candidate], distance_squared);
      const double quality_weight = std::clamp(
        max_sharpness
          / std::max(observations[candidate].sharpness, 0.1),
        0.75, 1.25);
      const double score =
        minimum_distance[candidate] * quality_weight;
      if (score > best_score
          || (score == best_score && best != observations.size()
              && naturalLess(
                observations[candidate].path,
                observations[best].path))) {
        best = candidate;
        best_score = score;
      }
    }
    if (best == observations.size()) {
      break;
    }
    selected.push_back(best);
    is_selected[best] = true;
  }

  for (const auto index : candidates) {
    if (is_selected[index]) {
      observations[index].selected = true;
    } else {
      observations[index].rejection_reason =
        "not selected by the diverse-view limit";
    }
  }
  std::sort(
    selected.begin(), selected.end(),
    [&](std::size_t left, std::size_t right) {
      return naturalLess(
        observations[left].path, observations[right].path);
    });
  return selected;
}

FitResult fitCalibration(
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& indices,
  const std::vector<cv::Point3f>& object_points,
  cv::Size image_size,
  int flags)
{
  std::vector<std::vector<cv::Point3f>> object_sets(
    indices.size(), object_points);
  std::vector<std::vector<cv::Point2f>> image_sets;
  image_sets.reserve(indices.size());
  for (const auto index : indices) {
    image_sets.push_back(observations[index].corners);
  }

  FitResult result;
  cv::Mat extrinsic_std_deviations;
  cv::Mat per_view_errors;
  result.rms = cv::calibrateCamera(
    object_sets, image_sets, image_size,
    result.camera_matrix, result.distortion_coefficients,
    result.rotation_vectors, result.translation_vectors,
    result.intrinsic_std_deviations,
    extrinsic_std_deviations, per_view_errors,
    flags,
    {cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
     kCalibrationIterations, kCalibrationEpsilon});

  result.camera_matrix.convertTo(result.camera_matrix, CV_64FC1);
  result.distortion_coefficients =
    result.distortion_coefficients.reshape(1, 1);
  result.intrinsic_std_deviations =
    result.intrinsic_std_deviations.reshape(1, 1);
  result.per_view_errors.reserve(per_view_errors.total());
  const cv::Mat flattened = per_view_errors.reshape(1, 1);
  for (int column = 0; column < flattened.cols; ++column) {
    result.per_view_errors.push_back(flattened.at<double>(0, column));
  }
  return result;
}

double median(std::vector<double> values)
{
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[middle - 1] + values[middle]);
  }
  return values[middle];
}

double quantile(std::vector<double> values, double probability)
{
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double position =
    std::clamp(probability, 0.0, 1.0)
    * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - lower;
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double robustOutlierThreshold(const std::vector<double>& errors)
{
  const double center = median(errors);
  std::vector<double> deviations;
  deviations.reserve(errors.size());
  for (const double error : errors) {
    deviations.push_back(std::abs(error - center));
  }
  const double scaled_mad = 1.4826 * median(std::move(deviations));
  return std::max(
    kOutlierAbsoluteFloor,
    center + 3.0 * finiteOrZero(scaled_mad));
}

std::vector<std::size_t> rejectReprojectionOutliers(
  std::vector<Observation>& observations,
  std::vector<std::size_t> active,
  const std::vector<cv::Point3f>& object_points,
  cv::Size image_size,
  std::size_t min_views)
{
  const std::size_t removal_limit =
    static_cast<std::size_t>(std::floor(active.size() * 0.15));
  std::size_t removed = 0;

  while (active.size() > min_views && removed < removal_limit) {
    const FitResult fit = fitCalibration(
      observations, active, object_points, image_size,
      cv::CALIB_FIX_K3);
    const double threshold =
      robustOutlierThreshold(fit.per_view_errors);
    const auto worst = std::max_element(
      fit.per_view_errors.begin(), fit.per_view_errors.end());
    if (worst == fit.per_view_errors.end() || *worst <= threshold) {
      break;
    }

    const std::size_t local_index =
      static_cast<std::size_t>(
        std::distance(fit.per_view_errors.begin(), worst));
    const std::size_t observation_index = active[local_index];
    observations[observation_index].reprojection_error = *worst;
    std::ostringstream reason;
    reason << "reprojection outlier " << std::fixed
           << std::setprecision(4) << *worst
           << " px (threshold " << threshold << " px)";
    observations[observation_index].rejection_reason = reason.str();
    observations[observation_index].selected = false;
    active.erase(active.begin() + static_cast<std::ptrdiff_t>(local_index));
    ++removed;
  }
  return active;
}

double calculateCrossValidationRms(
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& indices,
  const std::vector<cv::Point3f>& object_points,
  cv::Size image_size,
  int flags)
{
  constexpr std::size_t folds = 5;
  double total_squared_error = 0.0;
  std::size_t total_points = 0;

  for (std::size_t fold = 0; fold < folds; ++fold) {
    std::vector<std::size_t> training;
    std::vector<std::size_t> validation;
    for (std::size_t position = 0; position < indices.size(); ++position) {
      ((position % folds) == fold ? validation : training)
        .push_back(indices[position]);
    }
    if (training.size() < 10 || validation.empty()) {
      continue;
    }

    const FitResult fit = fitCalibration(
      observations, training, object_points, image_size, flags);
    for (const auto index : validation) {
      cv::Mat rotation_vector;
      cv::Mat translation_vector;
      if (!cv::solvePnP(
            object_points, observations[index].corners,
            fit.camera_matrix, fit.distortion_coefficients,
            rotation_vector, translation_vector, false,
            cv::SOLVEPNP_ITERATIVE)) {
        continue;
      }
      std::vector<cv::Point2f> projected;
      cv::projectPoints(
        object_points, rotation_vector, translation_vector,
        fit.camera_matrix, fit.distortion_coefficients, projected);
      for (std::size_t point = 0; point < projected.size(); ++point) {
        const cv::Point2f difference =
          projected[point] - observations[index].corners[point];
        total_squared_error +=
          static_cast<double>(difference.dot(difference));
      }
      total_points += projected.size();
    }
  }

  if (total_points == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(total_squared_error / total_points);
}

double matrixValue(const cv::Mat& matrix, std::size_t index)
{
  const cv::Mat flattened = matrix.reshape(1, 1);
  if (index >= flattened.total()) {
    return 0.0;
  }
  return flattened.at<double>(0, static_cast<int>(index));
}

BootstrapResult calculateBootstrap(
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& indices,
  const std::vector<cv::Point3f>& object_points,
  cv::Size image_size,
  int flags,
  const FitResult& initial_fit,
  int iterations,
  const std::function<void(const std::string&)>& progress)
{
  BootstrapResult result;
  result.requested = iterations;
  if (iterations == 0) {
    return result;
  }

  std::mt19937 generator{0x4e565343U};
  std::uniform_int_distribution<std::size_t> distribution{
    0, indices.size() - 1};
  std::vector<std::array<double, 9>> samples;
  samples.reserve(static_cast<std::size_t>(iterations));

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (progress) {
      std::ostringstream message;
      message << "bootstrap " << iteration + 1 << '/' << iterations;
      progress(message.str());
    }
    std::vector<std::vector<cv::Point3f>> object_sets;
    std::vector<std::vector<cv::Point2f>> image_sets;
    object_sets.reserve(indices.size());
    image_sets.reserve(indices.size());
    for (std::size_t sample = 0; sample < indices.size(); ++sample) {
      const auto index = indices[distribution(generator)];
      object_sets.push_back(object_points);
      image_sets.push_back(observations[index].corners);
    }

    cv::Mat camera_matrix = initial_fit.camera_matrix.clone();
    cv::Mat distortion = initial_fit.distortion_coefficients.clone();
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    try {
      cv::calibrateCamera(
        object_sets, image_sets, image_size,
        camera_matrix, distortion, rotations, translations,
        flags | cv::CALIB_USE_INTRINSIC_GUESS,
        {cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
         kCalibrationIterations, kCalibrationEpsilon});
    } catch (const cv::Exception&) {
      continue;
    }
    if (!cv::checkRange(camera_matrix)
        || !cv::checkRange(distortion)) {
      continue;
    }

    samples.push_back({
      camera_matrix.at<double>(0, 0),
      camera_matrix.at<double>(1, 1),
      camera_matrix.at<double>(0, 2),
      camera_matrix.at<double>(1, 2),
      matrixValue(distortion, 0),
      matrixValue(distortion, 1),
      matrixValue(distortion, 2),
      matrixValue(distortion, 3),
      matrixValue(distortion, 4)});
  }

  result.succeeded = static_cast<int>(samples.size());
  if (samples.size() < 2) {
    return result;
  }
  result.parameter_std_deviations.resize(9);
  for (std::size_t parameter = 0; parameter < 9; ++parameter) {
    double mean = 0.0;
    for (const auto& sample : samples) {
      mean += sample[parameter];
    }
    mean /= static_cast<double>(samples.size());
    double sum_squared = 0.0;
    for (const auto& sample : samples) {
      const double difference = sample[parameter] - mean;
      sum_squared += difference * difference;
    }
    result.parameter_std_deviations[parameter] =
      std::sqrt(sum_squared / static_cast<double>(samples.size() - 1));
  }
  return result;
}

std::string timestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm local_time{};
#if defined(_WIN32)
  if (localtime_s(&local_time, &now) != 0) {
    throw std::runtime_error("failed to convert output timestamp");
  }
#else
  if (localtime_r(&now, &local_time) == nullptr) {
    throw std::runtime_error("failed to convert output timestamp");
  }
#endif
  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");
  return stream.str();
}

std::filesystem::path createUniqueOutputDirectory(
  const std::filesystem::path& output_root)
{
  std::error_code error;
  std::filesystem::create_directories(output_root, error);
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to create calibration output root", output_root, error);
  }
  if (!std::filesystem::is_directory(output_root, error)) {
    throw std::runtime_error(
      "calibration output root is not a directory: "
      + output_root.string());
  }

  const std::string base = timestamp();
  for (std::uint64_t suffix = 0;; ++suffix) {
    std::string name = base;
    if (suffix > 0) {
      name += "_" + std::to_string(suffix);
    }
    const auto candidate = output_root / name;
    error.clear();
    if (std::filesystem::create_directory(candidate, error)) {
      return candidate;
    }
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to create calibration result directory",
        candidate, error);
    }
  }
}

void writeTextFileAtomic(
  const std::filesystem::path& path,
  const std::string& contents)
{
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream output{
      temporary, std::ios::out | std::ios::trunc};
    if (!output) {
      throw std::runtime_error(
        "failed to open output file: " + temporary);
    }
    output << contents;
    if (!output) {
      throw std::runtime_error(
        "failed to write output file: " + temporary);
    }
  }

  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to commit output file", path, error);
  }
}

std::string emitCalibrationYaml(
  cv::Size image_size,
  const FitResult& fit)
{
  YAML::Emitter output;
  output.SetDoublePrecision(17);
  output << YAML::BeginMap;
  output << YAML::Key << "calibration" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "image_width" << YAML::Value << image_size.width;
  output << YAML::Key << "image_height" << YAML::Value << image_size.height;
  output << YAML::Key << "camera_matrix" << YAML::Value << YAML::BeginSeq;
  for (int row = 0; row < 3; ++row) {
    output << YAML::Flow << YAML::BeginSeq;
    for (int column = 0; column < 3; ++column) {
      output << fit.camera_matrix.at<double>(row, column);
    }
    output << YAML::EndSeq;
  }
  output << YAML::EndSeq;
  output << YAML::Key << "distortion_coefficients"
         << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (std::size_t index = 0; index < 5; ++index) {
    output << matrixValue(fit.distortion_coefficients, index);
  }
  output << YAML::EndSeq;
  output << YAML::EndMap;
  output << YAML::EndMap;
  return std::string{output.c_str()} + '\n';
}

std::vector<std::string> evaluateQuality(
  const FitResult& fit,
  cv::Size image_size,
  double cross_validation_rms,
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& active,
  double per_view_p95,
  std::vector<std::string>& warnings)
{
  std::vector<std::string> failures;
  if (!std::isfinite(fit.rms) || fit.rms > kMaxFinalRms) {
    std::ostringstream message;
    message << "overall RMS " << fit.rms
            << " px exceeds " << kMaxFinalRms << " px";
    failures.push_back(message.str());
  }
  if (!std::isfinite(per_view_p95)
      || per_view_p95 > kMaxPerViewP95) {
    std::ostringstream message;
    message << "per-view p95 " << per_view_p95
            << " px exceeds " << kMaxPerViewP95 << " px";
    failures.push_back(message.str());
  }
  if (!std::isfinite(cross_validation_rms)
      || cross_validation_rms > kMaxCrossValidationRms) {
    std::ostringstream message;
    message << "cross-validation RMS " << cross_validation_rms
            << " px exceeds " << kMaxCrossValidationRms << " px";
    failures.push_back(message.str());
  }

  const double fx = fit.camera_matrix.at<double>(0, 0);
  const double fy = fit.camera_matrix.at<double>(1, 1);
  const double cx = fit.camera_matrix.at<double>(0, 2);
  const double cy = fit.camera_matrix.at<double>(1, 2);
  if (!cv::checkRange(fit.camera_matrix)
      || !cv::checkRange(fit.distortion_coefficients)
      || fx <= 0.0 || fy <= 0.0) {
    failures.emplace_back(
      "camera matrix or distortion contains invalid values");
  }
  if (cx < 0.0 || cx >= image_size.width
      || cy < 0.0 || cy >= image_size.height) {
    failures.emplace_back("principal point is outside the image");
  }

  const double fx_std = matrixValue(fit.intrinsic_std_deviations, 0);
  const double fy_std = matrixValue(fit.intrinsic_std_deviations, 1);
  const double cx_std = matrixValue(fit.intrinsic_std_deviations, 2);
  const double cy_std = matrixValue(fit.intrinsic_std_deviations, 3);
  if (!std::isfinite(fx_std) || !std::isfinite(fy_std)
      || fx_std / fx > kMaxFocalRelativeStdDev
      || fy_std / fy > kMaxFocalRelativeStdDev) {
    failures.emplace_back(
      "focal-length uncertainty exceeds 0.5 percent");
  }
  if (!std::isfinite(cx_std) || !std::isfinite(cy_std)
      || cx_std > kMaxPrincipalPointStdDev
      || cy_std > kMaxPrincipalPointStdDev) {
    failures.emplace_back(
      "principal-point uncertainty exceeds 2 pixels");
  }

  double min_x = 1.0;
  double max_x = 0.0;
  double min_y = 1.0;
  double max_y = 0.0;
  double min_area = std::numeric_limits<double>::infinity();
  double max_area = 0.0;
  for (const auto index : active) {
    const auto& observation = observations[index];
    min_x = std::min(min_x, observation.center_normalized.x);
    max_x = std::max(max_x, observation.center_normalized.x);
    min_y = std::min(min_y, observation.center_normalized.y);
    max_y = std::max(max_y, observation.center_normalized.y);
    min_area = std::min(min_area, observation.area_ratio);
    max_area = std::max(max_area, observation.area_ratio);
  }
  if (max_x - min_x < 0.5 || max_y - min_y < 0.5) {
    warnings.emplace_back(
      "checkerboard center coverage spans less than half the image");
  }
  if (!std::isfinite(min_area)
      || max_area / std::max(min_area, 1e-12) < 3.0) {
    warnings.emplace_back(
      "checkerboard scale range is below 3x; add near and far views");
  }
  return failures;
}

std::string emitReportYaml(
  const CameraCalibratorOptions& options,
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& active,
  const FitResult& fit,
  const BootstrapResult& bootstrap,
  const std::string& model,
  double four_parameter_cv,
  double five_parameter_cv,
  double selected_cv,
  std::size_t initially_selected,
  const std::vector<std::string>& warnings,
  const std::vector<std::string>& failures)
{
  const double mean_error = std::accumulate(
    fit.per_view_errors.begin(), fit.per_view_errors.end(), 0.0)
    / std::max<std::size_t>(fit.per_view_errors.size(), 1);
  YAML::Emitter output;
  output.SetDoublePrecision(15);
  output << YAML::BeginMap;
  output << YAML::Key << "input_directory"
         << YAML::Value << options.input_directory.string();
  output << YAML::Key << "pattern" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "type" << YAML::Value << "checkerboard";
  output << YAML::Key << "inner_corners"
         << YAML::Value << YAML::Flow << YAML::BeginSeq
         << options.pattern_size.width << options.pattern_size.height
         << YAML::EndSeq;
  output << YAML::Key << "squares"
         << YAML::Value << YAML::Flow << YAML::BeginSeq
         << options.pattern_size.width + 1
         << options.pattern_size.height + 1 << YAML::EndSeq;
  output << YAML::Key << "square_size"
         << YAML::Value << options.square_size;
  output << YAML::EndMap;

  const auto detected = std::count_if(
    observations.begin(), observations.end(),
    [](const Observation& observation) {
      return observation.detected;
    });
  const auto quality = std::count_if(
    observations.begin(), observations.end(),
    [](const Observation& observation) {
      return observation.quality_passed;
    });
  const auto selected = std::count_if(
    observations.begin(), observations.end(),
    [](const Observation& observation) {
      return observation.selected || observation.used;
    });
  output << YAML::Key << "counts" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "discovered"
         << YAML::Value << observations.size();
  output << YAML::Key << "detected" << YAML::Value << detected;
  output << YAML::Key << "quality_passed" << YAML::Value << quality;
  output << YAML::Key << "selected" << YAML::Value << selected;
  output << YAML::Key << "diversity_selected"
         << YAML::Value << initially_selected;
  output << YAML::Key << "reprojection_rejected"
         << YAML::Value << initially_selected - active.size();
  output << YAML::Key << "used" << YAML::Value << active.size();
  output << YAML::EndMap;

  output << YAML::Key << "model" << YAML::Value << model;
  output << YAML::Key << "errors_px" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "rms" << YAML::Value << fit.rms;
  output << YAML::Key << "mean_per_view"
         << YAML::Value << mean_error;
  output << YAML::Key << "median_per_view"
         << YAML::Value << median(fit.per_view_errors);
  output << YAML::Key << "p90_per_view"
         << YAML::Value << quantile(fit.per_view_errors, 0.90);
  output << YAML::Key << "p95_per_view"
         << YAML::Value << quantile(fit.per_view_errors, 0.95);
  output << YAML::Key << "max_per_view"
         << YAML::Value
         << *std::max_element(
              fit.per_view_errors.begin(), fit.per_view_errors.end());
  output << YAML::Key << "cross_validation_selected"
         << YAML::Value << selected_cv;
  output << YAML::Key << "cross_validation_fixed_k3"
         << YAML::Value << four_parameter_cv;
  output << YAML::Key << "cross_validation_free_k3"
         << YAML::Value << five_parameter_cv;
  output << YAML::EndMap;

  output << YAML::Key << "analytic_intrinsic_std_deviations"
         << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (std::size_t index = 0;
       index < fit.intrinsic_std_deviations.total(); ++index) {
    output << matrixValue(fit.intrinsic_std_deviations, index);
  }
  output << YAML::EndSeq;

  output << YAML::Key << "bootstrap" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "requested"
         << YAML::Value << bootstrap.requested;
  output << YAML::Key << "succeeded"
         << YAML::Value << bootstrap.succeeded;
  output << YAML::Key << "parameter_order"
         << YAML::Value << YAML::Flow << YAML::BeginSeq
         << "fx" << "fy" << "cx" << "cy"
         << "k1" << "k2" << "p1" << "p2" << "k3"
         << YAML::EndSeq;
  output << YAML::Key << "std_deviations"
         << YAML::Value << YAML::Flow << YAML::BeginSeq;
  for (const double deviation : bootstrap.parameter_std_deviations) {
    output << deviation;
  }
  output << YAML::EndSeq;
  output << YAML::EndMap;

  output << YAML::Key << "quality_passed"
         << YAML::Value << failures.empty();
  output << YAML::Key << "warnings"
         << YAML::Value << YAML::BeginSeq;
  for (const auto& warning : warnings) {
    output << warning;
  }
  output << YAML::EndSeq;
  output << YAML::Key << "failures"
         << YAML::Value << YAML::BeginSeq;
  for (const auto& failure : failures) {
    output << failure;
  }
  output << YAML::EndSeq;

  output << YAML::Key << "images" << YAML::Value << YAML::BeginSeq;
  for (const auto& observation : observations) {
    output << YAML::BeginMap;
    output << YAML::Key << "file"
           << YAML::Value << observation.path.filename().string();
    output << YAML::Key << "detected"
           << YAML::Value << observation.detected;
    output << YAML::Key << "quality_passed"
           << YAML::Value << observation.quality_passed;
    output << YAML::Key << "used"
           << YAML::Value << observation.used;
    if (observation.detected) {
      output << YAML::Key << "sharpness"
             << YAML::Value << observation.sharpness;
      output << YAML::Key << "contrast"
             << YAML::Value << observation.contrast;
      output << YAML::Key << "board_area_ratio"
             << YAML::Value << observation.area_ratio;
    }
    if (std::isfinite(observation.reprojection_error)) {
      output << YAML::Key << "reprojection_error_px"
             << YAML::Value << observation.reprojection_error;
    }
    if (!observation.rejection_reason.empty()) {
      output << YAML::Key << "reason"
             << YAML::Value << observation.rejection_reason;
    }
    output << YAML::EndMap;
  }
  output << YAML::EndSeq;
  output << YAML::EndMap;
  return std::string{output.c_str()} + '\n';
}

void writeUsedImages(
  const std::filesystem::path& output_directory,
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& active)
{
  std::ostringstream contents;
  for (const auto index : active) {
    contents << observations[index].path.string() << '\n';
  }
  writeTextFileAtomic(
    output_directory / "used_images.txt", contents.str());
}

void writeCoverageImage(
  const std::filesystem::path& output_directory,
  cv::Size source_size,
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& active)
{
  constexpr int canvas_width = 1000;
  constexpr int margin = 60;
  const double scale =
    static_cast<double>(canvas_width - 2 * margin) / source_size.width;
  const int canvas_height =
    static_cast<int>(std::lround(
      source_size.height * scale)) + 2 * margin;
  cv::Mat canvas(
    canvas_height, canvas_width, CV_8UC3, cv::Scalar{25, 25, 25});
  cv::rectangle(
    canvas,
    {margin, margin},
    {canvas_width - margin, canvas_height - margin},
    cv::Scalar{180, 180, 180}, 2);

  for (const auto index : active) {
    const auto& observation = observations[index];
    const cv::Point center{
      margin + static_cast<int>(
        std::lround(
          observation.center_normalized.x
          * (canvas_width - 2 * margin))),
      margin + static_cast<int>(
        std::lround(
          observation.center_normalized.y
          * (canvas_height - 2 * margin)))};
    const int radius = std::clamp(
      static_cast<int>(
        std::lround(std::sqrt(observation.area_ratio) * 45.0)),
      3, 14);
    cv::circle(canvas, center, radius, cv::Scalar{40, 220, 40}, 2);
  }
  cv::putText(
    canvas, "Selected checkerboard centers and relative scales",
    {margin, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.75,
    cv::Scalar{230, 230, 230}, 2, cv::LINE_AA);
  if (!cv::imwrite(
        (output_directory / "coverage.png").string(), canvas)) {
    throw std::runtime_error("failed to write coverage diagnostic");
  }
}

void writeErrorChart(
  const std::filesystem::path& output_directory,
  const FitResult& fit)
{
  constexpr int width = 1100;
  constexpr int height = 520;
  constexpr int margin = 60;
  cv::Mat chart(height, width, CV_8UC3, cv::Scalar{25, 25, 25});
  const double maximum = std::max(
    kMaxPerViewP95,
    *std::max_element(
      fit.per_view_errors.begin(), fit.per_view_errors.end())
      * 1.1);
  const int plot_width = width - 2 * margin;
  const int plot_height = height - 2 * margin;
  const double bar_width =
    static_cast<double>(plot_width) / fit.per_view_errors.size();

  for (std::size_t index = 0;
       index < fit.per_view_errors.size(); ++index) {
    const int x0 = margin
      + static_cast<int>(std::floor(index * bar_width));
    const int x1 = margin
      + static_cast<int>(std::ceil((index + 1) * bar_width));
    const int y = height - margin
      - static_cast<int>(
        std::lround(fit.per_view_errors[index] / maximum * plot_height));
    cv::rectangle(
      chart, {x0, y}, {std::max(x0 + 1, x1), height - margin},
      cv::Scalar{60, 190, 80}, cv::FILLED);
  }
  const int p95_line = height - margin
    - static_cast<int>(
      std::lround(kMaxPerViewP95 / maximum * plot_height));
  cv::line(
    chart, {margin, p95_line}, {width - margin, p95_line},
    cv::Scalar{60, 60, 230}, 2);
  cv::rectangle(
    chart, {margin, margin}, {width - margin, height - margin},
    cv::Scalar{180, 180, 180}, 2);
  cv::putText(
    chart, "Per-view reprojection RMS (px)",
    {margin, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.75,
    cv::Scalar{230, 230, 230}, 2, cv::LINE_AA);
  if (!cv::imwrite(
        (output_directory / "reprojection_errors.png").string(),
        chart)) {
    throw std::runtime_error(
      "failed to write reprojection-error diagnostic");
  }
}

void writeUndistortedPreviews(
  const std::filesystem::path& output_directory,
  const std::vector<Observation>& observations,
  const std::vector<std::size_t>& active,
  const FitResult& fit)
{
  const auto preview_directory =
    output_directory / "undistorted_preview";
  std::error_code error;
  std::filesystem::create_directory(preview_directory, error);
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to create undistorted preview directory",
      preview_directory, error);
  }

  const std::size_t count = std::min<std::size_t>(6, active.size());
  for (std::size_t preview_index = 0;
       preview_index < count; ++preview_index) {
    const std::size_t position =
      count == 1
      ? 0
      : preview_index * (active.size() - 1) / (count - 1);
    const auto& observation = observations[active[position]];
    const cv::Mat original =
      cv::imread(observation.path.string(), cv::IMREAD_COLOR);
    if (original.empty()) {
      continue;
    }
    cv::Mat undistorted;
    cv::undistort(
      original, undistorted,
      fit.camera_matrix, fit.distortion_coefficients);

    cv::Mat left;
    cv::Mat right;
    const double scale = std::min(1.0, 700.0 / original.cols);
    cv::resize(original, left, {}, scale, scale, cv::INTER_AREA);
    cv::resize(undistorted, right, {}, scale, scale, cv::INTER_AREA);
    cv::putText(
      left, "original", {15, 30},
      cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
    cv::putText(
      right, "undistorted", {15, 30},
      cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
    cv::Mat combined;
    cv::hconcat(left, right, combined);
    if (!cv::imwrite(
          (preview_directory / observation.path.filename()).string(),
          combined)) {
      throw std::runtime_error(
        "failed to write undistorted preview image");
    }
  }
}

}  // namespace

CameraCalibratorResult calibrateCameraImages(
  const CameraCalibratorOptions& options)
{
  validateOptions(options);
  const auto image_paths = discoverImages(options.input_directory);
  if (options.progress) {
    std::ostringstream message;
    message << "found " << image_paths.size()
            << " image files; detecting "
            << options.pattern_size.width << 'x'
            << options.pattern_size.height << " inner corners";
    options.progress(message.str());
  }

  cv::Size image_size;
  std::vector<Observation> observations =
    detectObservations(image_paths, options, image_size);
  const std::size_t detected_count = static_cast<std::size_t>(
    std::count_if(
      observations.begin(), observations.end(),
      [](const Observation& observation) {
        return observation.detected;
      }));
  const std::size_t quality_count = static_cast<std::size_t>(
    std::count_if(
      observations.begin(), observations.end(),
      [](const Observation& observation) {
        return observation.quality_passed;
      }));
  if (options.progress) {
    std::ostringstream message;
    message << "detected " << detected_count << '/'
            << observations.size() << ", quality passed "
            << quality_count << '/' << observations.size();
    options.progress(message.str());
  }
  if (quality_count < options.min_views) {
    std::ostringstream message;
    message << "only " << quality_count
            << " images passed checkerboard quality checks; at least "
            << options.min_views << " are required";
    throw std::runtime_error(message.str());
  }

  std::vector<std::size_t> selected =
    selectDiverseViews(
      observations, options.max_views, options.max_sharpness);
  const std::size_t initially_selected = selected.size();
  if (options.progress) {
    options.progress(
      "selected " + std::to_string(initially_selected)
      + " diverse views; rejecting reprojection outliers");
  }
  const auto object_points =
    makeObjectPoints(options.pattern_size, options.square_size);
  selected = rejectReprojectionOutliers(
    observations, std::move(selected), object_points,
    image_size, options.min_views);
  if (selected.size() < options.min_views) {
    throw std::runtime_error(
      "reprojection outlier rejection left too few calibration views");
  }
  if (options.progress) {
    options.progress(
      "using " + std::to_string(selected.size())
      + " views after robust rejection; fitting distortion models");
  }

  const FitResult fixed_k3 = fitCalibration(
    observations, selected, object_points, image_size,
    cv::CALIB_FIX_K3);
  const FitResult free_k3 = fitCalibration(
    observations, selected, object_points, image_size, 0);
  if (options.progress) {
    options.progress(
      "running five-fold cross-validation for fixed/free k3");
  }
  const double fixed_k3_cv = calculateCrossValidationRms(
    observations, selected, object_points, image_size,
    cv::CALIB_FIX_K3);
  const double free_k3_cv = calculateCrossValidationRms(
    observations, selected, object_points, image_size, 0);
  const double k3 = matrixValue(free_k3.distortion_coefficients, 4);
  const double k3_std =
    matrixValue(free_k3.intrinsic_std_deviations, 8);
  const bool k3_stable =
    std::isfinite(k3_std) && k3_std > 0.0
    && std::abs(k3) > 3.0 * k3_std;
  const bool free_k3_improves =
    std::isfinite(fixed_k3_cv) && std::isfinite(free_k3_cv)
    && free_k3_cv
       <= fixed_k3_cv * (1.0 - kModelImprovementRatio);
  const bool use_free_k3 = k3_stable && free_k3_improves;
  FitResult final_fit = use_free_k3 ? free_k3 : fixed_k3;
  const int final_flags =
    use_free_k3 ? 0 : cv::CALIB_FIX_K3;
  const std::string model =
    use_free_k3 ? "brown_conrady_5" : "brown_conrady_4_fixed_k3";
  const double selected_cv =
    use_free_k3 ? free_k3_cv : fixed_k3_cv;
  if (options.progress) {
    std::ostringstream message;
    message << "selected " << model << "; RMS "
            << final_fit.rms << " px, cross-validation "
            << selected_cv << " px";
    options.progress(message.str());
  }

  for (std::size_t local = 0; local < selected.size(); ++local) {
    auto& observation = observations[selected[local]];
    observation.selected = true;
    observation.used = true;
    observation.reprojection_error =
      final_fit.per_view_errors[local];
  }

  const BootstrapResult bootstrap = calculateBootstrap(
    observations, selected, object_points, image_size,
    final_flags, final_fit, options.bootstrap_iterations,
    options.progress);
  std::vector<std::string> warnings;
  if (!use_free_k3) {
    warnings.emplace_back(
      "k3 was fixed to zero because the five-parameter model did not "
      "improve cross-validation by at least 2 percent with stable k3");
  }
  if (options.bootstrap_iterations > 0
      && bootstrap.succeeded
           < std::max(2, options.bootstrap_iterations * 9 / 10)) {
    warnings.emplace_back(
      "fewer than 90 percent of bootstrap calibrations succeeded");
  }
  const double p95 = quantile(final_fit.per_view_errors, 0.95);
  const std::vector<std::string> failures = evaluateQuality(
    final_fit, image_size, selected_cv,
    observations, selected, p95, warnings);

  const auto output_directory =
    createUniqueOutputDirectory(options.output_root);
  if (options.progress) {
    options.progress(
      "writing calibration and diagnostics to "
      + output_directory.string());
  }
  writeTextFileAtomic(
    output_directory / "calibration.yaml",
    emitCalibrationYaml(image_size, final_fit));
  writeTextFileAtomic(
    output_directory / "report.yaml",
    emitReportYaml(
      options, observations, selected, final_fit, bootstrap,
      model, fixed_k3_cv, free_k3_cv, selected_cv,
      initially_selected,
      warnings, failures));
  writeUsedImages(output_directory, observations, selected);
  writeCoverageImage(
    output_directory, image_size, observations, selected);
  writeErrorChart(output_directory, final_fit);
  writeUndistortedPreviews(
    output_directory, observations, selected, final_fit);

  CameraCalibratorResult result;
  result.output_directory = output_directory;
  result.image_size = image_size;
  result.camera_matrix = final_fit.camera_matrix.clone();
  result.distortion_coefficients =
    final_fit.distortion_coefficients.clone();
  result.distortion_model = model;
  result.discovered_images = observations.size();
  result.detected_images = detected_count;
  result.quality_images = quality_count;
  result.selected_images = initially_selected;
  result.used_images = selected.size();
  result.rms = final_fit.rms;
  result.cross_validation_rms = selected_cv;
  result.per_view_p95 = p95;
  result.quality_passed = failures.empty();
  result.warnings = std::move(warnings);
  return result;
}

}  // namespace tools
