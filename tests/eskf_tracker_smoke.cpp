// 误差状态跟踪器的状态机与双缓冲检查。
//
// 用合成检测驱动完整链路：L2 角点 → 初始化 PnP → 关联 → 端点观测更新。验证
// Lost → Detecting → Tracking 的推进、丢帧后进 TempLost 再恢复、以及超时放弃。
//
// 与 eskf_target_smoke 的分工：那边验证估计本身对不对，这边验证生命周期管理。

#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/light_measure.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace VM = L3Estimation::VehicleModel;

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

constexpr auto kName = L3Estimation::ArmorName::Infantry3;
constexpr int kArmorNum = 4;
constexpr double kDt = 0.005;
using State = Eigen::Matrix<double, VM::kStateSize, 1>;

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = cv::Size(1440, 1080);
  calibration.camera_matrix =
    (cv::Mat_<double>(3, 3) << 1210.0, 0.0, 721.5, 0.0, 1208.0, 539.5, 0.0, 0.0, 1.0);
  calibration.distortion_coefficients =
    (cv::Mat_<double>(1, 5) << -0.12, 0.03, 0.0004, -0.0002, 0.0);
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = Eigen::Matrix3d{{0.0, 0.0, 1.0}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}};
  transform.translation() = Eigen::Vector3d{0.02, -0.01, 0.06};
  calibration.T_barrel_camera = transform;
  return calibration;
}

State makeTruth()
{
  State x = State::Zero();
  x[VM::idx::CX] = 3.0;
  x[VM::idx::CY] = 0.30;
  x[VM::idx::CZ] = 0.05;
  x[VM::idx::VCX] = 0.25;
  x[VM::idx::ROT_Z] = 0.20;
  x[VM::idx::VYAW] = 1.5;
  x[VM::idx::LOG_R1] = std::log(0.265);
  x[VM::idx::LOG_R2] = std::log(0.28);
  x[VM::idx::HEIGHT] = 0.035;
  return x;
}

// 枪管姿态取单位四元数：世界系与枪管系重合，相机位姿完全由 T_barrel_camera 决定。
const Eigen::Quaterniond kBarrelPose = Eigen::Quaterniond::Identity();

// 由真值状态合成一帧 L2 检测。
std::vector<L2Perception::Armor> synthesizeFrame(
  const State & x, const L1Sensor::CameraCalibration & calibration,
  const L3Estimation::ArmorConfig & armor_config)
{
  const Eigen::Isometry3d camera =
    L3Estimation::cameraInWorld(calibration, kBarrelPose);

  std::vector<L2Perception::Armor> frame;
  for (int id = 0; id < kArmorNum; ++id) {
    const auto pose_in_world = VM::armorPose<double>(x.data(), id, kArmorNum, kName);
    const Eigen::Isometry3d pose_in_camera = camera.inverse() * pose_in_world;
    // 只合成朝向相机的板。
    if ((-pose_in_camera.linear().col(0)).dot(-pose_in_camera.translation()) <= 0.0) {
      continue;
    }

    L3Estimation::LightContext ctx;
    ctx.armor_num = kArmorNum;
    ctx.id = id;
    ctx.name = kName;
    ctx.armor_config = armor_config;
    ctx.camera_in_world = camera;
    ctx.camera_matrix = calibration.camera_matrix;
    ctx.distortion_coefficients = calibration.distortion_coefficients;

    ctx.is_left = true;
    const auto left = L3Estimation::LightMeasure{ctx}.projectedPoints(x);
    ctx.is_left = false;
    const auto right = L3Estimation::LightMeasure{ctx}.projectedPoints(x);

    L2Perception::Armor armor;
    armor.corners = {left.first, right.first, right.second, left.second};
    armor.center =
      (armor.corners[0] + armor.corners[1] + armor.corners[2] + armor.corners[3]) / 4.0F;
    armor.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry3);
    armor.color = L2Perception::ArmorColor::Blue;
    armor.confidence = 0.9F;
    frame.push_back(armor);
  }
  return frame;
}

// 按真值状态投影出某块板某根灯条，用来造"没配成完整板"的独立灯条。
L2Perception::Light lightOfPlate(
  const State& x, int id, bool is_left,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::ArmorConfig& armor_config)
{
  L3Estimation::LightContext ctx;
  ctx.armor_num = kArmorNum;
  ctx.id = id;
  ctx.is_left = is_left;
  ctx.name = kName;
  ctx.armor_config = armor_config;
  ctx.camera_in_world =
    L3Estimation::cameraInWorld(calibration, kBarrelPose);
  ctx.camera_matrix = calibration.camera_matrix;
  ctx.distortion_coefficients = calibration.distortion_coefficients;

  const auto points = L3Estimation::LightMeasure{ctx}.projectedPoints(x);
  L2Perception::Light light;
  light.top = points.first;
  light.bottom = points.second;
  light.center = (light.top + light.bottom) * 0.5F;
  light.length = cv::norm(light.top - light.bottom);
  light.color = L2Perception::ArmorColor::Blue;
  return light;
}

// 正对相机、会被 synthesizeFrame 合成出来的板号。
std::vector<int> facingIds(const State& x, const L1Sensor::CameraCalibration& calibration)
{
  const Eigen::Isometry3d camera =
    L3Estimation::cameraInWorld(calibration, kBarrelPose);
  std::vector<int> ids;
  for (int id = 0; id < kArmorNum; ++id) {
    const auto pose_in_world = VM::armorPose<double>(x.data(), id, kArmorNum, kName);
    const Eigen::Isometry3d pose_in_camera = camera.inverse() * pose_in_world;
    if ((-pose_in_camera.linear().col(0)).dot(-pose_in_camera.translation()) > 0.0) {
      ids.push_back(id);
    }
  }
  return ids;
}

L2Perception::Light leftLightOf(const L2Perception::Armor& armor)
{
  L2Perception::Light light;
  light.top = armor.corners[0];
  light.bottom = armor.corners[3];
  light.center = (light.top + light.bottom) * 0.5F;
  light.length = cv::norm(light.top - light.bottom);
  light.color = armor.color;
  return light;
}

}  // namespace

int main()
{
  const L1Sensor::CameraCalibration calibration = makeCalibration();

  L3Estimation::EskfTrackerConfig tracker_config;
  tracker_config.tracking_thres = 5;
  tracker_config.lost_time_thres = 0.1;  // 20 帧 @ 5ms

  L3Estimation::EskfTargetConfig target_config;
  const L3Estimation::ArmorConfig armor_config;

  // --- 1. 标定无效时 ready() 为假 ------------------------------------
  {
    L1Sensor::CameraCalibration broken;
    broken.image_size = cv::Size(1440, 1080);
    L3Estimation::EskfTracker tracker(broken, armor_config, tracker_config, target_config);
    expect(!tracker.ready(), "标定缺失时 ready() 必须为假");

    const auto result = tracker.track({}, kBarrelPose, L3Estimation::TimePoint{});
    expect(!result.has_value(), "未就绪时 track 必须返回空");
  }

  // --- 2. Lost → Detecting → Tracking --------------------------------
  {
    L3Estimation::EskfTracker tracker(
      calibration, armor_config, tracker_config, target_config);
    expect(tracker.ready(), "有效标定下 ready() 应为真");
    expect(tracker.state() == L3Estimation::TrackState::Lost, "初始状态应为 Lost");

    State truth = makeTruth();
    const VM::Motion motion{.dt = kDt, .name = kName};
    auto now = L3Estimation::TimePoint{} + std::chrono::seconds(1);

    bool saw_detecting = false;
    int frames_to_tracking = -1;

    for (int step = 0; step < 40; ++step) {
      State next;
      motion(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));

      const auto frame = synthesizeFrame(truth, calibration, armor_config);
      const auto result = tracker.track(frame, kBarrelPose, now);

      if (tracker.state() == L3Estimation::TrackState::Detecting) {
        saw_detecting = true;
      }
      if (tracker.state() == L3Estimation::TrackState::Tracking && frames_to_tracking < 0) {
        frames_to_tracking = step;
      }
    }

    expect(saw_detecting, "应当经过 Detecting 状态");
    expect(
      tracker.state() == L3Estimation::TrackState::Tracking, "40 帧后应当进入 Tracking");
    expect(
      frames_to_tracking >= tracker_config.tracking_thres,
      "转入 Tracking 早于 tracking_thres 帧");

    const auto light_roi = tracker.lightRoi(
      kBarrelPose, now, calibration.image_size);
    expect(light_roi.has_value(), "Tracking 时应当启用独立灯条检测 ROI");
    expect(
      light_roi && !light_roi->empty() &&
        (*light_roi & cv::Rect(0, 0, calibration.image_size.width,
                              calibration.image_size.height)) == *light_roi,
      "独立灯条检测 ROI 必须非空且位于图像内");

    const auto poses = tracker.armorPoses();
    expect(
      poses.size() == static_cast<std::size_t>(kArmorNum), "Tracking 时板位姿列表应当非空");

    const auto& armor_update_lights = tracker.usedLights();
    expect(!armor_update_lights.empty(), "完整板更新应当发布实际使用的灯条");
    expect(
      armor_update_lights.size() ==
        static_cast<std::size_t>(tracker.lastMatchCount() * 2),
      "每块完整板应当对应两根更新灯条");
    expect(
      std::none_of(
        armor_update_lights.begin(), armor_update_lights.end(),
        [](const L3Estimation::UsedLight& light) {
          return light.isolated;
        }),
      "未传入独立灯条时不应发布 isolated 灯条");

    // 再传独立灯条。已经配成完整板的那根不能再算一次，而没配成板的邻板灯条
    // 应当通过 matchLight、以 isolated=true 出现在显示清单里。
    State next;
    motion(truth.data(), next.data());
    truth = next;
    now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
      std::chrono::duration<double>(kDt));
    const auto frame_with_light = synthesizeFrame(truth, calibration, armor_config);
    const auto facing = facingIds(truth, calibration);
    expect(!frame_with_light.empty(), "独立灯条测试帧不应为空");
    if (!frame_with_light.empty() && !facing.empty()) {
      // ① 完整板自己的灯条：已经作为板的角点进过观测，不应再作为独立灯条。
      const std::vector<L2Perception::Light> consumed_light{
        leftLightOf(frame_with_light.front())};
      tracker.track(frame_with_light, consumed_light, kBarrelPose, now);
      {
        const auto& used = tracker.usedLights();
        expect(
          std::none_of(
            used.begin(), used.end(),
            [](const L3Estimation::UsedLight& light) { return light.isolated; }),
          "完整板自己的灯条不应再作为独立灯条重复进更新");
        expect(
          used.size() == static_cast<std::size_t>(tracker.lastMatchCount() * 2),
          "只有完整板时灯条清单应当恰好是每板两根");
      }

      // ② 侧面那块板没被检出（整板网络常见的漏检）：它靠近正对板的那根灯条应当
      //    以 isolated=true 参与更新。按真值挑出最正对的板，把另一块朝向相机的
      //    板从帧里拿掉，再把那根灯条单独喂进去。
      motion(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));
      const auto side_facing = facingIds(truth, calibration);
      expect(side_facing.size() == 2, "侧边灯条测试帧应当有两块板朝向相机");
      if (side_facing.size() == 2) {
        const Eigen::Isometry3d camera =
          L3Estimation::cameraInWorld(calibration, kBarrelPose);
        const auto facing = [&](int id) {
          const auto pose = VM::armorPose<double>(truth.data(), id, kArmorNum, kName);
          const Eigen::Isometry3d in_camera = camera.inverse() * pose;
          return (-in_camera.linear().col(0)).dot(-in_camera.translation());
        };
        const bool first_in_front = facing(side_facing[0]) >= facing(side_facing[1]);
        const int front = first_in_front ? side_facing[0] : side_facing[1];
        const int side = first_in_front ? side_facing[1] : side_facing[0];
        // 右邻板靠近正对板的是它的左灯条，左邻板的是右灯条。
        const bool near_is_left = side == (front + 1) % kArmorNum;

        auto side_missed = synthesizeFrame(truth, calibration, armor_config);
        const L2Perception::Light side_top_left =
          lightOfPlate(truth, side, true, calibration, armor_config);
        std::erase_if(side_missed, [&](const L2Perception::Armor& armor) {
          return cv::norm(armor.corners[0] - side_top_left.top) < 1e-3;
        });
        expect(side_missed.size() == 1, "拿掉侧面板后帧里应当只剩正对的板");

        const std::vector<L2Perception::Light> side_lights{
          lightOfPlate(truth, side, near_is_left, calibration, armor_config)};
        tracker.track(side_missed, side_lights, kBarrelPose, now);
        const auto& used = tracker.usedLights();
        const auto isolated = std::count_if(
          used.begin(), used.end(),
          [](const L3Estimation::UsedLight& light) { return light.isolated; });
        expect(tracker.lastMatchCount() == 1, "侧边灯条帧应当只关联上正对的板");
        expect(isolated == 1, "侧面板靠近正对板的灯条应标记为 isolated 并参与更新");
        expect(
          used.size() == static_cast<std::size_t>(tracker.lastMatchCount() * 2 + 1),
          "灯条显示清单数量与完整板加侧边灯条不一致");
      }
    }
  }

  // --- 3. 丢帧进 TempLost，恢复后回 Tracking -------------------------
  {
    L3Estimation::EskfTracker tracker(
      calibration, armor_config, tracker_config, target_config);

    State truth = makeTruth();
    const VM::Motion motion{.dt = kDt, .name = kName};
    auto now = L3Estimation::TimePoint{} + std::chrono::seconds(1);

    const auto step_once = [&](bool feed_detections) {
      State next;
      motion(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));
      const auto frame = feed_detections
                           ? synthesizeFrame(truth, calibration, armor_config)
                           : std::vector<L2Perception::Armor>{};
      return tracker.track(frame, kBarrelPose, now);
    };

    for (int step = 0; step < 40; ++step) {
      step_once(true);
    }
    expect(tracker.state() == L3Estimation::TrackState::Tracking, "预热后应当在 Tracking");

    // 断供几帧：应当进 TempLost，但仍输出预测状态。
    const auto during_gap = step_once(false);
    expect(
      tracker.state() == L3Estimation::TrackState::TempLost, "丢一帧后应当进入 TempLost");
    expect(during_gap.has_value(), "TempLost 时仍应输出预测状态");
    expect(
      tracker.usedLights().empty(),
      "TempLost 纯预测帧不应泄漏上一帧的显示灯条");
    expect(
      tracker.lightRoi(kBarrelPose, now, calibration.image_size).has_value(),
      "TempLost 未超时时仍应保留独立灯条检测 ROI");

    // 恢复供给：应当回到 Tracking。
    for (int step = 0; step < 3; ++step) {
      step_once(true);
    }
    expect(
      tracker.state() == L3Estimation::TrackState::Tracking, "恢复观测后应当回到 Tracking");

    // 长时间断供：超过 lost_time_thres 应当放弃。
    for (int step = 0; step < 60; ++step) {
      step_once(false);
    }
    expect(tracker.state() == L3Estimation::TrackState::Lost, "长时间无观测后应当放弃目标");
    const auto after_loss = step_once(false);
    expect(!after_loss.has_value(), "Lost 之后 track 应当返回空");
    expect(tracker.armorPoses().empty(), "Lost 之后板位姿列表应当为空");
    expect(
      !tracker.lightRoi(kBarrelPose, now, calibration.image_size).has_value(),
      "Lost 之后不应继续请求独立灯条检测");
  }

  // --- 4. 收敛后 NIS 有意义 ------------------------------------------
  {
    L3Estimation::EskfTracker tracker(
      calibration, armor_config, tracker_config, target_config);

    State truth = makeTruth();
    const VM::Motion motion{.dt = kDt, .name = kName};
    auto now = L3Estimation::TimePoint{} + std::chrono::seconds(1);

    std::optional<L3Estimation::EskfTarget> last;
    for (int step = 0; step < 400; ++step) {
      State next;
      motion(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));
      last = tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    }

    expect(last.has_value(), "400 帧后应当仍在跟踪");
    if (last) {
      expect(last->jumped, "跑满后应当已经见过 0 号以外的板");
      expect(!last->diverged(), "滤波器发散了");
      expect(std::isfinite(last->lastNis()), "NIS 不是有限值");
      expect(last->lastNisDof() > 0, "NIS 自由度应当为正");
      // 每块完整板贡献两根灯条共 8 维端点观测；恰好单板时还会多一维 PnP 深度差。
      expect(
        last->lastNisDof() % 4 == 0 || last->lastNisDof() % 4 == 1,
        "NIS 自由度必须由四维端点块和可选的一维深度差组成");

      // 无噪合成数据下，收敛后的创新量应当很小。
      expect(
        last->lastNis() < static_cast<double>(last->lastNisDof()),
        "无噪数据下 NIS 不应超过自由度");

      const Eigen::VectorXd state = last->ekf_x();
      expect(
        std::abs(state[VM::idx::CX] - truth[VM::idx::CX]) < 0.03, "车心 x 未收敛");
      // ekf_x 必须吐线性半径。
      expect(
        state[VM::idx::LOG_R1] > 0.15 && state[VM::idx::LOG_R1] < 0.5,
        "ekf_x 的第 8 维不像线性半径");
    }
  }

  // --- 5. 网络 ROI 空间注意力 ----------------------------------------
  {
    L3Estimation::EskfTracker tracker(
      calibration, armor_config, tracker_config, target_config);
    const cv::Size image_size = calibration.image_size;
    const cv::Rect full(0, 0, image_size.width, image_size.height);

    // 未跟踪时必须是整图 —— 没有先验就不该缩小搜索范围。
    expect(
      tracker.netFocusRoi(kBarrelPose, L3Estimation::TimePoint{}, image_size, 1.0) == full,
      "未跟踪时网络 ROI 应当是整图");

    State truth = makeTruth();
    const VM::Motion motion{.dt = kDt, .name = kName};
    auto now = L3Estimation::TimePoint{} + std::chrono::seconds(1);
    for (int step = 0; step < 60; ++step) {
      State next;
      motion(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(kDt));
      tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    }
    expect(tracker.state() == L3Estimation::TrackState::Tracking, "预热后应在 Tracking");

    const cv::Rect roi = tracker.netFocusRoi(kBarrelPose, now, image_size, 1.0);
    expect(roi.area() > 0, "跟踪中网络 ROI 不应为空");
    expect(roi.area() < full.area(), "跟踪中网络 ROI 应当小于整图，否则聚焦没生效");
    expect((roi & full) == roi, "网络 ROI 越出了图像边界");
    // 请求 1:1 时应当是方形（未被边界裁剪的情况下）。
    if (roi.x > 0 && roi.y > 0 && roi.br().x < image_size.width &&
        roi.br().y < image_size.height) {
      expect(roi.width == roi.height, "请求 1:1 时 ROI 应当是方形");
    }

    // ROI 必须真的盖住目标：所有预测的板角点都应落在里面。
    {
      const auto poses = tracker.armorPoses();
      expect(!poses.empty(), "跟踪中应当有板位姿");
      const auto frame = synthesizeFrame(truth, calibration, armor_config);
      int covered = 0;
      int total = 0;
      for (const auto& armor : frame) {
        for (const auto& corner : armor.corners) {
          ++total;
          if (roi.contains(cv::Point(
                static_cast<int>(corner.x), static_cast<int>(corner.y)))) {
            ++covered;
          }
        }
      }
      expect(total > 0, "合成帧里应当有可见板");
      expect(covered == total, "网络 ROI 没有盖住全部可见板角点");
    }

    // 随丢失时长膨胀：同一状态、更晚的查询时刻，ROI 只会更大。
    const auto later = now + std::chrono::duration_cast<
                               L3Estimation::TimePoint::duration>(
                               std::chrono::duration<double>(
                                 tracker_config.lost_time_thres * 0.6));
    const cv::Rect grown = tracker.netFocusRoi(kBarrelPose, later, image_size, 1.0);
    expect(
      grown.width >= roi.width && grown.height >= roi.height,
      "ROI 未随丢失时长膨胀");

    // 超过 lost_time_thres 退化为整图：预测已经不可信，不该把网络锁死在一个
    // 大概率错误的小窗口里。
    const auto expired = now + std::chrono::duration_cast<
                                 L3Estimation::TimePoint::duration>(
                                 std::chrono::duration<double>(
                                   tracker_config.lost_time_thres * 1.5));
    expect(
      tracker.netFocusRoi(kBarrelPose, expired, image_size, 1.0) == full,
      "超时后网络 ROI 应当退化为整图");

    // 宽高比修正：请求 16:9 时，方形化之前的比例修正应当让 ROI 变宽。这里只能
    // 观察最终方形边长不小于 1:1 的情形。
    const cv::Rect wide = tracker.netFocusRoi(kBarrelPose, now, image_size, 16.0 / 9.0);
    expect(wide.area() > 0, "16:9 请求下 ROI 不应为空");
    expect(
      wide.width >= roi.width,
      "请求更宽的比例后 ROI 边长不应变小");

    // 枪管姿态缺失时无法把预测投到图像上，只能整图。
    expect(
      tracker.netFocusRoi(std::nullopt, now, image_size, 1.0) == full,
      "无枪管姿态时应当返回整图");
  }

  // --- 6. 断流复位与 TempLost 外推截止 --------------------------------
  {
    L3Estimation::EskfTrackerConfig config = tracker_config;
    config.max_frame_gap = 0.05;
    config.temp_lost_predict_time = 0.02;
    L3Estimation::EskfTracker tracker(calibration, armor_config, config, target_config);

    State truth = makeTruth();
    auto now = L3Estimation::TimePoint{} + std::chrono::seconds(1);
    const auto advance = [&](double seconds) {
      const VM::Motion step{.dt = seconds, .name = kName};
      State next;
      step(truth.data(), next.data());
      truth = next;
      now += std::chrono::duration_cast<L3Estimation::TimePoint::duration>(
        std::chrono::duration<double>(seconds));
    };

    for (int step = 0; step < 40; ++step) {
      advance(kDt);
      tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    }
    expect(tracker.state() == L3Estimation::TrackState::Tracking, "断流前应当在 Tracking");

    // 空帧里逐帧看输出：截止前还在按速度走，截止后原地不动。只比位置和
    // yaw，速度本来就不随外推变。
    const auto pose = [](const L3Estimation::EskfTarget::State & x) {
      return Eigen::Vector4d{x[VM::idx::CX], x[VM::idx::CY], x[VM::idx::CZ], x[VM::idx::ROT_Z]};
    };
    std::vector<Eigen::Vector4d> states;
    for (int step = 0; step < 8; ++step) {
      advance(kDt);
      const auto target = tracker.track({}, kBarrelPose, now);
      expect(target.has_value(), "TempLost 期间应当输出预测");
      if (target) {
        states.push_back(pose(target->rawState()));
      }
    }
    if (states.size() == 8) {
      // 最后一次更新后 5ms、10ms 两帧都在 20ms 截止之前。
      expect(
        (states[1] - states[0]).norm() > 1e-4,
        "外推截止之前预测应当还在按速度移动");
      // 25ms 之后都已过截止，位置和姿态不再变。
      expect(
        (states[7] - states[5]).norm() < 1e-9,
        "超过 temp_lost_predict_time 后预测应当原地保持");
    }
    expect(
      tracker.lightRoi(kBarrelPose, now, calibration.image_size).has_value(),
      "外推截止后未超时的目标仍应给出灯条 ROI");

    // 恢复观测回到 Tracking，再跳过一段超过 max_frame_gap 的时间：旧目标必须
    // 丢弃，这一帧直接按 Lost 用当前检测重建。
    for (int step = 0; step < 3; ++step) {
      advance(kDt);
      tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    }
    advance(0.2);
    const auto rebuilt =
      tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    expect(
      tracker.state() == L3Estimation::TrackState::Detecting,
      "帧间隔超过 max_frame_gap 后应当复位并用当前帧重建");
    expect(rebuilt.has_value() && !rebuilt->jumped, "重建出的目标应当是新初始化的");

    // 时间倒退同样按断流处理。
    for (int step = 0; step < 10; ++step) {
      advance(kDt);
      tracker.track(synthesizeFrame(truth, calibration, armor_config), kBarrelPose, now);
    }
    expect(tracker.state() == L3Estimation::TrackState::Tracking, "重建后应当再次转 Tracking");
    tracker.track(
      synthesizeFrame(truth, calibration, armor_config), kBarrelPose,
      now - std::chrono::milliseconds(20));
    expect(
      tracker.state() == L3Estimation::TrackState::Detecting, "时间倒退后应当复位重建");

    // Detecting 丢帧（本帧只有别的类别）：同一帧就用手上的检测重建，丢弃计数
    // 加一，状态上看起来一直是 Detecting。
    const std::size_t drops = tracker.dropCount();
    auto other = synthesizeFrame(truth, calibration, armor_config);
    for (auto & armor : other) {
      armor.class_id = static_cast<int>(L2Perception::ArmorClass::Infantry4);
    }
    advance(kDt);
    const auto switched = tracker.track(other, kBarrelPose, now);
    expect(
      tracker.state() == L3Estimation::TrackState::Detecting,
      "Detecting 丢帧后应当在同一帧用本帧检测重建");
    expect(tracker.dropCount() == drops + 1, "Detecting 丢帧应当计一次丢弃");
    expect(
      switched.has_value() && switched->name == L3Estimation::ArmorName::Infantry4,
      "重建的目标应当来自本帧的检测");
  }

  if (failure_count != 0) {
    std::cerr << "eskf tracker smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "eskf tracker smoke test passed\n";
  return 0;
}
