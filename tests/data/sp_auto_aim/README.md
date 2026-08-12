# SP auto-aim replay data

`demo.avi` and `demo.txt` are copied byte-for-byte from:

```text
sp_vision_25-main/assets/demo/demo.avi
sp_vision_25-main/assets/demo/demo.txt
```

Each text row is:

```text
timestamp_seconds quaternion_w quaternion_x quaternion_y quaternion_z
```

`records/*.avi` and `records/*.txt` use the same format, and `auto_aim_test`
defaults to `records/3m_high`.

`camera_calibration.yaml` converts the matching camera calibration and
`camera -> gimbal` transform from `sp_vision_25-main/configs/demo.yaml` into
newvision's calibration schema. It is calibrated under SP's convention, so
replays that use it must run with `--convention=sp`, which applies SP's
two-sided IMU-body axis flip before the exposure-time pose reaches
`L3Estimation::ITracker`. The default is `--convention=imu`, this project's
single-sided form, which suits `config/camera_config.yaml`.

Run this dataset from the `newvision` directory:

```text
xmake f --use_openvino=y
xmake run auto_aim_test -- tests/data/sp_auto_aim/demo \
  -c=tests/data/sp_auto_aim/camera_calibration.yaml --convention=sp
```

`--calibration` defaults to `config/camera_config.yaml`, so this dataset needs `-c=` to point back at its
own calibration. Flags must use `-c=value`; `cv::CommandLineParser` does not accept a space-separated
`-c value`.

The test needs a display. It opens two windows:

- `reprojection` — detector corners in red/blue, plus every physical armor of
  the current EKF vehicle reprojected in green, following SP-Vision's
  `auto_aim_test`. In orange, the same vehicle extrapolated `--predict-time`
  seconds ahead by `L4Planning::Predictor` (constant velocity *and* constant
  yaw rate), so the orange-to-green offset is exactly what delay compensation
  has to absorb; `--predict-time=0` turns it off. The rotation center is a
  filled green dot now, a hollow orange circle predicted, joined by a line.
  `cv::waitKey(--wait)` runs after each processed frame, so playback includes
  inference and drawing time; `--wait=0` steps frame by frame on any key.
  `q` or Escape quits.
- `pnp cost` — the PnP yaw-search cost curve for the armor the tracker is
  currently associated with (or, when there is no target, the detection nearest
  the image center).

The cost is exactly what `PnpSolver::armor_reprojection_error` minimizes: the
sum of the four corner reprojection distances under the SP fixed-pitch armor
model. It is sampled every 0.5 degrees across the solver's own ±70 degree
window, centered on the barrel yaw. Vertical markers show the raw single-PnP
yaw (`raw`), the yaw the solver returned (`solver`), the current EKF armor yaw
(`ekf`), and that armor's predicted yaw (`pred`); the red dot is the sampled
minimum. The `local minima` counter in
the header turns orange above 1 — that is the condition under which any
unimodal search (ternary/golden-section) can converge to the wrong branch.

The same values, plus the observation and EKF state, are streamed as JSON to
`127.0.0.1:9870` for PlotJuggler.

The replay runs with `TrackerConfig::require_quality = false`, so an observation
only has to be a committed `single_pnp` pose to reach the EKF. That keeps the
green and orange overlays alive while `ArmorQuality` is still being reworked;
`--require-quality=true` restores the real-robot gate, and the summary line
`观测门限` states which one was used.

`--start-index` seeks the video and the text file together, so the Tracker
starts cold at that frame and its filter state is not equivalent to a full
replay that reaches the same frame.
