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

`camera_calibration.yaml` converts the matching camera calibration and
`camera -> gimbal` transform from `sp_vision_25-main/configs/demo.yaml` into
newvision's calibration schema. The replay applies SP's recorded IMU-body axis
conversion before passing the exposure-time pose to `L3Estimation::Tracker`.

Run from the `newvision` directory:

```text
xmake f --use_openvino=y
xmake build auto_aim_test
xmake run auto_aim_test
```

The default run is headless and writes per-frame data to
`logs/sp_auto_aim_replay.csv`. Display the replay at 30 FPS with:

```text
xmake run auto_aim_test -- --show=true
```

The overlay is generated only by the test executable. Following SP-Vision's
`auto_aim_test`, it generates every physical armor pose from the current EKF
target and reprojects all armor outlines in green. The display loop also
matches SP-Vision and calls `cv::waitKey(30)` after each processed frame, so
the actual playback rate includes inference and drawing time and can be below
30 FPS.

The green vehicle overlay is display-only stabilized: consecutive projected
corners use an exponential smoothing factor of 0.4, and a missing projection
is retained in dark green for at most five frames. A jump larger than 120
pixels bypasses smoothing to avoid drawing a long false trail. This does not
modify Tracker state, association, CSV values, or the IPPE diagnostics.

When `--show=true`, a second `newvision yaw cost` window plots the exact
one-dimensional cost used by the current 140-degree yaw scan: the sum of the
four corner reprojection distances with the SP fixed-pitch model. It marks the
lower- and higher-RMSE IPPE solutions, the searched observation yaw, and the
closest EKF armor yaw. The replay CSV contains the same values per frame.

The IPPE branch reported as `EKF-nearest` is diagnostic only. The current
production path still calls single-result `solvePnP`, applies the yaw scan,
and sends only that searched yaw to the EKF; the test recomputes both IPPE
solutions without changing Tracker behavior.

Press Space to pause or resume, `n` to advance one frame while paused, and
`q` or Escape to quit. To reproduce the full-replay filter state around a
reported frame, keep `--start-index=0` and use, for example:

```text
xmake run auto_aim_test -- --show=true --show-from-index=380 --end-index=390
```

Frames before 380 are processed without display delay, so Tracker reaches the
same state as the full replay. Starting directly at frame 380 would instead
initialize a fresh Tracker and is not equivalent.
