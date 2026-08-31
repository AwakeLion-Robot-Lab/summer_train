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
`L3Estimation::Tracker`. The default is `--convention=imu`, this project's
single-sided form, which suits `config/camera_config.yaml`.

Run this dataset from the `newvision` directory:

```text
xmake f --use_openvino=y
xmake run auto_aim_test -- tests/data/sp_auto_aim/demo \
  -c=tests/data/sp_auto_aim/camera_calibration.yaml --convention=sp
```

The default estimator is the awakening-style iterated error-state EKF with UVL
light-bar observations. Use `--estimator=ekf` to replay the old YPDA EKF on the
same frames; `--estimator=ieskf` (also `iesekf` or `esekf`) selects the new path
explicitly.
Both paths continue through the same L4 planner and L5 fire-decision code, so
the command and switching statistics are directly comparable.

`--calibration` defaults to `config/camera_config.yaml`, so this dataset needs `-c=` to point back at its
own calibration. Flags must use `-c=value`; `cv::CommandLineParser` does not accept a space-separated
`-c value`.

The test needs a display. It opens two windows:

- `reprojection` — detector corners in red/blue, plus every physical armor of
  the current estimated vehicle reprojected in green, following SP-Vision's
  `auto_aim_test`. In orange, the estimator's target snapshot is extrapolated
  `--predict-time` seconds with its own motion model, so the orange-to-green
  offset is exactly what delay compensation has to absorb;
  `--predict-time=0` turns it off.
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

The same values, plus the observation, estimator state, NIS and NIS degrees of
freedom, are streamed as JSON to `127.0.0.1:9870` for PlotJuggler.
The per-frame console line reports the filter's center, velocity, yaw rate,
radii, height offsets, roll/pitch, and NIS; detector/tracker latency is not
printed. The same filter state is visible in both `sp` and `full` overlays.

The old EKF requires a committed `single_pnp` pose on every accepted
observation and then corrects with YPDA. The IESKF follows Awakening's split
entry: PnP is required for target initialization, while normal frames associate
the class and image corners directly. A matched full armor contributes two UVL
light-bar observations; once tracking, independent traditional light bars are
detected inside the predicted 1.6x vehicle ROI and associated with
length/angle/position gates. If exactly one full armor is matched, IPPE also
contributes the one-dimensional left-minus-right light-center depth difference.
The PnP calculations retained by this replay for the full-view cost plot are
diagnostics only and never enter the IESKF update.

`--start-index` seeks the video and the text file together, so the Tracker
starts cold at that frame and its filter state is not equivalent to a full
replay that reaches the same frame.
