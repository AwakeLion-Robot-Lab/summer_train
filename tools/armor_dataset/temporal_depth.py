"""Causal depth/velocity filtering. One filter state belongs to one tracked digit instance."""

from collections import deque
from dataclasses import asdict, dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class DepthFilterConfig:
    process_noise_m2_s3: float = 0.5
    measurement_std_m: float = 0.04
    measurement_relative_std: float = 0.015
    initial_velocity_std_m_s: float = 1.0
    gate_sigma: float = 4.0
    max_gap_s: float = 0.25
    max_prediction_s: float = 0.10
    outlier_reset_count: int = 3

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid temporal filter setting: {key}")
        if self.max_prediction_s > self.max_gap_s:
            raise ValueError("max_prediction_s must not exceed max_gap_s")
        if not isinstance(self.outlier_reset_count, int) or self.outlier_reset_count < 2:
            raise ValueError("outlier_reset_count must be an integer >=2")


class DepthKalmanFilter:
    """State is [camera optical-axis Z, dZ/dt], with continuous white acceleration Q.

    Timestamps must be capture times in seconds on a single monotonic clock.
    The innovation gate rejects isolated spikes. A short consistent new level or
    a measurement gap resets the state, so genuine changes cannot freeze it.
    """

    def __init__(self, config=None):
        self.config = config or DepthFilterConfig()
        self.reset()

    def reset(self):
        self.x = None
        self.p = None
        self.timestamp_s = None
        self.last_measurement_s = None
        self.outliers = deque(maxlen=self.config.outlier_reset_count)

    def _noise_std(self, measurement):
        return max(self.config.measurement_std_m,
                   self.config.measurement_relative_std*abs(measurement))

    def _initialize(self, measurement, timestamp, variance):
        self.x = np.array([measurement, 0.0], float)
        self.p = np.diag([variance, self.config.initial_velocity_std_m_s**2])
        self.timestamp_s = self.last_measurement_s = timestamp
        self.outliers.clear()

    def update(self, depth_z_m, timestamp_s, measurement_std_m=None):
        started = time.perf_counter()
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("Capture timestamp must be finite")
        raw = float(depth_z_m) if depth_z_m is not None else None
        observed = raw is not None and math.isfinite(raw) and raw > 0
        if not observed:
            raw = None
        sigma = self._noise_std(raw if observed else (self.x[0] if self.x is not None else 1.0))
        if measurement_std_m is not None:
            sigma = float(measurement_std_m)
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("Measurement standard deviation must be positive and finite")
        variance = sigma*sigma
        status, accepted, innovation, nis = "waiting", False, None, None
        if self.x is not None and timestamp <= self.timestamp_s:
            self.reset()
            status = "reset_clock"
        if self.x is None:
            if observed:
                self._initialize(raw, timestamp, variance)
                accepted = True
                if status == "waiting":
                    status = "initialized"
        elif observed and timestamp-self.last_measurement_s > self.config.max_gap_s:
            self._initialize(raw, timestamp, variance)
            status, accepted = "reset_gap", True
        else:
            dt = timestamp-self.timestamp_s
            transition = np.array([[1.0, dt], [0.0, 1.0]])
            process = self.config.process_noise_m2_s3*np.array(
                [[dt**3/3, dt*dt/2], [dt*dt/2, dt]])
            self.x = transition@self.x
            self.p = transition@self.p@transition.T+process
            self.timestamp_s = timestamp
            if observed:
                innovation = raw-float(self.x[0])
                innovation_variance = float(self.p[0, 0]+variance)
                nis = innovation*innovation/innovation_variance
                if nis <= self.config.gate_sigma**2:
                    gain = self.p[:, 0]/innovation_variance
                    self.x += gain*innovation
                    identity_minus_gain = np.eye(2)-np.outer(gain, [1.0, 0.0])
                    self.p = identity_minus_gain@self.p@identity_minus_gain.T+np.outer(gain, gain)*variance
                    self.p = (self.p+self.p.T)/2
                    self.last_measurement_s = timestamp
                    self.outliers.clear()
                    status, accepted = "updated", True
                else:
                    self.outliers.append(raw)
                    status = "outlier_rejected"
                    # Reacquire a consistent changed depth without accepting a lone spike.
                    if len(self.outliers) == self.config.outlier_reset_count:
                        level = float(np.median(self.outliers))
                        spread = max(self.outliers)-min(self.outliers)
                        if spread <= max(0.20, 4*self._noise_std(level)):
                            self._initialize(raw, timestamp, variance)
                            status, accepted = "reset_new_level", True
            else:
                self.outliers.clear()
                status = "predicted"
        age = timestamp-self.last_measurement_s if self.x is not None else None
        valid = (self.x is not None and age <= self.config.max_prediction_s+1e-9
                 and math.isfinite(self.x[0]) and self.x[0] > 0)
        if self.x is not None and not valid:
            status = "expired" if age > self.config.max_prediction_s else "invalid_prediction"
        return dict(depth_z_m=float(self.x[0]) if valid else None, raw_depth_z_m=raw,
                    valid=bool(valid), velocity_z_m_s=float(self.x[1]) if valid else None,
                    depth_std_m=math.sqrt(max(0, float(self.p[0, 0]))) if valid else None,
                    measurement_accepted=accepted, measurement_age_s=age,
                    innovation_m=innovation, normalized_innovation_squared=nis,
                    status=status, timestamp_s=timestamp,
                    filter_compute_ms=(time.perf_counter()-started)*1000)


class DepthFilterBank:
    def __init__(self, config=None):
        self.config = config or DepthFilterConfig()
        self.filters = {}

    def reset(self, target_id=None):
        if target_id is None:
            self.filters.clear()
        else:
            self.filters.pop(target_id, None)

    def update(self, target_id, depth_z_m, timestamp_s):
        if target_id is None:
            raise ValueError("Temporal filtering requires a stable instance ID, not a digit class")
        if target_id not in self.filters:
            self.filters[target_id] = DepthKalmanFilter(self.config)
        state = self.filters[target_id]
        return state.update(depth_z_m, timestamp_s)

    def prune(self, timestamp_s):
        for key, state in list(self.filters.items()):
            if state.last_measurement_s is None or timestamp_s-state.last_measurement_s > self.config.max_gap_s:
                del self.filters[key]


class DigitBoxAssociator:
    """Approximate offline association using only 2D digit boxes, never distance labels.

    Live callers should supply their digit track IDs. The evaluator has only sparse
    teacher-guided boxes, so its generated identities are not identity ground truth.
    """

    def __init__(self, max_gap_s=0.25):
        self.max_gap_s = max_gap_s
        self.tracks = {}
        self.next_id = 0
        self.timestamp = None

    def update(self, boxes, timestamp):
        if self.timestamp is not None and timestamp <= self.timestamp:
            self.tracks.clear()
        self.timestamp = timestamp
        self.tracks = {key: value for key, value in self.tracks.items()
                       if timestamp-value["time"] <= self.max_gap_s}
        boxes = [np.asarray(box, float) for box in boxes]
        centers = [(box[:2]+box[2:])/2 for box in boxes]
        candidates = []
        for key, previous in self.tracks.items():
            dt = timestamp-previous["time"]
            predicted = previous["center"]+previous["velocity"]*dt
            old_size = previous["box"][2:]-previous["box"][:2]
            for index, (box, center) in enumerate(zip(boxes, centers)):
                size = box[2:]-box[:2]
                if (size <= 0).any():
                    raise ValueError("Digit box must have positive width and height")
                relative_height = abs(math.log(size[1]/old_size[1]))
                distance = float(np.linalg.norm(center-predicted)/max(8.0, np.linalg.norm(old_size)))
                if distance < 1.5 and relative_height < math.log(1.8):
                    candidates.append((distance+relative_height, key, index))
        assigned, used_tracks = {}, set()
        for _, key, index in sorted(candidates):
            if key not in used_tracks and index not in assigned:
                assigned[index] = key
                used_tracks.add(key)
        result = []
        for index, (box, center) in enumerate(zip(boxes, centers)):
            if (box[2:] <= box[:2]).any():
                raise ValueError("Digit box must have positive width and height")
            key = assigned.get(index)
            velocity = np.zeros(2)
            if key is None:
                key = self.next_id
                self.next_id += 1
            else:
                previous = self.tracks[key]
                dt = timestamp-previous["time"]
                velocity = .5*previous["velocity"]+.5*(center-previous["center"])/dt
            self.tracks[key] = dict(box=box, center=center, velocity=velocity, time=timestamp)
            result.append(key)
        return result
