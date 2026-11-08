#!/usr/bin/env python3
"""Deterministic behavioural tests; no recording, model weights or PyTorch required."""

import unittest

import numpy as np

from temporal_depth import DepthFilterBank, DepthFilterConfig, DepthKalmanFilter, DigitBoxAssociator


class TemporalDepthTests(unittest.TestCase):
    def test_stationary_noise_is_reduced(self):
        rng=np.random.default_rng(17)
        raw=3+rng.normal(0,.05,600)
        f=DepthKalmanFilter()
        filtered=[f.update(float(z),i/30)["depth_z_m"] for i,z in enumerate(raw)]
        self.assertLess(np.std(filtered[100:]),.8*np.std(raw[100:]))

    def test_constant_velocity_has_little_steady_tracking_bias(self):
        f=DepthKalmanFilter()
        errors=[]
        for i in range(180):
            t=i/30;truth=3+.6*t;r=f.update(truth,t)
            if t>1:errors.append(r["depth_z_m"]-truth)
        self.assertLess(abs(np.mean(errors)),.01)
        self.assertAlmostEqual(r["velocity_z_m_s"],.6,places=2)

    def test_lone_spike_does_not_jump(self):
        f=DepthKalmanFilter()
        for i in range(60):f.update(3,i/30)
        r=f.update(5,2)
        self.assertFalse(r["measurement_accepted"])
        self.assertEqual(r["status"],"outlier_rejected")
        self.assertLess(abs(r["depth_z_m"]-3),.02)
        self.assertTrue(f.update(3,2+1/30)["measurement_accepted"])

    def test_real_step_reacquires_in_three_observations(self):
        f=DepthKalmanFilter()
        for i in range(60):f.update(3,i/30)
        for i in range(3):r=f.update(3.8,2+i/30)
        self.assertTrue(r["measurement_accepted"])
        self.assertAlmostEqual(r["depth_z_m"],3.8,places=2)

    def test_prediction_expires_and_long_gap_resets(self):
        f=DepthKalmanFilter()
        f.update(3,0)
        self.assertTrue(f.update(None,.05)["valid"])
        r=f.update(None,.11)
        self.assertFalse(r["valid"]);self.assertIsNone(r["depth_z_m"])
        r=f.update(4,.4)
        self.assertEqual(r["status"],"reset_gap")
        self.assertAlmostEqual(r["depth_z_m"],4)

    def test_clock_rollback_and_duplicate_reset(self):
        f=DepthKalmanFilter();f.update(3,10)
        r=f.update(4,1)
        self.assertEqual(r["status"],"reset_clock")
        self.assertEqual(r["depth_z_m"],4)
        self.assertEqual(f.update(5,1)["status"],"reset_clock")

    def test_invalid_values_do_not_poison_state(self):
        f=DepthKalmanFilter();f.update(3,0)
        for i,z in enumerate([float("nan"),float("inf"),-1,0]):
            r=f.update(z,.01*(i+1))
            self.assertTrue(r["valid"])
            self.assertAlmostEqual(r["depth_z_m"],3)
        with self.assertRaises(ValueError):f.update(3,float("nan"))

    def test_tracks_with_same_digit_are_independent(self):
        bank=DepthFilterBank()
        for i in range(30):
            a=bank.update("same_digit_instance_a",2,i/30)
            b=bank.update("same_digit_instance_b",5,i/30)
        self.assertAlmostEqual(a["depth_z_m"],2)
        self.assertAlmostEqual(b["depth_z_m"],5)
        with self.assertRaises(ValueError):bank.update(None,3,1)

    def test_covariance_remains_positive_with_irregular_timing(self):
        rng=np.random.default_rng(4);f=DepthKalmanFilter();t=0
        for _ in range(300):
            t+=rng.uniform(.006,.08)
            f.update(3+.2*np.sin(t)+rng.normal(0,.03),t)
            self.assertTrue(np.isfinite(f.p).all())
            self.assertGreaterEqual(np.linalg.eigvalsh(f.p).min(),-1e-10)

    def test_configuration_is_checked(self):
        with self.assertRaises(ValueError):DepthFilterConfig(measurement_std_m=0)
        with self.assertRaises(ValueError):DepthFilterConfig(max_prediction_s=1)

    def test_box_association_survives_detection_order_changes(self):
        a=DigitBoxAssociator()
        ids=a.update([[10,10,30,50],[100,10,120,50]],0)
        reversed_ids=a.update([[101,10,121,50],[11,10,31,50]],.03)
        self.assertEqual(ids,reversed_ids[::-1])
        after_gap=a.update([[11,10,31,50]],1)
        self.assertNotIn(after_gap[0],ids)


if __name__=="__main__":
    unittest.main()
