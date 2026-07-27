import numpy as np
import torch
import unittest

from verl.trainer.ppo.grsd_reflect_utils import compute_grsd_token_advantage
from verl.trainer.ppo.grsd_ray_trainer import GRSDRayTrainer
from verl.trainer.ppo.grsd_utils import compute_grsd_turn_advantage
from verl.trainer.ppo.rlsd_utils import compute_rlsd_token_advantage


def _common_inputs(delta, dtype=torch.float32):
    delta = torch.tensor(delta, dtype=dtype)
    return {
        "seq_advantages": torch.ones_like(delta),
        "student_log_probs": torch.zeros_like(delta),
        "teacher_log_probs": delta,
        "response_mask": torch.ones_like(delta),
        "traj_index": np.arange(delta.size(0)),
        "valid_mask": np.ones(delta.size(0), dtype=np.float32),
        "uid_index": np.zeros(delta.size(0), dtype=np.int64),
        "grsd_lambda": 0.5,
        "eta": 0.0,
        "g_hat_max": 10.0,
    }


class TestGrsdExponentialAdvantage(unittest.TestCase):
    def test_turn_advantage_uses_zero_centered_exponential_signal(self):
        inputs = _common_inputs([[1.0], [0.0], [-1.0]])

        advantages, metrics = compute_grsd_turn_advantage(**inputs)

        signal = torch.expm1(torch.tensor([1.0, 0.0, -1.0]))
        q_norm = signal / signal.abs().mean()
        expected = 1.0 + 0.5 * q_norm
        torch.testing.assert_close(advantages[:, 0], expected)
        self.assertAlmostEqual(metrics["grsd/aligned_gap_mean"], 0.0)
        self.assertAlmostEqual(metrics["grsd/exponential_signal_mean"], signal.mean().item())

    def test_token_advantage_uses_zero_centered_exponential_signal(self):
        for token_norm in ("group", "none"):
            with self.subTest(token_norm=token_norm):
                inputs = _common_inputs([[1.0, 0.0, -1.0]])

                advantages, metrics = compute_grsd_token_advantage(**inputs, token_norm=token_norm)

                signal = torch.expm1(torch.tensor([1.0, 0.0, -1.0]))
                if token_norm == "group":
                    signal = signal / signal.abs().mean()
                expected = 1.0 + 0.5 * signal
                torch.testing.assert_close(advantages[0], expected)
                self.assertAlmostEqual(metrics["grsd/aligned_gap_mean"], 0.0)

    def test_token_none_matches_rlsd_exponential_mixture(self):
        inputs = _common_inputs([[0.5, -0.5]])
        inputs["epsilon"] = 0.1

        advantages, _ = compute_grsd_token_advantage(**inputs, token_norm="none")

        q = torch.tensor([0.5, -0.5])
        expected = (1.0 - inputs["grsd_lambda"]) + inputs["grsd_lambda"] * torch.exp(q)
        torch.testing.assert_close(advantages[0], expected)

    def test_token_none_matches_rlsd_implementation_at_equal_clip(self):
        inputs = _common_inputs([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
        inputs["seq_advantages"][1] = -1.0
        inputs["g_hat_max"] = 0.2

        grsd_advantages, _ = compute_grsd_token_advantage(**inputs, token_norm="none")
        rlsd_advantages = compute_rlsd_token_advantage(
            seq_advantages=inputs["seq_advantages"],
            student_log_probs=inputs["student_log_probs"],
            teacher_log_probs=inputs["teacher_log_probs"],
            response_mask=inputs["response_mask"],
            rlsd_lambda=inputs["grsd_lambda"],
            rlsd_clip_eps=inputs["g_hat_max"],
        )

        torch.testing.assert_close(grsd_advantages, rlsd_advantages)

    def test_exponential_signal_is_finite_for_extreme_logprob_gaps(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                inputs = _common_inputs([[1000.0, -1000.0]], dtype=dtype)

                advantages, metrics = compute_grsd_token_advantage(**inputs, token_norm="none")

                self.assertTrue(torch.isfinite(advantages).all())
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_token_advantage_keeps_invalid_groups_and_padding_unmodulated(self):
        inputs = _common_inputs([[2.0, 2.0], [2.0, 2.0]])
        inputs["valid_mask"] = np.array([0.0, 1.0], dtype=np.float32)
        inputs["uid_index"] = np.array([0, 1], dtype=np.int64)
        inputs["response_mask"] = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

        advantages, _ = compute_grsd_token_advantage(**inputs, token_norm="none")

        torch.testing.assert_close(advantages[0], torch.ones(2))
        self.assertGreater(advantages[1, 0], 1.0)
        self.assertEqual(advantages[1, 1], 0.0)


class TestGrsdLambdaSchedule(unittest.TestCase):
    @staticmethod
    def _trainer(warmdown_steps):
        trainer = GRSDRayTrainer.__new__(GRSDRayTrainer)
        trainer.grsd_lambda_init = 0.5
        trainer.grsd_warmdown_steps = warmdown_steps
        return trainer

    def test_non_positive_decay_keeps_lambda_fixed(self):
        for decay_steps in (-1, 0):
            trainer = self._trainer(decay_steps)
            self.assertEqual(trainer._get_grsd_lambda(1), 0.5)
            self.assertEqual(trainer._get_grsd_lambda(1000), 0.5)

    def test_positive_decay_is_linear_and_clamped_at_zero(self):
        trainer = self._trainer(100)
        self.assertEqual(trainer._get_grsd_lambda(0), 0.5)
        self.assertEqual(trainer._get_grsd_lambda(50), 0.25)
        self.assertAlmostEqual(trainer._get_grsd_lambda(99), 0.005)
        self.assertEqual(trainer._get_grsd_lambda(100), 0.0)
        self.assertEqual(trainer._get_grsd_lambda(101), 0.0)


if __name__ == "__main__":
    unittest.main()
