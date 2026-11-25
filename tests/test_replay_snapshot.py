"""Restore decision states into different arenas and reproduce their next actions."""
import unittest
import torch
import carl
from carl.gymnasium import CARLTorchVectorEnv


class ReplaySnapshotTest(unittest.TestCase):
    def test_restore_at_different_ticks_and_lanes(self):
        for team_size in (1, 2):
            with self.subTest(team_size=team_size):
                self.round_trip(team_size)

    def round_trip(self, team_size):
        cars = 2 * team_size
        mode = f"{team_size}v{team_size}"

        def environment(seed):
            return CARLTorchVectorEnv(4, team_size, team_size, seed=seed, frameskip=8,
                                      discrete_actions=True, normalize=True, invert_orange=True)
        source = environment(142)
        target = environment(531)
        try:
            source.reset()
            target.reset()
            take = torch.tensor([3, 1], device=source.device)
            put = torch.tensor([0, 2], device=source.device)
            rng = torch.Generator(device=source.device).manual_seed(773)
            shape = (4, cars, 7)
            neutral = torch.tensor([1, 1, 2, 0, 0, 1, 0], dtype=torch.int32, device=source.device)
            buttons = neutral.expand(shape).clone()
            next_input = neutral.expand(shape).clone()
            for frame in range(65):
                if frame in (0, 4, 19, 33, 64):
                    tape = torch.zeros((2, getattr(carl, f"REPLAY_{mode.upper()}_WORDS")), dtype=torch.int32, device=source.device)
                    getattr(source._env, f"capture_replay_{mode}")(take, tape)
                    getattr(target._env, f"restore_replay_{mode}")(put, tape)
                    def selected(env, getter, ids):
                        data = env._tensor(getattr(env._env, getter)()).reshape(4, -1)
                        return data[ids].clone()
                    for getter in ("get_obs", "get_state"):
                        torch.testing.assert_close(selected(source, getter, take),
                                                   selected(target, getter, put), rtol=0, atol=0)
                    next_input[put] = buttons[take]
                    target.step(next_input.reshape(4 * cars, 7))
                    source.step(buttons.reshape(4 * cars, 7))
                    for getter in ("get_obs", "get_transition_state"):
                        torch.testing.assert_close(selected(source, getter, take),
                                                   selected(target, getter, put), rtol=0, atol=0)
                else:
                    source.step(buttons.reshape(4 * cars, 7))
                buttons[..., 0] = torch.randint(3, (4, cars), generator=rng, device=source.device)
                buttons[..., 4] = 1
                buttons[..., 6] = int(frame in (5, 6, 12, 35))
        finally:
            source.close()
            target.close()


if __name__ == "__main__":
    unittest.main()
