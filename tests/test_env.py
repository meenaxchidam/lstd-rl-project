"""Member A — environment, feature and ground-truth tests."""
import numpy as np
import pytest

from src.env import BoyanChain, RandomWalk, generate_episodes, true_values
from src.features import boyan_features, random_features, random_walk_features

PAPER_THETA = np.array([-24.0, -16.0, -8.0, 0.0])  # paper Sec. 4


# ----------------------------------------------------------------------------- Boyan chain

def test_boyan_episodes_start_at_12_and_end_at_0():
    for episode in generate_episodes(BoyanChain(), 200, seed=0):
        assert episode[0][0] == 12
        assert episode[-1][2] == 0 and episode[-1][3]
        assert not any(done for *_, done in episode[:-1])


def test_boyan_transitions_and_rewards_follow_spec():
    for episode in generate_episodes(BoyanChain(), 200, seed=1):
        for s, r, s_next, _ in episode:
            if s == 1:
                assert (s_next, r) == (0, -2.0)
            else:
                assert s - s_next in (1, 2) and r == -3.0


def test_boyan_transition_probabilities_are_half():
    episodes = generate_episodes(BoyanChain(), 2000, seed=2)
    from_12 = [s_next for ep in episodes for s, _, s_next, _ in ep if s == 12]
    assert abs(np.mean(np.array(from_12) == 11) - 0.5) < 0.05


def test_generate_episodes_is_reproducible():
    env = BoyanChain()
    assert generate_episodes(env, 20, seed=7) == generate_episodes(env, 20, seed=7)
    assert generate_episodes(env, 20, seed=7) != generate_episodes(env, 20, seed=8)


def test_step_after_termination_raises():
    env = BoyanChain()
    env.reset()
    done = False
    while not done:
        _, _, done = env.step()
    with pytest.raises(RuntimeError):
        env.step()


def test_boyan_true_values_hand_computed():
    v = true_values(BoyanChain())
    assert v[0] == 0.0
    assert v[1] == pytest.approx(-2.0)
    assert v[2] == pytest.approx(-3.0 + 0.5 * -2.0 + 0.5 * 0.0)  # -4
    np.testing.assert_allclose(v, -2.0 * np.arange(13), atol=1e-12)


# ----------------------------------------------------------------------------- Boyan features

def test_boyan_features_shape_rows_and_anchors():
    phi = boyan_features()
    assert phi.shape == (13, 4)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0)
    for s, k in ((12, 0), (8, 1), (4, 2), (0, 3)):
        np.testing.assert_array_equal(phi[s], np.eye(4)[k])


def test_boyan_features_match_paper_figure_4():
    phi = boyan_features()
    np.testing.assert_allclose(phi[11], [0.75, 0.25, 0, 0])
    np.testing.assert_allclose(phi[10], [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(phi[3], [0, 0, 0.75, 0.25])
    np.testing.assert_allclose(phi[2], [0, 0, 0.5, 0.5])
    np.testing.assert_allclose(phi[1], [0, 0, 0.25, 0.75])


def test_true_value_is_exactly_representable_with_paper_theta():
    np.testing.assert_allclose(boyan_features() @ PAPER_THETA, true_values(BoyanChain()), atol=1e-12)


# ----------------------------------------------------------------------------- random walk

def test_random_walk_true_values():
    env = RandomWalk(19)
    v = true_values(env)
    s = np.arange(1, 20)
    np.testing.assert_allclose(v[1:20], (s - 10) / 10, atol=1e-12)
    assert v[0] == v[20] == 0.0


def test_random_walk_episodes_terminate_with_correct_reward():
    for episode in generate_episodes(RandomWalk(19), 100, seed=0):
        assert episode[0][0] == 10
        s, r, s_next, done = episode[-1]
        assert done and s_next in (0, 20)
        assert r == (-1.0 if s_next == 0 else 1.0)
        assert all(r == 0.0 for _, r, _, _ in episode[:-1])


@pytest.mark.parametrize("kind,d", [("tabular", 19), ("aggregation", 5)])
def test_random_walk_features(kind, d):
    phi = random_walk_features(19, kind=kind, num_groups=5)
    assert phi.shape == (21, d)
    np.testing.assert_array_equal(phi[[0, 20]], 0.0)
    np.testing.assert_allclose(phi[1:20].sum(axis=1), 1.0)


def test_random_features_unit_norm():
    phi = random_features(13, 32, seed=0)
    assert phi.shape == (13, 32)
    np.testing.assert_allclose(np.linalg.norm(phi, axis=1), 1.0)
