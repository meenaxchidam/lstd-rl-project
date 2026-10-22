"""Member B — TD(lambda) tests on the shared mock chain."""
import numpy as np
import pytest

from src.td_lambda import step_size, td_alpha_sweep, td_lambda, td_lambda_curve

GAMMA = 1.0


def _rmse(v_hat, v_true):
    return float(np.sqrt(np.mean((v_hat - v_true) ** 2)))


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_td_converges_on_mock_chain(mock_episodes, mock_phi, mock_v_true, lam, online):
    theta = td_lambda(mock_episodes, mock_phi, GAMMA, lam, alpha=0.1, online=online)
    assert _rmse(mock_phi @ theta, mock_v_true) < 0.05


def test_td_decay_schedule_converges(mock_episodes, mock_phi, mock_v_true):
    theta = td_lambda(mock_episodes, mock_phi, GAMMA, 0.5, alpha=0.1, schedule="decay", n0=1e3)
    assert _rmse(mock_phi @ theta, mock_v_true) < 0.05


def test_step_size_schedules():
    assert step_size(1, 0.1) == step_size(500, 0.1) == 0.1
    assert step_size(1, 0.1, "decay", n0=100) == pytest.approx(0.1)  # paper Eq. 8: alpha_1 = a0
    sizes = [step_size(n, 0.1, "decay", n0=100) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    with pytest.raises(ValueError):
        step_size(1, 0.1, "decay")
    with pytest.raises(ValueError):
        step_size(1, 0.1, "bogus")


def test_one_offline_update_by_hand(mock_phi):
    # theta = 0, lambda = 0, alpha = 0.5: deltas are -1 at states 2 and 1
    episode = [[(2, -1.0, 1, False), (1, -1.0, 0, True)]]
    theta = td_lambda(episode, mock_phi, GAMMA, 0.0, alpha=0.5)
    np.testing.assert_allclose(theta, [0.0, -0.5, -0.5])


def test_curve_last_row_matches_td_lambda(mock_episodes, mock_phi):
    curve = td_lambda_curve(mock_episodes, mock_phi, GAMMA, 0.3, 0.1, [1, 10, 50])
    assert curve.shape == (3, 3)
    np.testing.assert_allclose(curve[-1], td_lambda(mock_episodes, mock_phi, GAMMA, 0.3, 0.1))
    np.testing.assert_allclose(curve[1], td_lambda(mock_episodes[:10], mock_phi, GAMMA, 0.3, 0.1))


def test_zero_step_size_keeps_initial_theta(mock_episodes, mock_phi):
    theta0 = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(td_lambda(mock_episodes, mock_phi, GAMMA, 0.5, 0.0, theta0=theta0), theta0)


def test_divergence_gives_nan_rows(mock_episodes, mock_phi):
    with np.errstate(all="ignore"):
        curve = td_lambda_curve(mock_episodes, mock_phi, GAMMA, 1.0, 1e8, [1, 50])
    assert np.isnan(curve[-1]).all()


def test_alpha_sweep_returns_theta_per_alpha(mock_episodes, mock_phi):
    result = td_alpha_sweep(mock_episodes, mock_phi, GAMMA, 0.5, [0.01, 0.1])
    assert set(result) == {0.01, 0.1}
    assert all(theta.shape == (3,) for theta in result.values())


def test_invalid_checkpoints_raise(mock_episodes, mock_phi):
    with pytest.raises(ValueError):
        td_lambda_curve(mock_episodes, mock_phi, GAMMA, 0.5, 0.1, [10, 5])
    with pytest.raises(ValueError):
        td_lambda_curve(mock_episodes, mock_phi, GAMMA, 0.5, 0.1, [51])
