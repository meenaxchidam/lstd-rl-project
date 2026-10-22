"""Member C — LSTD(lambda), ridge and recursive LSTD tests."""
import numpy as np
import pytest

from src.lstd import lstd_lambda, lstd_lambda_curve, lstd_statistics, recursive_lstd, solve_theta

GAMMA = 1.0


def _random_episodes(n, n_states=5, seed=0):
    """Random terminating chain on states 0..n_states-1 (0 terminal), random rewards."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n):
        s, episode = n_states - 1, []
        while True:
            s_next = int(rng.integers(0, n_states))
            r = float(rng.normal())
            episode.append((s, r, s_next, s_next == 0))
            if s_next == 0:
                break
            s = s_next
        episodes.append(episode)
    return episodes


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_lstd_exact_on_mock_chain(mock_episodes, mock_phi, mock_v_true, lam):
    theta = lstd_lambda(mock_episodes, mock_phi, GAMMA, lam)
    np.testing.assert_allclose(mock_phi @ theta, mock_v_true, atol=1e-10)


def test_lstd_exact_from_a_single_episode(mock_episodes, mock_phi, mock_v_true):
    # deterministic chain: one trajectory already pins down V (A is singular -> SVD fallback)
    np.testing.assert_allclose(mock_phi @ lstd_lambda(mock_episodes[:1], mock_phi, GAMMA, 0.4), mock_v_true, atol=1e-10)


def test_statistics_by_hand(mock_phi):
    A, b = lstd_statistics([[(2, -1.0, 1, False), (1, -1.0, 0, True)]], mock_phi, GAMMA, 0.0)
    expected_A = np.zeros((3, 3))
    expected_A[2, 2], expected_A[2, 1], expected_A[1, 1] = 1.0, -1.0, 1.0
    np.testing.assert_allclose(A, expected_A)
    np.testing.assert_allclose(b, [0.0, -1.0, -1.0])


def test_ridge_solves_regularised_system(mock_episodes, mock_phi):
    A, b = lstd_statistics(mock_episodes, mock_phi, GAMMA, 0.5)
    theta = lstd_lambda(mock_episodes, mock_phi, GAMMA, 0.5, reg=2.0)
    np.testing.assert_allclose((A + 2.0 * np.eye(3)) @ theta, b, atol=1e-10)
    assert np.linalg.norm(theta) < np.linalg.norm(lstd_lambda(mock_episodes, mock_phi, GAMMA, 0.5))


def test_solve_theta_falls_back_on_singular_matrix():
    theta = solve_theta(np.zeros((2, 2)), np.zeros(2))
    np.testing.assert_array_equal(theta, [0.0, 0.0])


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_recursive_matches_batch(mock_episodes, mock_phi, lam):
    delta = 1e3
    np.testing.assert_allclose(
        recursive_lstd(mock_episodes, mock_phi, GAMMA, lam, delta=delta),
        lstd_lambda(mock_episodes, mock_phi, GAMMA, lam, reg=1.0 / delta),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        recursive_lstd(mock_episodes, mock_phi, GAMMA, lam, delta=1e8),
        lstd_lambda(mock_episodes, mock_phi, GAMMA, lam),
        atol=1e-6,
    )


def test_recursive_matches_batch_on_random_chain():
    episodes = _random_episodes(100, seed=3)
    phi = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(
        recursive_lstd(episodes, phi, 0.9, 0.7, delta=1e2),
        lstd_lambda(episodes, phi, 0.9, 0.7, reg=1e-2),
        atol=1e-8,
    )


def test_curve_matches_prefix_solutions(mock_episodes, mock_phi):
    curve = lstd_lambda_curve(mock_episodes, mock_phi, GAMMA, 0.5, [1, 7, 50], reg=0.1)
    for row, n in zip(curve, (1, 7, 50)):
        np.testing.assert_allclose(row, lstd_lambda(mock_episodes[:n], mock_phi, GAMMA, 0.5, reg=0.1))


def test_lstd0_tabular_equals_model_based_solution():
    """Paper Sec. 3: with tabular features LSTD(0) gives v = (N - C)^{-1} s."""
    n_states = 5
    episodes = _random_episodes(200, n_states=n_states, seed=1)
    N, C, s_sum = np.zeros(n_states), np.zeros((n_states, n_states)), np.zeros(n_states)
    for episode in episodes:
        for s, r, s_next, done in episode:
            N[s] += 1
            s_sum[s] += r
            if not done:
                C[s, s_next] += 1
    nt = slice(1, n_states)
    v_model = np.linalg.solve((np.diag(N) - C)[nt, nt], s_sum[nt])
    theta = lstd_lambda(episodes, np.eye(n_states), GAMMA, 0.0)
    np.testing.assert_allclose(theta[nt], v_model, atol=1e-8)


def test_lstd1_equals_monte_carlo_regression():
    """Paper Sec. 2: LSTD(1) = least-squares regression of features onto observed returns."""
    episodes = _random_episodes(200, seed=2)
    phi = np.random.default_rng(1).normal(size=(5, 3))
    X, y = [], []
    for episode in episodes:
        rewards = [r for _, r, _, _ in episode]
        for t, (s, *_rest) in enumerate(episode):
            X.append(phi[s])
            y.append(sum(rewards[t:]))
    theta_mc = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
    np.testing.assert_allclose(lstd_lambda(episodes, phi, GAMMA, 1.0), theta_mc, atol=1e-8)
