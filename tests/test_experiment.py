"""Member D — metric tests, end-to-end smoke run and a Boyan-chain integration check."""
import numpy as np
import pytest

from src import plot_results
from src.env import BoyanChain, generate_episodes, true_values
from src.features import boyan_features
from src.lstd import lstd_lambda
from src.run_experiment import (
    DEFAULT_CONFIG, load_config, mean_se, rmse, run_learning_curves, run_ridge, run_timing, td_configs,
)
from src.td_lambda import td_lambda


def test_rmse():
    v = np.array([0.0, 1.0, 2.0])
    assert rmse(v, v) == 0.0
    assert rmse(v + 2.0, v) == pytest.approx(2.0)
    assert rmse([0.0, 0.0, 3.0], v, mask=np.array([False, False, True])) == pytest.approx(1.0)
    np.testing.assert_allclose(rmse(np.stack([v, v + 1.0]), v), [0.0, 1.0])


def test_mean_se():
    m, se = mean_se(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(m, [2.0, 3.0])
    np.testing.assert_allclose(se, [1.0, 1.0])


def test_config_parses_and_quick_overrides():
    cfg = load_config(DEFAULT_CONFIG)
    assert all(isinstance(tc["n0"], (float, type(None))) for tc in td_configs(cfg["learning_curves"]["td"]))
    assert all(isinstance(e, float) for e in cfg["ridge"]["epsilons"])
    quick = load_config(DEFAULT_CONFIG, quick=True)
    assert quick["output_dir"] == "results/quick" and quick["n_seeds"] < cfg["n_seeds"]


def test_boyan_integration_lstd_beats_td():
    env, phi = BoyanChain(), boyan_features()
    v_true = true_values(env)
    episodes = generate_episodes(env, 100, seed=0)
    err_lstd = rmse(phi @ lstd_lambda(episodes, phi, 1.0, 0.4), v_true)
    err_td = rmse(phi @ td_lambda(episodes, phi, 1.0, 0.4, alpha=0.01), v_true)
    assert err_lstd < 1.0
    assert err_lstd < err_td


def _tiny_config(tmp_path):
    cfg = load_config(DEFAULT_CONFIG, quick=True)
    cfg["output_dir"] = str(tmp_path)
    cfg["n_seeds"] = 2
    cfg["learning_curves"]["n_trajectories"] = [1, 5, 20]
    cfg["learning_curves"]["lambdas"] = [0.0, 1.0]
    cfg["timing"].update(n_trajectories=5, dims=[4, 8], repeats=1)
    cfg["ridge"].update(n_seeds=2, n_trajectories=[1, 2], epsilons=[0.1, 1.0], td_alphas=[0.1])
    cfg["plots"].update(summary_n=[5, 20], lambda_plot_n=[5, 20])
    return cfg


def test_end_to_end_smoke(tmp_path):
    cfg = _tiny_config(tmp_path)
    raw, figs = tmp_path / "raw", tmp_path / "figures"
    raw.mkdir()

    lc = np.load(run_learning_curves(cfg, raw, workers=1))
    n_td = len(td_configs(cfg["learning_curves"]["td"]))
    assert lc["lstd_err"].shape == (2, 2, 3)
    assert lc["td_err"].shape == (2, 2, n_td, 3)

    timing = np.load(run_timing(cfg, raw))
    assert timing["times"].shape == (4, 2)

    for path in run_ridge(cfg, raw, workers=1):
        data = np.load(path)
        assert data["ridge_err"].shape == (2, 2, 2)

    saved = plot_results.plot_all(cfg, raw, figs)
    names = {p.name for p in saved}
    assert {"fig1_headline.png", "fig2_td_alpha.png", "fig3_lstd_lambda.png", "fig4_timing.png"} <= names
    assert all(p.exists() for p in saved)
