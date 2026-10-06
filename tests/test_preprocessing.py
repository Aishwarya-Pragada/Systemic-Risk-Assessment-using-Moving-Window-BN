import numpy as np
import pandas as pd
import pytest

from bnsr.config import load_config
from bnsr.data.preprocessing import run_preprocessing
from bnsr.data.universe import UNIVERSE, SECTOR_COUNTS, symbols


@pytest.fixture(scope="module")
def out():
    return run_preprocessing(load_config(), strict=True)


def test_universe():
    assert len(UNIVERSE) == 60 and len(set(symbols())) == 60
    counts = pd.Series([s for *_, s in UNIVERSE]).value_counts().to_dict()
    assert counts == SECTOR_COUNTS


def test_calendar_and_shapes(out):
    assert out["prices"].shape == (3404, 60)
    assert out["returns"].shape == (3403, 60)
    assert out["X"].shape == (3403, 60)
    assert out["returns"].index[0] == pd.Timestamp("2008-01-03")
    assert out["returns"].index[-1] == pd.Timestamp("2021-11-04")
    assert out["hsi"].shape[0] == 3404 and out["hsi"]["t"].iloc[0] == 0


def test_binary_indicator_rule(out):
    R, X = out["returns"], out["X"]
    valid = R.notna()
    assert set(np.unique(X.stack().dropna().astype(int))) <= {0, 1}
    assert ((X.astype(float) == 1) == (R >= 0))[valid].all().all()
    assert X.isna().equals(R.isna())


def test_hsi_series(out):
    h = out["hsi"].iloc[1:]
    assert np.allclose(h["abs_ret"], h["ret"].abs())
    assert np.allclose(h["loss"], -np.minimum(h["ret"], 0))
    assert (h["loss"] >= 0).all()
    assert np.allclose(h["ret"], np.log(out["hsi"]["close"]).diff().iloc[1:])


def test_node_counts_46_to_60(out):
    v = out["node_counts"]
    assert v.iloc[0] == 46 and v.iloc[-1] == 60
    assert (v.diff().dropna() >= 0).all()


def test_window_helper(out):
    from bnsr.data.dataset import Dataset
    ds = Dataset(X=out["X"], returns=out["returns"], prices=out["prices"],
                 hsi=out["hsi"], universe=out["universe"])
    w30 = ds.window(30, 30)
    assert w30.shape == (30, 46)
    assert ds.window(ds.T, 30).shape == (30, 60)
    assert ds.window(30, 30).index[0] == pd.Timestamp("2008-01-03")
    assert ds.date_of(0) == pd.Timestamp("2008-01-02")
    with pytest.raises(ValueError):
        ds.window(10, 30)
