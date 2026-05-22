import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from pycmt.modules.generate_timeseries import (
    process_data,
    get_nice_step,
    generate_rainfall_plot,
    generate_tseries
)

# =========================================================================
# 1. TESTS UNITAIRES DES FONCTIONS DE TRAITEMENT ET LOGIQUE GRAPHIQUE
# =========================================================================

def test_process_data():
    fake_data = pd.Series([10.0, -999.0, 5.0, 15.0])
    daily, cumul = process_data(fake_data)
    
    assert daily.iloc[1] == 0.0
    assert cumul.iloc[0] == 10.0
    assert cumul.iloc[1] == 10.0  
    assert cumul.iloc[2] == 15.0  
    assert cumul.iloc[3] == 30.0  

def test_get_nice_step():
    imax, step = get_nice_step(45.0, is_cumul=False)
    assert imax == 46
    assert step in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    imax_cumul, step_cumul = get_nice_step(320.0, is_cumul=True)
    assert imax_cumul == 326
    assert step_cumul == 40

@patch("pycmt.modules.generate_timeseries.plt.savefig")
@patch("pycmt.modules.generate_timeseries.Path.mkdir")
def test_generate_rainfall_plot(mock_mkdir, mock_savefig):
    dates = pd.date_range(start="2026-01-01", periods=10)
    pcur_ts = xr.DataArray(np.random.uniform(0, 50, 10), coords=[("time", dates)])
    pclim_ts = xr.DataArray(np.random.uniform(0, 40, 10), coords=[("time", dates)])
    
    # Exécution sécurisée sans aucune interaction avec les dossiers physiques du disque
    generate_rainfall_plot(
        pcur_ts, pclim_ts, "Dakar", 14.7, -17.4, "001", 30, "Senegal", "arc2"
    )
    
    # Vérification que le fichier a bien tenté de se sauvegarder via Matplotlib
    mock_savefig.assert_called_once()

# =========================================================================
# 2. TEST D'ORCHESTRATION DU PIPELINE (GENERATE_TSERIES)
# =========================================================================

@patch("pycmt.modules.generate_timeseries.open_CtlDataset")
@patch("pycmt.modules.generate_timeseries.generate_rainfall_plot")
@patch("builtins.open", new_callable=mock_open, read_data="001 14.76 -17.44 Dakar\n")
def test_generate_tseries_pipeline(mock_file, mock_plot_func, mock_open_ctl):
    mock_dataset = MagicMock()
    mock_var = MagicMock()
    
    mock_var.sel.return_value.isel.return_value.load.return_value = MagicMock()
    mock_dataset.__getitem__.return_value = mock_var
    mock_open_ctl.return_value = mock_dataset

    generate_tseries("SEN", "Senegal", "rfe2")

    mock_file.assert_called_once()
    assert mock_open_ctl.call_count == 2  
    assert mock_plot_func.call_count == 6