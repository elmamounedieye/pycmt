from unittest.mock import patch
from pycmt.climonitor import (
    plot_precip_ts,
    generate_spp,
    generate_spi_,
    generated_vhi,
    generate_dashboard
)

# =========================================================================
# 1. TEST : PLOT_PRECIP_TS
# =========================================================================
@patch("pycmt.climonitor.generate_html_map")
@patch("pycmt.climonitor.generate_tseries")
@patch("pycmt.climonitor.plot_precip")
@patch("pycmt.climonitor.plot_pix_coordinates")
@patch("pycmt.climonitor.generate_pixel_arguments")
@patch("pycmt.climonitor.run_workflow")
@patch("pycmt.climonitor.download_arc2_data")
@patch("pycmt.climonitor.download_rfe2")
def test_plot_precip_ts(
    mock_rfe2,
    mock_arc2,
    mock_workflow,
    mock_pixel,
    mock_pix,
    mock_plot,
    mock_ts,
    mock_html
):
    plot_precip_ts("Senegal", "arc2", 0.25, 0.5)

    mock_workflow.assert_called_once()
    mock_pixel.assert_called_once()
    mock_pix.assert_called_once()
    mock_plot.assert_called_once()
    mock_ts.assert_called_once()
    mock_html.assert_called_once()

# =========================================================================
# 2. TEST : GENERATE_SPP
# =========================================================================
@patch("pycmt.climonitor.run_orchestrator_spp")
@patch("pycmt.climonitor.download_spp_noaa")
def test_generate_spp(mock_download_spp, mock_orchestrator_spp):
    generate_spp("Senegal", "SEN")

    assert mock_download_spp.call_count == 2
    assert mock_orchestrator_spp.call_count == 2

# =========================================================================
# 3. TEST : GENERATE_SPI_
# =========================================================================
@patch("pycmt.climonitor.generate_spi")
@patch("pycmt.climonitor.generate_soilmoisture")
@patch("pycmt.climonitor.generate_runoff")
@patch("pycmt.climonitor.download_spi")
@patch("pycmt.climonitor.download_xsm_data")
@patch("pycmt.climonitor.download_runoff_data")
def test_generate_spi_(
    mock_down_runoff,
    mock_down_xsm,
    mock_down_spi,
    mock_gen_runoff,
    mock_gen_soil,
    mock_gen_spi
):
    generate_spi_("Senegal", "SEN")

    mock_down_runoff.assert_called_once()
    mock_down_xsm.assert_called_once()
    assert mock_down_spi.call_count == 2
    mock_gen_runoff.assert_called_once()
    mock_gen_soil.assert_called_once()
    assert mock_gen_spi.call_count == 2

# =========================================================================
# 4. TEST : GENERATED_VHI
# =========================================================================
@patch("pycmt.climonitor.do_vhi")
@patch("pycmt.climonitor.run_retrieval_vhi")
def test_generated_vhi(mock_retrieval_vhi, mock_do_vhi):
    generated_vhi("Senegal", "SEN")

    mock_retrieval_vhi.assert_called_once()
    mock_do_vhi.assert_called_once()

# =========================================================================
# 5. TEST : GENERATE_DASHBOARD
# =========================================================================
@patch("pycmt.climonitor.build_country_dashboard")
def test_generate_dashboard(mock_build_dashboard):
    generate_dashboard("Senegal", "arc2")

    mock_build_dashboard.assert_called_once_with("Senegal", "arc2")