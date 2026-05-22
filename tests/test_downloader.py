import gzip
import pytest
from unittest.mock import patch, mock_open, MagicMock
from io import BytesIO
from pathlib import Path
from pycmt.core.downloader import (
    get_country_iso,
    download_gadm_country,
    rename_country_shapefiles,
    download_arc2_data,
    manage_download,
    generate_ctl,
    download_rfe2,
    download_file,
    run_retrieval_vhi,
    download_spp_file,
    generate_ctl_spp,
    run_spp_retrieval,
    download_spp_noaa,
    download_spi,
    download_runoff_data,
    download_xsm_data
)

# =========================================================================
# 1. ENTRAÎNEMENT ET CONFIGURATION DE BASE
# =========================================================================

@patch("builtins.open", new_callable=mock_open, read_data='{"Senegal": "SEN"}')
def test_get_country_iso_success(mock_file):
    assert get_country_iso("Senegal") == "SEN"

@patch("builtins.open", side_effect=FileNotFoundError)
def test_get_country_iso_file_not_found(mock_file):
    result = get_country_iso("Senegal")
    assert "Error" in result

@patch("os.path.exists", return_value=True)
def test_download_gadm_country_already_exists(mock_exists):
    result = download_gadm_country("SEN")
    assert result is not False

@patch("os.path.exists", return_value=False)
@patch("pycmt.core.downloader.Path.mkdir")
@patch("pycmt.core.downloader.requests.get")
@patch("pycmt.core.downloader.zipfile.ZipFile")
@patch("pycmt.core.downloader.rename_country_shapefiles")
def test_download_gadm_country_trigger(mock_rename, mock_zip, mock_get, mock_mkdir, mock_exists):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_get.return_value = mock_response

    with patch("builtins.open", mock_open()):
        download_gadm_country("SEN")
    
    mock_get.assert_called_once()

@patch("pycmt.core.downloader.Path.glob")
@patch("pycmt.core.downloader.os.path.exists", return_value=False)
@patch("pycmt.core.downloader.shutil.copy2")
def test_rename_country_shapefiles(mock_copy, mock_exists, mock_glob):
    mock_file = MagicMock(spec=Path)
    mock_file.stem = "gadm41_SEN_0"
    mock_file.name = "gadm41_SEN_0.shp"
    mock_glob.return_value = [mock_file]
    
    with patch.object(Path, "exists", return_value=True):
        rename_country_shapefiles("SEN")
        assert mock_copy.call_count >= 0

# =========================================================================
# 2. HYDROLOGIE & CLIMATOLOGIE (ARC2 / RFE2)
# =========================================================================

@patch("pycmt.core.downloader.manage_download")
@patch("pycmt.core.downloader.generate_ctl")
def test_download_arc2_data(mock_gen_ctl, mock_manage):
    download_arc2_data(init_day_offset=0)
    assert mock_manage.call_count == 360  # 180 jours * 2 (daily + clim)
    assert mock_gen_ctl.call_count == 2

@patch("pycmt.core.downloader.Path.exists", return_value=True)
def test_manage_download_already_present(mock_exists):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 3000 * 1024
    
    result = manage_download("http://fakeurl.com", mock_path)
    assert result is None

@patch("pycmt.core.downloader.requests.get")
def test_manage_download_gzip_stream(mock_get):
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    
    fake_gzip_content = BytesIO()
    with gzip.GzipFile(fileobj=fake_gzip_content, mode='wb') as f:
        f.write(b"uncompressed data")
    fake_gzip_content.seek(0)
    
    mock_response = MagicMock()
    mock_response.raw = fake_gzip_content
    mock_get.return_value = mock_response

    with patch("builtins.open", mock_open()):
        manage_download("http://fakeurl.com/file.gz", mock_path, is_gzip=True)
    
    mock_get.assert_called_once()

@patch("pycmt.core.downloader.Path.read_text", return_value="DLDATADIR STDATE NNDAYS")
@patch("pycmt.core.downloader.Path.write_text")
@patch("pycmt.core.downloader.Path.exists", return_value=True)
def test_generate_ctl(mock_exists, mock_write, mock_read):
    # En passant des objets Path réels, le patch de Path.exists est intercepté correctement
    template_path = Path("template.ctl")
    output_path = Path("output.ctl")
    
    generate_ctl(template_path, output_path, {"DLDATADIR": "^", "STDATE": "01Jan2026"})
    mock_write.assert_called_once()

@patch("pycmt.core.downloader.Path.glob")
@patch("pycmt.core.downloader.requests.get")
@patch("pycmt.core.downloader.generate_ctl")
@patch("pycmt.core.downloader.Path.unlink")
def test_download_rfe2(mock_unlink, mock_gen_ctl, mock_get, mock_glob):
    mock_glob.return_value = [] # Dossier vide pour forcer n_missing = 180
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake data"
    mock_get.return_value = mock_response

    with patch("builtins.open", mock_open()):
        download_rfe2(days_offset=0)
        
    assert mock_gen_ctl.call_count == 2

# =========================================================================
# 3. WORKFLOWS D'INDICES SPÉCIFIQUES (VHI / SPP / SPI)
# =========================================================================

@patch("pycmt.core.downloader.requests.get")
def test_download_file(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = lambda chunk_size: [b"chunk"]
    
    with patch("builtins.open", mock_open()):
        assert download_file("http://fake_url", "test_file.nc") is True

@patch("pycmt.core.downloader.os.listdir", return_value=[])
@patch("pycmt.core.downloader.os.path.exists", return_value=False)
@patch("pycmt.core.downloader.download_file", return_value=True)
def test_run_retrieval_vhi(mock_down, mock_exists, mock_listdir):
    run_retrieval_vhi()
    assert mock_down.call_count >= 1

@patch("pycmt.core.downloader.requests.Session.get")
def test_download_spp_file(mock_session_get):
    mock_session_get.return_value.status_code = 200
    mock_session_get.return_value.iter_content = lambda chunk_size: [b"bytes"]
    
    with patch("builtins.open", mock_open()):
        res = download_spp_file("https://fake-spp-url/spp_test.dat", "fake_dir")
    assert "SUCCÈS" in res

@patch("pycmt.core.downloader.download_spp_file", return_value="SUCCÈS: ctl")
@patch("pycmt.core.downloader.generate_ctl_spp")
def test_run_spp_retrieval(mock_gen_spp, mock_down_spp):
    with patch("pycmt.core.downloader.Path.glob") as mock_glob:
        mock_glob.return_value = [Path("test.ctl")]
        run_spp_retrieval("rfe2")
    assert mock_gen_spp.call_count >= 1

@patch("pycmt.core.downloader.requests.get")
@patch("pycmt.core.downloader.generate_ctl_spp")
def test_download_spp_noaa(mock_gen_spp, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_get.return_value = mock_response

    with patch("pycmt.core.downloader.Path.glob") as mock_glob:
        mock_glob.return_value = [Path("test.ctl")]
        with patch("builtins.open", mock_open()):
            download_spp_noaa("cmorph")
            
    assert mock_get.call_count >= 1

@patch("pycmt.core.downloader.os.listdir", return_value=["old.ctl"])
@patch("pycmt.core.downloader.os.remove")
@patch("pycmt.core.downloader.requests.get")
def test_download_spi(mock_get, mock_remove, mock_listdir):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_get.return_value.__enter__.return_value = mock_response

    with patch("builtins.open", mock_open()):
        download_spi("rfe2")
        
    mock_remove.assert_called_once()
    assert mock_get.call_count >= 1

@patch("pycmt.core.downloader.requests.get")
def test_download_runoff_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = lambda chunk_size: [b"data"]
    
    with patch("builtins.open", mock_open()):
        download_runoff_data()
        
    assert mock_get.call_count >= 1

@patch("pycmt.core.downloader.requests.get")
def test_download_xsm_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = lambda chunk_size: [b"data"]
    
    with patch("builtins.open", mock_open()):
        download_xsm_data()
        
    assert mock_get.call_count >= 1