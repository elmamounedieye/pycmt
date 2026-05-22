import pytest
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import pandas as pd
from pycmt.visualization.generate_html import generate_html_map, build_country_dashboard

# Configuration de la structure attendue pour simuler pixelargs
FAKE_PIXELARGS_CONTENT = (
    "1 14.5 -16.5 14.25 14.75 -16.75 -16.25 150.0 200.0 grid\n"
    "2 14.7 -16.8 14.50 15.00 -17.00 -16.50 160.0 210.0 Dakar\n"
    "short_line_invalid_to_test_security_skip\n"
)

# =========================================================================
# 1. TESTS POUR LA FONCTION : GENERATE_HTML_MAP
# =========================================================================

@patch("pycmt.visualization.generate_html.os.path.exists")
@patch("pycmt.visualization.generate_html.os.makedirs")
@patch("pycmt.visualization.generate_html.json.load")
def test_generate_html_map_success(mock_json_load, mock_makedirs, mock_exists):
    # Simulation de l'existence du fichier d'arguments de pixels
    mock_exists.return_value = True
    
    # Simulation du dictionnaire structuré contenant les balises <area> pré-générées
    mock_json_load.return_value = {
        "7": "<area href='1_7.png'>",
        "10": "<area href='1_10.png'>",
        "30": "<area href='1_30.png'>",
        "60": "<area href='1_60.png'>",
        "90": "<area href='1_90.png'>",
        "180": "<area href='1_180.png'>"
    }
    
    m_open = mock_open(read_data=FAKE_PIXELARGS_CONTENT)
    
    with patch("builtins.open", m_open):
        generate_html_map("Senegal", "arc2")
        
    # Vérifications de la création des répertoires de séries temporelles
    mock_makedirs.assert_called()
    
    # open() doit être appelé pour lire pixelargs, lire formatted_areas JSON, 
    # puis écrire les 6 fichiers de corps HTML (un par période) = 8 ouvertures au total
    assert m_open.call_count == 8


@patch("pycmt.visualization.generate_html.os.path.exists", return_value=False)
@patch("pycmt.visualization.generate_html.print")
def test_generate_html_map_missing_file(mock_print, mock_exists):
    # Test de la sécurité si le fichier pixelargs est introuvable sur le disque
    generate_html_map("Senegal", "arc2")
    
    # Vérification que le message d'erreur d'absence a bien été imprimé
    assert any("Erreur : Le fichier" in call[0][0] for call in mock_print.call_args_list)


# =========================================================================
# 2. TESTS POUR LA FONCTION : BUILD_COUNTRY_DASHBOARD
# =========================================================================

@patch("pycmt.visualization.generate_html.Template")
@patch("pycmt.visualization.generate_html.webbrowser.open")
def test_build_country_dashboard_success(mock_webbrowser, mock_template):
    # On simule l'existence des dossiers d'emprises cartographiques
    # Pour parer aux restrictions de lecture seule, on utilise un side_effect sur le type Path
    def side_effect_exists(self):
        return True
        
    # Simulation des listes de fichiers renvoyées par le scanneur glob() de Path
    def side_effect_glob(self, pattern):
        if "vhi*.png" in pattern:
            f1, f2 = MagicMock(spec=Path), MagicMock(spec=Path)
            f1.stem = "Senegal_vhi20"
            f2.stem = "Senegal_vhi21"
            return [f1, f2]
        elif "pix_*_body.html" in pattern:
            f3, f4 = MagicMock(spec=Path), MagicMock(spec=Path)
            f3.stem = "pix_7day_body"
            f4.stem = "pix_30day_body"
            return [f3, f4]
        return []

    # Mock de l'instance du moteur Jinja2 Template
    mock_template_instance = MagicMock()
    mock_template_instance.render.return_value = "<html>Rendered Dashboard Content</html>"
    mock_template.return_value = mock_template_instance
    
    m_open = mock_open(read_data="<h1>Dashboard Template {{ country }}</h1>")
    
    # Application croisée des patchs comportementaux sur la classe Path de pathlib
    with patch.object(Path, "exists", side_effect_exists), \
         patch.object(Path, "glob", side_effect_glob), \
         patch("builtins.open", m_open):
         
        build_country_dashboard("Senegal", "arc2")
        
    # Vérifications de la compilation du dashboard
    mock_template.assert_called_once_with("<h1>Dashboard Template {{ country }}</h1>")
    mock_template_instance.render.assert_called_once()
    
    # L'appel au navigateur Web doit être intercepté sans planter
    mock_webbrowser.assert_called_once()
    # open() doit s'ouvrir une fois pour charger le template brut, et une fois pour exporter l'index final
    assert m_open.call_count == 2


@patch("pycmt.visualization.generate_html.print")
def test_build_country_dashboard_template_missing(mock_print):
    # Test d'anticipation : Lever une exception propre si le template HTML d'origine n'existe pas
    def side_effect_exists(self):
        if "template_pycmt_index.html" in str(self):
            return False
        return True

    with patch.object(Path, "exists", side_effect_exists):
        build_country_dashboard("Senegal", "arc2")
        
    # Validation que le bloc try/except capture bien la panne et log l'erreur avec la croix rouge
    assert any("❌ Erreur lors de la génération" in call[0][0] for call in mock_print.call_args_list)