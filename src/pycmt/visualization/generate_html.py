import os
from pathlib import Path
from jinja2 import Template
import shutil
import json
import webbrowser


def generate_html_map(country_name, rndta):
    country_display = country_name.capitalize()
    pix_coord_file = Path.cwd().resolve().parents[3] / "data" / f"pixelargs_{country_name}.txt"
    formatted_areas = Path.cwd().resolve().parents[3] / "data" /f"formatted_areas_{country_name}.json"
    # 2. Lecture du fichier de coordonnées
    if not os.path.exists(pix_coord_file):
        print(f"Erreur : Le fichier {pix_coord_file} est introuvable.")
        return

    with open(pix_coord_file, 'r') as f:
        lines = f.readlines()

    with open(formatted_areas, "r", encoding="utf-8") as f:
        formatted_areas = json.load(f)
    period = [7, 10, 30, 60, 90, 180]

    npixtot = len(lines)
    print(f"========== The total number of points is : {npixtot}")
    for periods in period:
        # 3. Génération des balises <area>
        areas_html = []
        for line in lines:
            parts = line.split()
            if len(parts) < 10: continue # Sécurité si ligne incomplète
            
            pixnum = parts[0]
            xpix = float(parts[7])
            ypix = float(parts[8])
            gridtype = parts[9]

            if gridtype == "grid":
                area = f'<area href="{pixnum}_{periods}.png" shape="circle" coords="{xpix}, {ypix}, 5" ALT="Station Location Marker">'
            else:
                # Pour les stations, on ajoute le nom en bulle d'aide (TITLE)
                area = f'<area href="{pixnum}_{periods}.png" shape="circle" coords="{xpix}, {ypix}, 5" TITLE="{gridtype}">'
            
            areas_html.append(area)

        # 4. Construction du fichier HTML complet
        map_name = f"{country_name}_{rndta}_{periods}"
        #formatted_areas = "\n                ".join(areas_html)#(areas_html)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                /* Force l'image à s'adapter à la largeur de la table ou de l'écran */
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                }}
                body {{ font-family: Arial, sans-serif; }}
            </style>
        </head>
        <body>

        <table border="0" width="80%" bgcolor="#FFFFFF" align="center" cellpadding="5">
            <tr>
                <td align="center">
                    <h3>Time Series at Pixel Locations</h3>
                </td>
            </tr>
        </table>

        <table border="0" width="90%" bgcolor="#FFFFFF" align="center" cellpadding="5">
            <tr>
                <td align="center">
                    <map name="{map_name}">
                        {formatted_areas[str(periods)]}
                    </map>
                    <img src="{country_name}_grid.png" 
                        alt="{country_display}" 
                        border="1" 
                        usemap="#{map_name}">
                </td>
            </tr>
        </table>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/image-map-resizer/1.0.10/js/imageMapResizer.min.js"></script>
        <script>
            imageMapResizer();
        </script>

        </body>
        </html>
        """

        # 5. Détermination du chemin de sortie et sauvegarde
        if periods == "rev":
            output_dir = Path("..") / ".." / "data" / "ts_maps" / f"{country_name}"/ rndta
            filename = f"pix_{periods}_body.html"
        else:
            output_dir = Path("..") / ".." / "data" / "ts_maps" / f"{country_name}" / rndta
            filename = f"pix_{periods}day_body.html"

        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Fichier généré avec succès : {full_path}")
        

################ Generate country dashboard ##################

def build_country_dashboard(country_name, rndta):
    # --- CONFIGURATION DES CHEMINS ---
    # Utilisation de Path pour la robustesse Windows/Linux
    data_root = Path.cwd().resolve().parents[3] / "data"
    template_name = data_root / "template_pycmt_index.html"
    output_name = data_root / f"{country_name}_index.html"
    
    paths = {
        "vhi": data_root / "vhi" / "vhi_maps" / country_name,
        "spatial": data_root / "spatial_maps" / country_name / rndta,
        "spi": data_root / "spi" / "spi_maps" / country_name,
        "spp": data_root / "spp" / "spp_maps" / country_name,
        "ts": data_root / "ts_maps" / country_name / rndta
    }

    # Structure de données envoyée au HTML (Jinja2)
    data = {
        "country": country_name,
        "vhi_weeks": [],
        "stations": [],
        "data" : rndta
    }

    # 1. Détection VHI (Extraction des numéros de semaines)
    if paths["vhi"].exists():
        files = list(paths["vhi"].glob(f"{country_name}_vhi*.png"))
        if files:
            # On extrait le nombre après 'vhi' dans le nom du fichier
            data["vhi_weeks"] = sorted([int(f.stem.split('vhi')[-1]) for f in files])
            print(f"✅ {len(data['vhi_weeks'])} semaines VHI détectées.")

    # 2. Détection Stations (Points)
    # Important pour générer la grille "Time Series" dans le HTML
    # Détection Stations (Fichiers HTML interactifs)
    if paths["ts"].exists():
        # On cherche les fichiers type 'pix_7day_body.html'
        ts_html_files = list(paths["ts"].glob("pix_*_body.html"))
        
        # On extrait la période (ex: 7, 10, 30) pour créer les colonnes
        periods = []
        for f in ts_html_files:
            # f.stem est 'pix_7day_body' -> on split pour avoir '7'
            p = f.stem.split('_')[1].replace('day', '')
            periods.append(p)
        
        data["ts_periods"] = sorted(list(set(periods)), key=int)
        print(f"✅ {len(data['ts_periods'])} périodes HTML détectées : {data['ts_periods']}")

    # --- GÉNÉRATION DU FICHIER HTML ---
    print(f"🛠️  Génération du dashboard : {output_name}")
    
    try:
        if not template_name.exists():
            raise FileNotFoundError(f"Le template est introuvable à l'adresse : {template_name}")

        # Lecture du template
        with open(template_name, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        tmpl = Template(template_content)
        
        # Injection des variables (country, vhi_weeks, stations)
        html_rendered = tmpl.render(data)

        # Sauvegarde
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write(html_rendered)
            
        print(f"🚀 Dashboard {country_name} finalisé.")

        # --- OUVERTURE AUTOMATIQUE ---
        abs_path = os.path.abspath(output_name)
        # Compatibilité Windows pour l'URL file
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        
        webbrowser.open(file_url)
        print(f"🌐 Ouverture dans le navigateur effectuée.")

    except Exception as e:
        print(f"❌ Erreur lors de la génération : {str(e)}")
