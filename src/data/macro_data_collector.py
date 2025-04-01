"""
Module pour la collecte des données macroéconomiques via l'API FRED.
"""

import os
import pandas as pd
from fredapi import Fred
from datetime import datetime
import logging
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()


class MacroDataCollector:
    """
    Classe pour collecter et prétraiter les données macroéconomiques.
    """

    def __init__(self, api_key=None):
        """
        Initialise le collecteur de données macroéconomiques.

        Args:
            api_key (str, optional): Clé API FRED. Si None, la clé est récupérée 
                                    depuis les variables d'environnement.
        """
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
            if api_key is None:
                raise ValueError("Aucune clé API FRED fournie ou trouvée dans les variables d'environnement.")
        
        self.fred = Fred(api_key=api_key)
        logger.info("MacroDataCollector initialisé avec succès.")
        
        # Dictionnaire des séries économiques importantes avec leur description
        self.macro_series = {
            # Indicateurs de croissance
            'GDPC1': 'PIB réel',
            'INDPRO': 'Production industrielle',
            'PAYEMS': 'Emploi non-agricole',
            'UNRATE': 'Taux de chômage',
            
            # Indicateurs d'inflation
            'CPIAUCSL': 'Indice des prix à la consommation',
            'PCEPI': 'Indice des prix des dépenses personnelles de consommation',
            'PPIFIS': 'Indice des prix à la production',
            
            # Indicateurs de politique monétaire
            'FEDFUNDS': 'Taux des fonds fédéraux',
            'DFF': 'Taux effectif des fonds fédéraux',
            'T10Y2Y': 'Spread de taux 10 ans - 2 ans',
            'T10YIE': 'Anticipations d'inflation à 10 ans',
            
            # Indicateurs de confiance
            'UMCSENT': 'Indice de confiance des consommateurs de l'Université du Michigan',
            'CSCICP03USM665S': 'Indice de confiance des consommateurs OCDE',
            
            # Indicateurs de marché
            'VIXCLS': 'Indice de volatilité VIX',
            'BAMLH0A0HYM2': 'Spread de crédit obligations à haut rendement',
            'DTWEXBGS': 'Indice du dollar américain',
        }
    
    def get_series(self, series_id, start_date=None, end_date=None, frequency=None):
        """
        Récupère une série de données spécifique depuis FRED.

        Args:
            series_id (str): Identifiant de la série FRED.
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'.
            frequency (str, optional): Fréquence des données ('d', 'm', 'q', 'a').

        Returns:
            pd.Series: Série temporelle des données.
        """
        logger.info(f"Récupération de la série {series_id}")
        try:
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date, 
                observation_end=end_date,
                frequency=frequency
            )
            logger.info(f"Série {series_id} récupérée avec succès. {len(data)} observations.")
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la série {series_id}: {e}")
            return None
    
    def get_all_series(self, start_date=None, end_date=None, frequency='m'):
        """
        Récupère toutes les séries définies dans self.macro_series.

        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'.
            frequency (str, optional): Fréquence des données ('d', 'm', 'q', 'a').

        Returns:
            pd.DataFrame: DataFrame contenant toutes les séries temporelles.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Par défaut, récupérer les données sur 20 ans
            start_date = (datetime.now().year - 20)
            start_date = f"{start_date}-01-01"
        
        logger.info(f"Récupération de toutes les séries du {start_date} au {end_date} avec fréquence {frequency}")
        
        all_data = {}
        for series_id, description in self.macro_series.items():
            series = self.get_series(series_id, start_date, end_date, frequency)
            if series is not None:
                all_data[series_id] = series
        
        # Combinaison des séries en un DataFrame
        df = pd.DataFrame(all_data)
        
        # Ajout des noms de séries comme métadonnées
        df.attrs['series_descriptions'] = {k: v for k, v in self.macro_series.items() if k in df.columns}
        
        return df
    
    def calculate_yoy_changes(self, df):
        """
        Calcule les variations en glissement annuel pour chaque série.

        Args:
            df (pd.DataFrame): DataFrame contenant les séries temporelles.

        Returns:
            pd.DataFrame: DataFrame contenant les variations en glissement annuel.
        """
        yoy_df = df.pct_change(periods=12) * 100
        yoy_df.columns = [f"{col}_YOY" for col in yoy_df.columns]
        return yoy_df

    def calculate_mom_changes(self, df):
        """
        Calcule les variations en glissement mensuel pour chaque série.

        Args:
            df (pd.DataFrame): DataFrame contenant les séries temporelles.

        Returns:
            pd.DataFrame: DataFrame contenant les variations en glissement mensuel.
        """
        mom_df = df.pct_change() * 100
        mom_df.columns = [f"{col}_MOM" for col in mom_df.columns]
        return mom_df
    
    def preprocess_data(self, df):
        """
        Prétraite les données en calculant les variations et en traitant les valeurs manquantes.

        Args:
            df (pd.DataFrame): DataFrame contenant les séries temporelles brutes.

        Returns:
            pd.DataFrame: DataFrame prétraité avec les séries originales et les variations.
        """
        # Calcul des variations
        yoy_changes = self.calculate_yoy_changes(df)
        mom_changes = self.calculate_mom_changes(df)
        
        # Combinaison des données
        combined_df = pd.concat([df, yoy_changes, mom_changes], axis=1)
        
        # Traitement des valeurs manquantes
        # Pour simplifier, on utilise une méthode d'imputation forward fill
        combined_df = combined_df.fillna(method='ffill')
        
        return combined_df
    
    def save_data(self, df, file_path):
        """
        Sauvegarde les données dans un fichier CSV.

        Args:
            df (pd.DataFrame): DataFrame à sauvegarder.
            file_path (str): Chemin du fichier de sortie.

        Returns:
            bool: True si la sauvegarde a réussi, False sinon.
        """
        try:
            # Création du répertoire parent si nécessaire
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Sauvegarde du DataFrame
            df.to_csv(file_path)
            logger.info(f"Données sauvegardées avec succès dans {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {e}")
            return False


if __name__ == "__main__":
    # Exemple d'utilisation
    collector = MacroDataCollector()
    
    # Récupération de toutes les séries macroéconomiques
    macro_data = collector.get_all_series(start_date="2000-01-01", frequency='m')
    
    # Prétraitement des données
    processed_data = collector.preprocess_data(macro_data)
    
    # Sauvegarde des données
    collector.save_data(
        processed_data, 
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                     "data", "processed", "macro_data.csv")
    )
