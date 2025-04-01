"""
Module pour la collecte des données sectorielles via Yahoo Finance.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SectorDataCollector:
    """
    Classe pour collecter et prétraiter les données sectorielles.
    """

    def __init__(self):
        """
        Initialise le collecteur de données sectorielles.
        """
        # Définition des ETFs sectoriels
        self.sector_etfs = {
            'XLY': 'Consommation discrétionnaire',
            'XLP': 'Consommation de base',
            'XLE': 'Énergie',
            'XLF': 'Finance',
            'XLV': 'Santé',
            'XLI': 'Industrie',
            'XLB': 'Matériaux',
            'XLK': 'Technologie',
            'XLU': 'Services publics',
            'XLRE': 'Immobilier',
            'XLC': 'Services de communication'
        }
        
        # ETFs de référence pour le marché
        self.market_etfs = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ-100',
            'IWM': 'Russell 2000',
            'VGK': 'Europe',
            'EWJ': 'Japon',
            'MCHI': 'Chine',
            'EEM': 'Marchés émergents'
        }
        
        # ETFs de style d'investissement
        self.style_etfs = {
            'IVW': 'Croissance grandes capitalisations',
            'IVE': 'Valeur grandes capitalisations',
            'IJK': 'Croissance moyennes capitalisations',
            'IJJ': 'Valeur moyennes capitalisations',
            'IWO': 'Croissance petites capitalisations',
            'IWN': 'Valeur petites capitalisations'
        }
        
        # ETFs obligataires et de matières premières
        self.bond_commodity_etfs = {
            'TLT': 'Obligations long terme',
            'IEF': 'Obligations moyen terme',
            'SHY': 'Obligations court terme',
            'LQD': 'Obligations d\'entreprise investment grade',
            'HYG': 'Obligations d\'entreprise high yield',
            'GLD': 'Or',
            'SLV': 'Argent',
            'USO': 'Pétrole',
            'DBC': 'Matières premières diversifiées'
        }
        
        # Combinaison de tous les ETFs
        self.all_etfs = {}
        self.all_etfs.update(self.sector_etfs)
        self.all_etfs.update(self.market_etfs)
        self.all_etfs.update(self.style_etfs)
        self.all_etfs.update(self.bond_commodity_etfs)
        
        logger.info("SectorDataCollector initialisé avec succès.")
    
    def get_etf_data(self, ticker, start_date=None, end_date=None):
        """
        Récupère les données historiques d'un ETF.

        Args:
            ticker (str): Symbole de l'ETF.
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: DataFrame contenant les données de l'ETF.
        """
        logger.info(f"Récupération des données pour l'ETF {ticker}")
        try:
            etf = yf.Ticker(ticker)
            data = etf.history(start=start_date, end=end_date)
            logger.info(f"Données de l'ETF {ticker} récupérées avec succès. {len(data)} observations.")
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour l'ETF {ticker}: {e}")
            return None
    
    def get_all_etf_data(self, start_date=None, end_date=None):
        """
        Récupère les données historiques de tous les ETFs.

        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            dict: Dictionnaire contenant les DataFrames pour chaque ETF.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Par défaut, récupérer les données sur 20 ans
            start_date = (datetime.now().year - 20)
            start_date = f"{start_date}-01-01"
        
        logger.info(f"Récupération des données pour tous les ETFs du {start_date} au {end_date}")
        
        tickers = list(self.all_etfs.keys())
        
        try:
            # Récupération des données pour tous les tickers en une seule requête
            data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
            
            # Transformation en dictionnaire de DataFrames
            etf_data = {}
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    etf_df = data[ticker].copy()
                    etf_df.columns = [col.lower() for col in etf_df.columns]
                    etf_data[ticker] = etf_df
                    logger.info(f"Données pour {ticker} récupérées: {len(etf_df)} observations")
                else:
                    logger.warning(f"Pas de données pour {ticker}")
            
            return etf_data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour tous les ETFs: {e}")
            return {}
    
    def extract_prices(self, etf_data):
        """
        Extrait les prix de clôture ajustés de tous les ETFs.

        Args:
            etf_data (dict): Dictionnaire contenant les DataFrames pour chaque ETF.

        Returns:
            pd.DataFrame: DataFrame contenant les prix de clôture ajustés pour tous les ETFs.
        """
        prices = {}
        
        for ticker, df in etf_data.items():
            if 'adj close' in df.columns:
                prices[ticker] = df['adj close']
            elif 'close' in df.columns:
                prices[ticker] = df['close']
        
        return pd.DataFrame(prices)
    
    def calculate_returns(self, price_df, periods=None):
        """
        Calcule les rendements des ETFs sur différentes périodes.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix.
            periods (dict, optional): Dictionnaire des périodes à calculer.

        Returns:
            dict: Dictionnaire contenant les DataFrames de rendements.
        """
        if periods is None:
            periods = {
                'daily': 1,
                'weekly': 5,
                'monthly': 21,
                '3_months': 63,
                '6_months': 126,
                '12_months': 252
            }
        
        returns = {}
        
        # Pour chaque période, calculer les rendements
        for period_name, period_days in periods.items():
            period_returns = price_df.pct_change(period_days) * 100
            period_returns.columns = [f"{col}_{period_name}" for col in period_returns.columns]
            returns[period_name] = period_returns
        
        return returns
    
    def calculate_relative_strength(self, price_df, benchmark='SPY', periods=None):
        """
        Calcule la force relative des ETFs par rapport à un benchmark.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix.
            benchmark (str, optional): Ticker du benchmark. Par défaut 'SPY'.
            periods (dict, optional): Dictionnaire des périodes à calculer.

        Returns:
            dict: Dictionnaire contenant les DataFrames de force relative.
        """
        if benchmark not in price_df.columns:
            logger.error(f"Le benchmark {benchmark} n'est pas dans les données de prix.")
            return {}
        
        if periods is None:
            periods = {
                '1_month': 21,
                '3_months': 63,
                '6_months': 126,
                '12_months': 252
            }
        
        relative_strength = {}
        
        # Pour chaque période, calculer la force relative
        for period_name, period_days in periods.items():
            # Calcul des rendements
            returns = price_df.pct_change(period_days)
            
            # Calcul de la force relative (rendement ETF / rendement benchmark)
            benchmark_returns = returns[benchmark]
            rs = pd.DataFrame({
                f"{col}_RS_{period_name}": returns[col] / benchmark_returns
                for col in price_df.columns if col != benchmark
            })
            
            relative_strength[period_name] = rs
        
        return relative_strength
    
    def calculate_volatility(self, price_df, windows=None):
        """
        Calcule la volatilité des ETFs sur différentes fenêtres.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix.
            windows (dict, optional): Dictionnaire des fenêtres à calculer.

        Returns:
            dict: Dictionnaire contenant les DataFrames de volatilité.
        """
        if windows is None:
            windows = {
                '1_month': 21,
                '3_months': 63,
                '6_months': 126
            }
        
        volatility = {}
        
        # Calcul des rendements journaliers
        daily_returns = price_df.pct_change()
        
        # Pour chaque fenêtre, calculer la volatilité annualisée
        for window_name, window_days in windows.items():
            vol = daily_returns.rolling(window=window_days).std() * np.sqrt(252) * 100  # Annualisée et en pourcentage
            vol.columns = [f"{col}_VOL_{window_name}" for col in vol.columns]
            volatility[window_name] = vol
        
        return volatility
    
    def preprocess_data(self, etf_data):
        """
        Prétraite les données en calculant rendements, force relative et volatilité.

        Args:
            etf_data (dict): Dictionnaire contenant les DataFrames pour chaque ETF.

        Returns:
            pd.DataFrame: DataFrame prétraité avec toutes les métriques.
        """
        # Extraction des prix
        price_df = self.extract_prices(etf_data)
        
        # Calcul des rendements
        returns = self.calculate_returns(price_df)
        
        # Calcul de la force relative
        relative_strength = self.calculate_relative_strength(price_df)
        
        # Calcul de la volatilité
        volatility = self.calculate_volatility(price_df)
        
        # Initialisation du DataFrame de résultats avec les prix
        result_dfs = [price_df]
        
        # Ajout des rendements
        for period, df in returns.items():
            result_dfs.append(df)
        
        # Ajout de la force relative
        for period, df in relative_strength.items():
            result_dfs.append(df)
        
        # Ajout de la volatilité
        for window, df in volatility.items():
            result_dfs.append(df)
        
        # Combinaison de tous les DataFrames
        result = pd.concat(result_dfs, axis=1)
        
        # Ajout des noms d'ETFs comme métadonnées
        result.attrs['etf_descriptions'] = self.all_etfs
        result.attrs['sector_etfs'] = self.sector_etfs
        result.attrs['market_etfs'] = self.market_etfs
        result.attrs['style_etfs'] = self.style_etfs
        result.attrs['bond_commodity_etfs'] = self.bond_commodity_etfs
        
        return result
    
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
    collector = SectorDataCollector()
    
    # Récupération des données de tous les ETFs
    etf_data = collector.get_all_etf_data(start_date="2000-01-01")
    
    # Prétraitement des données
    processed_data = collector.preprocess_data(etf_data)
    
    # Sauvegarde des données
    collector.save_data(
        processed_data, 
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                     "data", "processed", "sector_data.csv")
    )
