"""
Tests unitaires pour les collecteurs de données.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ajout du répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector


class TestMacroDataCollector(unittest.TestCase):
    """
    Tests pour le collecteur de données macroéconomiques.
    """
    
    @patch('src.data.macro_data_collector.Fred')
    def setUp(self, mock_fred):
        """Configuration des tests."""
        # Mock de l'API FRED
        self.mock_fred_instance = mock_fred.return_value
        self.mock_fred_instance.get_series.return_value = pd.Series(
            [1.0, 1.1, 1.2, 1.3],
            index=pd.date_range(start='2022-01-01', periods=4, freq='M')
        )
        
        # Création du collecteur avec la mock
        self.collector = MacroDataCollector(api_key='dummy_key')
    
    def test_get_series(self):
        """Test de la récupération d'une série."""
        series = self.collector.get_series('GDPC1')
        
        # Vérification que la méthode get_series de l'API FRED a été appelée
        self.mock_fred_instance.get_series.assert_called_once()
        
        # Vérification que la série retournée est une pandas Series
        self.assertIsInstance(series, pd.Series)
        
        # Vérification que la série a 4 éléments
        self.assertEqual(len(series), 4)
    
    def test_get_all_series(self):
        """Test de la récupération de toutes les séries."""
        all_series = self.collector.get_all_series(start_date='2022-01-01', end_date='2022-04-01')
        
        # Vérification que la méthode get_series de l'API FRED a été appelée pour chaque série
        self.assertEqual(self.mock_fred_instance.get_series.call_count, len(self.collector.macro_series))
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(all_series, pd.DataFrame)
    
    def test_calculate_yoy_changes(self):
        """Test du calcul des variations en glissement annuel."""
        df = pd.DataFrame({
            'GDPC1': [100, 102, 105, 108],
            'INDPRO': [50, 52, 53, 55]
        }, index=pd.date_range(start='2022-01-01', periods=4, freq='M'))
        
        yoy_changes = self.collector.calculate_yoy_changes(df)
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(yoy_changes, pd.DataFrame)
        
        # Vérification des noms de colonnes
        self.assertEqual(set(yoy_changes.columns), {'GDPC1_YOY', 'INDPRO_YOY'})
        
        # Les trois premières valeurs devraient être NaN car nous n'avons pas 12 mois de données
        self.assertTrue(yoy_changes.iloc[:3].isna().all().all())
    
    def test_calculate_mom_changes(self):
        """Test du calcul des variations en glissement mensuel."""
        df = pd.DataFrame({
            'GDPC1': [100, 102, 105, 108],
            'INDPRO': [50, 52, 53, 55]
        }, index=pd.date_range(start='2022-01-01', periods=4, freq='M'))
        
        mom_changes = self.collector.calculate_mom_changes(df)
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(mom_changes, pd.DataFrame)
        
        # Vérification des noms de colonnes
        self.assertEqual(set(mom_changes.columns), {'GDPC1_MOM', 'INDPRO_MOM'})
        
        # Vérification des calculs
        np.testing.assert_almost_equal(mom_changes['GDPC1_MOM'].iloc[1], 2.0)
        np.testing.assert_almost_equal(mom_changes['INDPRO_MOM'].iloc[1], 4.0)
    
    def test_preprocess_data(self):
        """Test du prétraitement des données."""
        df = pd.DataFrame({
            'GDPC1': [100, 102, 105, 108],
            'INDPRO': [50, 52, 53, 55]
        }, index=pd.date_range(start='2022-01-01', periods=4, freq='M'))
        
        processed_data = self.collector.preprocess_data(df)
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # Vérification que les colonnes des variations ont été ajoutées
        expected_columns = {'GDPC1', 'INDPRO', 'GDPC1_YOY', 'INDPRO_YOY', 'GDPC1_MOM', 'INDPRO_MOM'}
        self.assertEqual(set(processed_data.columns), expected_columns)


class TestSectorDataCollector(unittest.TestCase):
    """
    Tests pour le collecteur de données sectorielles.
    """
    
    @patch('src.data.sector_data_collector.yf.Ticker')
    @patch('src.data.sector_data_collector.yf.download')
    def setUp(self, mock_download, mock_ticker):
        """Configuration des tests."""
        # Mock de l'API Yahoo Finance
        self.mock_ticker_instance = mock_ticker.return_value
        self.mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200],
            'Dividends': [0, 0, 0],
            'Stock Splits': [0, 0, 0]
        }, index=pd.date_range(start='2022-01-01', periods=3, freq='D'))
        
        # Mock de la fonction download
        mock_download.return_value = pd.concat(
            [self.mock_ticker_instance.history() for _ in range(2)],
            keys=['XLY', 'SPY'],
            axis=1
        )
        
        # Création du collecteur
        self.collector = SectorDataCollector()
    
    def test_get_etf_data(self):
        """Test de la récupération des données d'un ETF."""
        etf_data = self.collector.get_etf_data('XLY')
        
        # Vérification que la méthode history de l'API Yahoo Finance a été appelée
        self.mock_ticker_instance.history.assert_called_once()
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(etf_data, pd.DataFrame)
        
        # Vérification que le DataFrame a 3 lignes
        self.assertEqual(len(etf_data), 3)
    
    def test_get_all_etf_data(self):
        """Test de la récupération des données de tous les ETFs."""
        all_etf_data = self.collector.get_all_etf_data(start_date='2022-01-01', end_date='2022-01-03')
        
        # Vérification que le résultat est un dictionnaire
        self.assertIsInstance(all_etf_data, dict)
        
        # Vérification que le dictionnaire contient des DataFrames
        for ticker, df in all_etf_data.items():
            self.assertIsInstance(df, pd.DataFrame)
    
    def test_extract_prices(self):
        """Test de l'extraction des prix."""
        etf_data = {
            'XLY': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'adj close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range(start='2022-01-01', periods=3, freq='D')),
            'SPY': pd.DataFrame({
                'open': [200, 201, 202],
                'high': [205, 206, 207],
                'low': [195, 196, 197],
                'close': [202, 203, 204],
                'adj close': [201, 202, 203],
                'volume': [2000, 2100, 2200]
            }, index=pd.date_range(start='2022-01-01', periods=3, freq='D'))
        }
        
        prices = self.collector.extract_prices(etf_data)
        
        # Vérification que le résultat est un DataFrame
        self.assertIsInstance(prices, pd.DataFrame)
        
        # Vérification des colonnes
        self.assertEqual(set(prices.columns), {'XLY', 'SPY'})
        
        # Vérification des valeurs
        np.testing.assert_array_equal(prices['XLY'].values, [101, 102, 103])
        np.testing.assert_array_equal(prices['SPY'].values, [201, 202, 203])
    
    def test_calculate_returns(self):
        """Test du calcul des rendements."""
        price_df = pd.DataFrame({
            'XLY': [100, 102, 105, 108, 110, 112, 114],
            'SPY': [200, 204, 210, 216, 220, 224, 228]
        }, index=pd.date_range(start='2022-01-01', periods=7, freq='D'))
        
        returns = self.collector.calculate_returns(price_df, periods={'daily': 1, 'weekly': 5})
        
        # Vérification que le résultat est un dictionnaire
        self.assertIsInstance(returns, dict)
        
        # Vérification des clés
        self.assertEqual(set(returns.keys()), {'daily', 'weekly'})
        
        # Vérification des valeurs
        self.assertIsInstance(returns['daily'], pd.DataFrame)
        self.assertIsInstance(returns['weekly'], pd.DataFrame)
        
        # Vérification des calculs pour les rendements journaliers
        np.testing.assert_almost_equal(returns['daily']['XLY_daily'].iloc[1], 2.0)
        np.testing.assert_almost_equal(returns['daily']['SPY_daily'].iloc[1], 2.0)
        
        # La première ligne devrait être NaN car nous n'avons pas de jour précédent
        self.assertTrue(returns['daily'].iloc[0].isna().all())
        
        # Les 4 premières lignes pour les rendements hebdomadaires devraient être NaN
        self.assertTrue(returns['weekly'].iloc[:4].isna().all().all())
        
        # Vérification du calcul pour le rendement hebdomadaire
        np.testing.assert_almost_equal(returns['weekly']['XLY_weekly'].iloc[5], 10.0)
        np.testing.assert_almost_equal(returns['weekly']['SPY_weekly'].iloc[5], 10.0)


if __name__ == '__main__':
    unittest.main()
