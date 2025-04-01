"""
Tests unitaires pour les collecteurs de données.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ajout du répertoire racine au path pour l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector


class TestMacroDataCollector(unittest.TestCase):
    """Tests pour la classe MacroDataCollector."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Mock pour la clé API
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            self.collector = MacroDataCollector()
        
        # Mock pour la classe Fred
        self.mock_fred = MagicMock()
        self.collector.fred = self.mock_fred

    def test_init(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.collector.fred)
        self.assertIsNotNone(self.collector.macro_series)

    def test_get_series(self):
        """Test de la méthode get_series."""
        # Configuration du mock
        mock_series = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
        self.mock_fred.get_series.return_value = mock_series
        
        # Appel de la méthode
        result = self.collector.get_series('GDPC1')
        
        # Vérification
        self.mock_fred.get_series.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_get_all_series(self):
        """Test de la méthode get_all_series."""
        # Configuration du mock
        mock_series = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
        self.mock_fred.get_series.return_value = mock_series
        
        # Appel de la méthode
        result = self.collector.get_all_series(start_date='2023-01-01', end_date='2023-01-03')
        
        # Vérification
        self.assertIsNotNone(result)
        # Le nombre d'appels devrait être égal au nombre de séries dans macro_series
        self.assertEqual(self.mock_fred.get_series.call_count, len(self.collector.macro_series))

    def test_calculate_yoy_changes(self):
        """Test de la méthode calculate_yoy_changes."""
        # Données de test
        test_data = pd.DataFrame({
            'A': [100, 110, 121, 133.1],
            'B': [200, 180, 162, 145.8]
        }, index=pd.date_range('2020-01-01', periods=4, freq='Q'))
        
        # Appel de la méthode
        result = self.collector.calculate_yoy_changes(test_data)
        
        # Vérification
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, test_data.shape)
        # Vérification des noms de colonnes
        self.assertEqual(list(result.columns), ['A_YOY', 'B_YOY'])
        # Vérification des valeurs calculées (seule la dernière valeur peut être calculée avec 4 périodes)
        np.testing.assert_almost_equal(result.iloc[3, 0], 33.1)  # A: (133.1 / 100 - 1) * 100
        np.testing.assert_almost_equal(result.iloc[3, 1], -27.1)  # B: (145.8 / 200 - 1) * 100

    def test_calculate_mom_changes(self):
        """Test de la méthode calculate_mom_changes."""
        # Données de test
        test_data = pd.DataFrame({
            'A': [100, 110, 121, 133.1],
            'B': [200, 180, 162, 145.8]
        }, index=pd.date_range('2020-01-01', periods=4, freq='M'))
        
        # Appel de la méthode
        result = self.collector.calculate_mom_changes(test_data)
        
        # Vérification
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, test_data.shape)
        # Vérification des noms de colonnes
        self.assertEqual(list(result.columns), ['A_MOM', 'B_MOM'])
        # Vérification des valeurs calculées
        np.testing.assert_almost_equal(result.iloc[1, 0], 10.0)  # A: (110 / 100 - 1) * 100
        np.testing.assert_almost_equal(result.iloc[1, 1], -10.0)  # B: (180 / 200 - 1) * 100

    def test_preprocess_data(self):
        """Test de la méthode preprocess_data."""
        # Données de test
        test_data = pd.DataFrame({
            'A': [100, 110, 121, 133.1],
            'B': [200, 180, 162, 145.8]
        }, index=pd.date_range('2020-01-01', periods=4, freq='Q'))
        
        # Configuration des mocks pour les méthodes appelées
        with patch.object(self.collector, 'calculate_yoy_changes') as mock_yoy:
            with patch.object(self.collector, 'calculate_mom_changes') as mock_mom:
                # Valeurs de retour des mocks
                mock_yoy.return_value = pd.DataFrame({
                    'A_YOY': [np.nan, np.nan, np.nan, 33.1],
                    'B_YOY': [np.nan, np.nan, np.nan, -27.1]
                }, index=test_data.index)
                mock_mom.return_value = pd.DataFrame({
                    'A_MOM': [np.nan, 10.0, 10.0, 10.0],
                    'B_MOM': [np.nan, -10.0, -10.0, -10.0]
                }, index=test_data.index)
                
                # Appel de la méthode
                result = self.collector.preprocess_data(test_data)
                
                # Vérification
                self.assertIsNotNone(result)
                mock_yoy.assert_called_once()
                mock_mom.assert_called_once()
                # Vérification des colonnes attendues
                expected_columns = list(test_data.columns) + list(mock_yoy.return_value.columns) + list(mock_mom.return_value.columns)
                self.assertEqual(list(result.columns), expected_columns)


class TestSectorDataCollector(unittest.TestCase):
    """Tests pour la classe SectorDataCollector."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.collector = SectorDataCollector()

    def test_init(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.collector.sector_etfs)
        self.assertIsNotNone(self.collector.market_etfs)
        self.assertIsNotNone(self.collector.style_etfs)
        self.assertIsNotNone(self.collector.bond_commodity_etfs)
        self.assertIsNotNone(self.collector.all_etfs)

    @patch('yfinance.Ticker')
    def test_get_etf_data(self, mock_ticker):
        """Test de la méthode get_etf_data."""
        # Configuration du mock
        mock_instance = MagicMock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Appel de la méthode
        result = self.collector.get_etf_data('SPY')
        
        # Vérification
        mock_ticker.assert_called_once_with('SPY')
        mock_instance.history.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    @patch('yfinance.download')
    def test_get_all_etf_data(self, mock_download):
        """Test de la méthode get_all_etf_data."""
        # Configuration du mock
        mock_data = pd.DataFrame({
            ('SPY', 'Open'): [100, 101, 102],
            ('SPY', 'High'): [105, 106, 107],
            ('SPY', 'Low'): [95, 96, 97],
            ('SPY', 'Close'): [102, 103, 104],
            ('SPY', 'Volume'): [1000, 1100, 1200],
            ('QQQ', 'Open'): [200, 201, 202],
            ('QQQ', 'High'): [205, 206, 207],
            ('QQQ', 'Low'): [195, 196, 197],
            ('QQQ', 'Close'): [202, 203, 204],
            ('QQQ', 'Volume'): [2000, 2100, 2200]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data
        
        # Appel de la méthode
        result = self.collector.get_all_etf_data(start_date='2023-01-01', end_date='2023-01-03')
        
        # Vérification
        mock_download.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # SPY et QQQ
        self.assertIn('SPY', result)
        self.assertIn('QQQ', result)

    def test_extract_prices(self):
        """Test de la méthode extract_prices."""
        # Données de test
        test_data = {
            'SPY': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'adj close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range('2023-01-01', periods=3)),
            'QQQ': pd.DataFrame({
                'open': [200, 201, 202],
                'high': [205, 206, 207],
                'low': [195, 196, 197],
                'close': [202, 203, 204],
                'adj close': [201, 202, 203],
                'volume': [2000, 2100, 2200]
            }, index=pd.date_range('2023-01-01', periods=3))
        }
        
        # Appel de la méthode
        result = self.collector.extract_prices(test_data)
        
        # Vérification
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3, 2))  # 3 dates, 2 colonnes (SPY, QQQ)
        self.assertIn('SPY', result.columns)
        self.assertIn('QQQ', result.columns)
        # Vérification des valeurs (adj close)
        self.assertEqual(result['SPY'].iloc[0], 101)
        self.assertEqual(result['QQQ'].iloc[0], 201)

    def test_calculate_returns(self):
        """Test de la méthode calculate_returns."""
        # Données de test
        test_data = pd.DataFrame({
            'SPY': [100, 110, 121, 133.1],
            'QQQ': [200, 220, 242, 266.2]
        }, index=pd.date_range('2020-01-01', periods=4, freq='M'))
        
        # Périodes de test
        test_periods = {
            'monthly': 1,
            'quarterly': 3
        }
        
        # Appel de la méthode
        result = self.collector.calculate_returns(test_data, periods=test_periods)
        
        # Vérification
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # deux périodes
        self.assertIn('monthly', result)
        self.assertIn('quarterly', result)
        # Vérification des valeurs calculées
        np.testing.assert_almost_equal(result['monthly']['SPY_monthly'].iloc[1], 10.0)  # (110/100 - 1) * 100
        np.testing.assert_almost_equal(result['quarterly']['SPY_quarterly'].iloc[3], 21.0)  # (133.1/110 - 1) * 100

    def test_preprocess_data(self):
        """Test de la méthode preprocess_data."""
        # Données de test
        test_data = {
            'SPY': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'adj close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range('2023-01-01', periods=3)),
            'QQQ': pd.DataFrame({
                'open': [200, 201, 202],
                'high': [205, 206, 207],
                'low': [195, 196, 197],
                'close': [202, 203, 204],
                'adj close': [201, 202, 203],
                'volume': [2000, 2100, 2200]
            }, index=pd.date_range('2023-01-01', periods=3))
        }
        
        # Configuration des mocks pour les méthodes appelées
        with patch.object(self.collector, 'extract_prices') as mock_extract:
            with patch.object(self.collector, 'calculate_returns') as mock_returns:
                with patch.object(self.collector, 'calculate_relative_strength') as mock_rs:
                    with patch.object(self.collector, 'calculate_volatility') as mock_vol:
                        # Valeurs de retour des mocks
                        mock_extract.return_value = pd.DataFrame({
                            'SPY': [101, 102, 103],
                            'QQQ': [201, 202, 203]
                        }, index=pd.date_range('2023-01-01', periods=3))
                        mock_returns.return_value = {
                            'daily': pd.DataFrame({
                                'SPY_daily': [np.nan, 0.99, 0.98],
                                'QQQ_daily': [np.nan, 0.5, 0.495]
                            }, index=pd.date_range('2023-01-01', periods=3))
                        }
                        mock_rs.return_value = {
                            '1_month': pd.DataFrame({
                                'QQQ_RS_1_month': [np.nan, 0.5, 0.505]
                            }, index=pd.date_range('2023-01-01', periods=3))
                        }
                        mock_vol.return_value = {
                            '1_month': pd.DataFrame({
                                'SPY_VOL_1_month': [np.nan, 10.0, 9.5],
                                'QQQ_VOL_1_month': [np.nan, 15.0, 14.2]
                            }, index=pd.date_range('2023-01-01', periods=3))
                        }
                        
                        # Appel de la méthode
                        result = self.collector.preprocess_data(test_data)
                        
                        # Vérification
                        self.assertIsNotNone(result)
                        mock_extract.assert_called_once()
                        mock_returns.assert_called_once()
                        mock_rs.assert_called_once()
                        mock_vol.assert_called_once()
                        # Vérification des colonnes attendues (combinaison de tous les DataFrames)
                        expected_columns = list(mock_extract.return_value.columns) + \
                                          list(mock_returns.return_value['daily'].columns) + \
                                          list(mock_rs.return_value['1_month'].columns) + \
                                          list(mock_vol.return_value['1_month'].columns)
                        self.assertEqual(sorted(list(result.columns)), sorted(expected_columns))


if __name__ == '__main__':
    unittest.main()
