"""
Tests unitaires pour le module de collecte de données macroéconomiques.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ajout du répertoire racine au path pour l'importation des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.macro_data_collector import MacroDataCollector


class MockFred:
    """Mock pour simuler l'API FRED dans les tests."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def get_series(self, series_id, observation_start=None, observation_end=None, frequency=None):
        """Simule la récupération d'une série de données."""
        # Génération de données simulées pour les tests
        dates = pd.date_range(start='2020-01-01', end='2020-12-01', freq='M')
        
        if series_id == 'GDPC1':
            # PIB réel
            return pd.Series([19000 + i*100 for i in range(len(dates))], index=dates)
        elif series_id == 'CPIAUCSL':
            # Inflation
            return pd.Series([250 + i*0.5 for i in range(len(dates))], index=dates)
        elif series_id == 'UNRATE':
            # Taux de chômage
            return pd.Series([5 + np.sin(i/3) for i in range(len(dates))], index=dates)
        elif series_id == 'FEDFUNDS':
            # Taux d'intérêt directeur
            return pd.Series([1.5 for _ in range(len(dates))], index=dates)
        else:
            # Autres séries
            return pd.Series([100 + i for i in range(len(dates))], index=dates)


class TestMacroDataCollector(unittest.TestCase):
    """Tests pour la classe MacroDataCollector."""
    
    @patch('src.data.macro_data_collector.Fred', MockFred)
    def setUp(self):
        """Configuration initiale pour les tests."""
        self.collector = MacroDataCollector(api_key='test_key')
    
    def test_initialization(self):
        """Teste l'initialisation du collecteur."""
        self.assertIsNotNone(self.collector.fred)
        self.assertEqual(self.collector.fred.api_key, 'test_key')
        self.assertIn('GDPC1', self.collector.macro_series)
        self.assertIn('CPIAUCSL', self.collector.macro_series)
        self.assertIn('UNRATE', self.collector.macro_series)
    
    def test_get_series(self):
        """Teste la récupération d'une série spécifique."""
        # Récupération du PIB
        gdp_series = self.collector.get_series('GDPC1', start_date='2020-01-01', end_date='2020-12-01')
        
        self.assertIsInstance(gdp_series, pd.Series)
        self.assertEqual(len(gdp_series), 12)  # 12 mois de données
        self.assertTrue((gdp_series.index.month == range(1, 13)).all())
        self.assertTrue(np.all(np.diff(gdp_series.values) > 0))  # Valeurs croissantes
    
    def test_get_all_series(self):
        """Teste la récupération de toutes les séries."""
        all_data = self.collector.get_all_series(start_date='2020-01-01', end_date='2020-12-01')
        
        self.assertIsInstance(all_data, pd.DataFrame)
        self.assertTrue(len(all_data.columns) > 0)
        self.assertEqual(len(all_data), 12)  # 12 mois de données
        
        # Vérification de la présence des séries principales
        self.assertIn('GDPC1', all_data.columns)
        self.assertIn('CPIAUCSL', all_data.columns)
        self.assertIn('UNRATE', all_data.columns)
    
    def test_calculate_yoy_changes(self):
        """Teste le calcul des variations en glissement annuel."""
        # Création d'un DataFrame de test
        dates = pd.date_range(start='2020-01-01', end='2021-12-01', freq='M')
        data = {
            'GDPC1': [19000 + i*100 for i in range(len(dates))],
            'CPIAUCSL': [250 + i*0.5 for i in range(len(dates))]
        }
        df = pd.DataFrame(data, index=dates)
        
        # Calcul des variations
        yoy_changes = self.collector.calculate_yoy_changes(df)
        
        self.assertIsInstance(yoy_changes, pd.DataFrame)
        self.assertEqual(len(yoy_changes), len(df))
        self.assertIn('GDPC1_YOY', yoy_changes.columns)
        self.assertIn('CPIAUCSL_YOY', yoy_changes.columns)
        
        # Vérification que les 12 premiers mois sont NaN (pas d'année précédente)
        self.assertTrue(yoy_changes.iloc[:12].isna().all().all())
        
        # Vérification du calcul correct pour les mois avec données YoY
        # Pour GDPC1, l'augmentation est de 100 par mois, soit 1200 par an
        # Donc pour une valeur de départ de 19000, après 12 mois on a 20200, soit 6.32% d'augmentation
        expected_yoy = (20200 / 19000 - 1) * 100  # En pourcentage
        self.assertAlmostEqual(yoy_changes.iloc[12]['GDPC1_YOY'], expected_yoy, places=2)
    
    def test_calculate_mom_changes(self):
        """Teste le calcul des variations en glissement mensuel."""
        # Création d'un DataFrame de test
        dates = pd.date_range(start='2020-01-01', end='2020-12-01', freq='M')
        data = {
            'GDPC1': [19000 + i*100 for i in range(len(dates))],
            'CPIAUCSL': [250 + i*0.5 for i in range(len(dates))]
        }
        df = pd.DataFrame(data, index=dates)
        
        # Calcul des variations
        mom_changes = self.collector.calculate_mom_changes(df)
        
        self.assertIsInstance(mom_changes, pd.DataFrame)
        self.assertEqual(len(mom_changes), len(df))
        self.assertIn('GDPC1_MOM', mom_changes.columns)
        self.assertIn('CPIAUCSL_MOM', mom_changes.columns)
        
        # Vérification que le premier mois est NaN (pas de mois précédent)
        self.assertTrue(mom_changes.iloc[0].isna().all())
        
        # Vérification du calcul correct pour les mois suivants
        # Pour GDPC1, l'augmentation est de 100 par mois
        # Donc pour une valeur de 19000, l'augmentation au mois suivant est de 0.526%
        expected_mom = (100 / 19000) * 100  # En pourcentage
        self.assertAlmostEqual(mom_changes.iloc[1]['GDPC1_MOM'], expected_mom, places=2)
    
    def test_preprocess_data(self):
        """Teste le prétraitement complet des données."""
        # Récupération des données
        data = self.collector.get_all_series(start_date='2020-01-01', end_date='2020-12-01')
        
        # Prétraitement
        processed_data = self.collector.preprocess_data(data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertTrue(len(processed_data.columns) > len(data.columns))
        
        # Vérification de la présence des colonnes de variations
        for col in data.columns:
            self.assertIn(f"{col}_YOY", processed_data.columns)
            self.assertIn(f"{col}_MOM", processed_data.columns)
        
        # Vérification de l'absence de valeurs manquantes
        self.assertTrue(processed_data.notna().all().all())
    
    def test_save_data(self):
        """Teste la sauvegarde des données."""
        # Création d'un DataFrame de test
        dates = pd.date_range(start='2020-01-01', end='2020-12-01', freq='M')
        data = {
            'GDPC1': [19000 + i*100 for i in range(len(dates))],
            'CPIAUCSL': [250 + i*0.5 for i in range(len(dates))]
        }
        df = pd.DataFrame(data, index=dates)
        
        # Sauvegarde dans un fichier temporaire
        temp_file = 'temp_test_data.csv'
        result = self.collector.save_data(df, temp_file)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(temp_file))
        
        # Vérification que le fichier contient les bonnes données
        loaded_df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(df, loaded_df)
        
        # Nettoyage
        os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
