"""
Tests unitaires pour le module de collecte de données macroéconomiques.
"""

import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ajout du répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.macro_data_collector import MacroDataCollector


class TestMacroDataCollector(unittest.TestCase):
    """
    Classe de tests pour MacroDataCollector.
    """

    def setUp(self):
        """
        Configuration des tests.
        """
        # Mock de l'API FRED pour éviter les appels réels pendant les tests
        self.fred_patcher = patch('src.data.macro_data_collector.Fred')
        self.mock_fred = self.fred_patcher.start()
        
        # Configuration du mock pour get_series
        self.mock_fred_instance = MagicMock()
        self.mock_fred.return_value = self.mock_fred_instance
        
        # Création d'une série fictive pour les tests
        self.dummy_series = pd.Series(
            [1.0, 1.1, 1.2, 1.3],
            index=pd.date_range(start='2020-01-01', periods=4, freq='M')
        )
        self.mock_fred_instance.get_series.return_value = self.dummy_series
        
        # Création du collecteur avec une clé API fictive
        self.collector = MacroDataCollector(api_key='dummy_api_key')

    def tearDown(self):
        """
        Nettoyage après les tests.
        """
        self.fred_patcher.stop()

    def test_init(self):
        """
        Test de l'initialisation du collecteur.
        """
        self.assertEqual(self.collector.fred, self.mock_fred_instance)
        self.assertTrue(len(self.collector.macro_series) > 0)

    def test_get_series(self):
        """
        Test de la récupération d'une série.
        """
        series = self.collector.get_series('GDPC1')
        self.assertIsNotNone(series)
        self.mock_fred_instance.get_series.assert_called_once()
        
        # Vérification des arguments
        args, kwargs = self.mock_fred_instance.get_series.call_args
        self.assertEqual(args[0], 'GDPC1')

    def test_get_all_series(self):
        """
        Test de la récupération de toutes les séries.
        """
        start_date = '2020-01-01'
        end_date = '2020-04-01'
        
        all_data = self.collector.get_all_series(start_date=start_date, end_date=end_date)
        
        # Vérifier que get_series a été appelé pour chaque série
        expected_calls = len(self.collector.macro_series)
        self.assertEqual(self.mock_fred_instance.get_series.call_count, expected_calls)
        
        # Vérifier que le DataFrame résultant contient des données
        self.assertIsInstance(all_data, pd.DataFrame)
        self.assertGreater(len(all_data), 0)

    def test_calculate_yoy_changes(self):
        """
        Test du calcul des variations en glissement annuel.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'A': [100, 110, 120, 130],
            'B': [200, 210, 220, 230]
        }, index=pd.date_range(start='2020-01-01', periods=4, freq='M'))
        
        # Calcul des variations
        yoy_df = self.collector.calculate_yoy_changes(df)
        
        # Vérification du nombre de colonnes et de leurs noms
        self.assertEqual(len(yoy_df.columns), 2)
        self.assertTrue('A_YOY' in yoy_df.columns)
        self.assertTrue('B_YOY' in yoy_df.columns)
        
        # Les 12 premières observations doivent être NaN pour les variations annuelles
        self.assertTrue(yoy_df.iloc[0].isna().all())

    def test_calculate_mom_changes(self):
        """
        Test du calcul des variations en glissement mensuel.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'A': [100, 110, 120, 130],
            'B': [200, 210, 220, 230]
        }, index=pd.date_range(start='2020-01-01', periods=4, freq='M'))
        
        # Calcul des variations
        mom_df = self.collector.calculate_mom_changes(df)
        
        # Vérification du nombre de colonnes et de leurs noms
        self.assertEqual(len(mom_df.columns), 2)
        self.assertTrue('A_MOM' in mom_df.columns)
        self.assertTrue('B_MOM' in mom_df.columns)
        
        # La première observation doit être NaN pour les variations mensuelles
        self.assertTrue(mom_df.iloc[0].isna().all())
        
        # Vérification des valeurs calculées
        self.assertAlmostEqual(mom_df.iloc[1]['A_MOM'], 10.0)
        self.assertAlmostEqual(mom_df.iloc[1]['B_MOM'], 5.0)

    def test_preprocess_data(self):
        """
        Test du prétraitement des données.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'A': [100, 110, 120, 130],
            'B': [200, 210, 220, 230]
        }, index=pd.date_range(start='2020-01-01', periods=4, freq='M'))
        
        # Prétraitement des données
        processed_df = self.collector.preprocess_data(df)
        
        # Vérification du nombre de colonnes
        expected_columns = len(df.columns) * 3  # Originales + YOY + MOM
        self.assertEqual(len(processed_df.columns), expected_columns)
        
        # Vérification de la présence des colonnes de variations
        self.assertTrue('A_YOY' in processed_df.columns)
        self.assertTrue('A_MOM' in processed_df.columns)
        self.assertTrue('B_YOY' in processed_df.columns)
        self.assertTrue('B_MOM' in processed_df.columns)
        
        # Vérification que les valeurs manquantes ont été traitées
        self.assertEqual(processed_df.isna().sum().sum(), 0)

    def test_save_data(self):
        """
        Test de la sauvegarde des données.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'A': [100, 110, 120, 130],
            'B': [200, 210, 220, 230]
        }, index=pd.date_range(start='2020-01-01', periods=4, freq='M'))
        
        # Mock de la méthode to_csv pour éviter d'écrire réellement un fichier
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            # Test de la sauvegarde
            result = self.collector.save_data(df, 'test_file.csv')
            
            # Vérification que to_csv a été appelé
            mock_to_csv.assert_called_once()
            
            # Vérification que la fonction renvoie True
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
