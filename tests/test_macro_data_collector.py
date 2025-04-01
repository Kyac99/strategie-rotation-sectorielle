"""
Tests pour le module de collecte de données macroéconomiques.
"""

import os
import sys
import unittest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Ajout du répertoire parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import du module à tester
from src.data.macro_data_collector import MacroDataCollector


class TestMacroDataCollector(unittest.TestCase):
    """
    Tests pour la classe MacroDataCollector.
    """
    
    def setUp(self):
        """
        Configuration des tests.
        """
        # Mock de la clé API
        self.api_key = "test_api_key"
        
        # Création d'une instance avec la clé mockée
        with patch.dict('os.environ', {'FRED_API_KEY': self.api_key}):
            self.collector = MacroDataCollector()
    
    @patch('src.data.macro_data_collector.Fred')
    def test_init(self, mock_fred):
        """
        Test de l'initialisation.
        """
        # Vérification que Fred est initialisé avec la bonne clé API
        mock_fred.assert_called_once_with(api_key=self.api_key)
        
        # Vérification que les séries économiques sont définies
        self.assertIsNotNone(self.collector.macro_series)
        self.assertGreater(len(self.collector.macro_series), 0)
    
    @patch('src.data.macro_data_collector.Fred')
    def test_get_series(self, mock_fred):
        """
        Test de la méthode get_series.
        """
        # Configuration du mock
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range(start='2020-01-01', periods=3, freq='M')
        )
        
        # Appel de la méthode
        series = self.collector.get_series('GDPC1', '2020-01-01', '2020-03-01', 'm')
        
        # Vérifications
        mock_fred_instance.get_series.assert_called_once_with(
            'GDPC1', 
            observation_start='2020-01-01', 
            observation_end='2020-03-01',
            frequency='m'
        )
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), 3)
    
    @patch('src.data.macro_data_collector.Fred')
    def test_get_all_series(self, mock_fred):
        """
        Test de la méthode get_all_series.
        """
        # Configuration du mock
        mock_fred_instance = mock_fred.return_value
        
        def mock_get_series(series_id, *args, **kwargs):
            return pd.Series(
                [1.0, 2.0, 3.0],
                index=pd.date_range(start='2020-01-01', periods=3, freq='M'),
                name=series_id
            )
        
        mock_fred_instance.get_series.side_effect = mock_get_series
        
        # Appel de la méthode
        all_series = self.collector.get_all_series('2020-01-01', '2020-03-01', 'm')
        
        # Vérifications
        self.assertIsInstance(all_series, pd.DataFrame)
        self.assertEqual(len(all_series), 3)
        self.assertGreaterEqual(len(all_series.columns), 1)
    
    @patch('src.data.macro_data_collector.Fred')
    def test_calculate_yoy_changes(self, mock_fred):
        """
        Test de la méthode calculate_yoy_changes.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'GDPC1': [100, 105, 110, 115, 120],
            'UNRATE': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.date_range(start='2020-01-01', periods=5, freq='M'))
        
        # Appel de la méthode
        yoy = self.collector.calculate_yoy_changes(df)
        
        # Vérifications
        self.assertIsInstance(yoy, pd.DataFrame)
        self.assertEqual(len(yoy), 5)
        self.assertEqual(len(yoy.columns), 2)
        
        # Les 12 premières valeurs doivent être NaN (pas assez d'historique)
        # Mais comme on n'a que 5 valeurs, tout devrait être NaN
        self.assertTrue(yoy.iloc[:4].isna().all().all())
    
    @patch('src.data.macro_data_collector.Fred')
    def test_calculate_mom_changes(self, mock_fred):
        """
        Test de la méthode calculate_mom_changes.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'GDPC1': [100, 105, 110, 115, 120],
            'UNRATE': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.date_range(start='2020-01-01', periods=5, freq='M'))
        
        # Appel de la méthode
        mom = self.collector.calculate_mom_changes(df)
        
        # Vérifications
        self.assertIsInstance(mom, pd.DataFrame)
        self.assertEqual(len(mom), 5)
        self.assertEqual(len(mom.columns), 2)
        
        # La première valeur doit être NaN, les autres non
        self.assertTrue(mom.iloc[0].isna().all())
        self.assertFalse(mom.iloc[1:].isna().all().all())
        
        # Vérification des changements (en %)
        self.assertAlmostEqual(mom['GDPC1_MOM'].iloc[1], 5.0)
        self.assertAlmostEqual(mom['GDPC1_MOM'].iloc[2], 4.761905, places=5)
    
    @patch('src.data.macro_data_collector.Fred')
    def test_preprocess_data(self, mock_fred):
        """
        Test de la méthode preprocess_data.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'GDPC1': [100, 105, 110, 115, 120],
            'UNRATE': [5.0, 5.1, 5.2, 5.3, 5.4]
        }, index=pd.date_range(start='2020-01-01', periods=5, freq='M'))
        
        # Appel de la méthode
        processed = self.collector.preprocess_data(df)
        
        # Vérifications
        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(len(processed), 5)
        
        # Vérification des colonnes créées
        expected_columns = ['GDPC1', 'UNRATE', 'GDPC1_YOY', 'UNRATE_YOY', 'GDPC1_MOM', 'UNRATE_MOM']
        for col in expected_columns:
            self.assertIn(col, processed.columns)
    
    @patch('src.data.macro_data_collector.Fred')
    def test_save_data(self, mock_fred):
        """
        Test de la méthode save_data.
        """
        # Création d'un DataFrame de test
        df = pd.DataFrame({
            'GDPC1': [100, 105, 110],
            'UNRATE': [5.0, 5.1, 5.2]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='M'))
        
        # Création d'un fichier temporaire
        temp_file = 'temp_test_file.csv'
        
        # Appel de la méthode
        success = self.collector.save_data(df, temp_file)
        
        # Vérifications
        self.assertTrue(success)
        self.assertTrue(os.path.exists(temp_file))
        
        # Vérification du contenu du fichier
        saved_df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
        self.assertEqual(len(saved_df), 3)
        self.assertEqual(len(saved_df.columns), 2)
        
        # Nettoyage
        os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
