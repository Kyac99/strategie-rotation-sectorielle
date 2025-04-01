"""
Tests pour le module de classification des cycles économiques.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from unittest.mock import patch, MagicMock

# Ajout du répertoire parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import du module à tester
from src.models.economic_cycle_classifier import EconomicCycleClassifier


class TestEconomicCycleClassifier(unittest.TestCase):
    """
    Tests pour la classe EconomicCycleClassifier.
    """
    
    def setUp(self):
        """
        Configuration des tests.
        """
        # Création d'un ensemble de données synthétique
        self.create_synthetic_data()
        
        # Création d'une instance non supervisée
        self.classifier_unsupervised = EconomicCycleClassifier(supervised=False)
        
        # Création d'une instance supervisée
        self.classifier_supervised = EconomicCycleClassifier(supervised=True)
    
    def create_synthetic_data(self):
        """
        Création d'un ensemble de données synthétique pour les tests.
        """
        # Création d'un index de dates
        dates = pd.date_range(start='2000-01-01', periods=120, freq='M')
        
        # Création de données cycliques pour simuler les cycles économiques
        # Cycle de 48 mois (4 ans) pour le PIB
        gdp_cycle = np.sin(np.linspace(0, 3*np.pi, 120))
        
        # Cycle de 48 mois décalé de 6 mois pour le chômage (inverse du PIB)
        unemployment_cycle = -np.sin(np.linspace(0, 3*np.pi, 120) + np.pi/4)
        
        # Cycle de 24 mois (2 ans) pour l'inflation
        inflation_cycle = np.sin(np.linspace(0, 6*np.pi, 120))
        
        # Création de données aléatoires pour d'autres indicateurs
        fed_funds = np.cumsum(np.random.normal(0, 0.1, 120))
        yield_spread = np.random.normal(0, 1, 120)
        
        # Création du DataFrame
        self.data = pd.DataFrame({
            'GDPC1_YOY': 2 + gdp_cycle * 3,  # Croissance PIB entre -1% et 5%
            'INDPRO_YOY': 2 + gdp_cycle * 5,  # Production industrielle plus volatile
            'UNRATE': 5 + unemployment_cycle * 3,  # Chômage entre 2% et 8%
            'UNRATE_YOY': unemployment_cycle * 2,  # Variation du chômage
            'CPIAUCSL_YOY': 2 + inflation_cycle * 1.5,  # Inflation entre 0.5% et 3.5%
            'FEDFUNDS': np.abs(fed_funds) * 2,  # Taux Fed Funds positif
            'T10Y2Y': yield_spread,  # Spread de taux
            'BAMLH0A0HYM2': 3 + inflation_cycle * 2,  # Spread de crédit HY
            'UMCSENT': 80 + gdp_cycle * 20,  # Confiance des consommateurs
            'VIXCLS': 15 - gdp_cycle * 10  # VIX
        }, index=dates)
        
        # Labels pour les tests supervisés
        # Création d'étiquettes basées sur la position dans le cycle du PIB
        phase_values = []
        for i in range(len(gdp_cycle)):
            if gdp_cycle[i] > 0.5 and inflation_cycle[i] < 0:
                phase = 'Expansion'
            elif gdp_cycle[i] > 0.5 and inflation_cycle[i] > 0:
                phase = 'Surchauffe'
            elif gdp_cycle[i] < -0.5 and inflation_cycle[i] > 0:
                phase = 'Ralentissement'
            elif gdp_cycle[i] < -0.5 and inflation_cycle[i] < 0:
                phase = 'Récession'
            else:
                phase = 'Reprise'
            phase_values.append(phase)
        
        self.labels = pd.Series(phase_values, index=dates)
    
    def test_init(self):
        """
        Test de l'initialisation.
        """
        # Vérification des attributs de l'instance non supervisée
        self.assertFalse(self.classifier_unsupervised.supervised)
        self.assertIsNotNone(self.classifier_unsupervised.scaler)
        self.assertIsNotNone(self.classifier_unsupervised.model)
        self.assertIsNotNone(self.classifier_unsupervised.cycle_labels)
        self.assertIsNotNone(self.classifier_unsupervised.key_indicators)
        
        # Vérification des attributs de l'instance supervisée
        self.assertTrue(self.classifier_supervised.supervised)
        self.assertIsNotNone(self.classifier_supervised.scaler)
        self.assertIsNotNone(self.classifier_supervised.model)
        self.assertIsNotNone(self.classifier_supervised.cycle_labels)
        self.assertIsNotNone(self.classifier_supervised.key_indicators)
    
    def test_select_features(self):
        """
        Test de la méthode _select_features.
        """
        # Appel de la méthode
        features = self.classifier_unsupervised._select_features(self.data)
        
        # Vérifications
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        
        # Vérification que les colonnes sont bien parmi les key_indicators
        for col in features.columns:
            self.assertIn(col, self.classifier_unsupervised.key_indicators)
    
    def test_fit_unsupervised(self):
        """
        Test de la méthode fit avec le modèle non supervisé.
        """
        # Appel de la méthode
        self.classifier_unsupervised.fit(self.data)
        
        # Vérifications
        self.assertIsNotNone(self.classifier_unsupervised.model.cluster_centers_)
        self.assertEqual(len(self.classifier_unsupervised.model.cluster_centers_), 5)  # 5 clusters
    
    def test_fit_supervised(self):
        """
        Test de la méthode fit avec le modèle supervisé.
        """
        # Appel de la méthode
        self.classifier_supervised.fit(self.data, self.labels)
        
        # Vérifications
        self.assertIsNotNone(self.classifier_supervised.model.feature_importances_)
    
    def test_predict_unsupervised(self):
        """
        Test de la méthode predict avec le modèle non supervisé.
        """
        # Entrainement du modèle
        self.classifier_unsupervised.fit(self.data)
        
        # Appel de la méthode
        phases = self.classifier_unsupervised.predict(self.data)
        
        # Vérifications
        self.assertIsInstance(phases, pd.Series)
        self.assertEqual(len(phases), len(self.data))
        
        # Vérification que toutes les phases sont dans les labels définis
        for phase in phases.unique():
            self.assertIn(phase, self.classifier_unsupervised.cycle_labels.values())
    
    def test_predict_supervised(self):
        """
        Test de la méthode predict avec le modèle supervisé.
        """
        # Entrainement du modèle
        self.classifier_supervised.fit(self.data, self.labels)
        
        # Appel de la méthode
        phases = self.classifier_supervised.predict(self.data)
        
        # Vérifications
        self.assertIsInstance(phases, pd.Series)
        self.assertEqual(len(phases), len(self.data))
        
        # Vérification que toutes les phases sont dans les labels définis
        for phase in phases.unique():
            self.assertIn(phase, list(self.labels.unique()))
    
    def test_save_and_load_model(self):
        """
        Test des méthodes save_model et load_model.
        """
        # Entrainement du modèle
        self.classifier_unsupervised.fit(self.data)
        
        # Création d'un fichier temporaire
        temp_file = 'temp_test_model.joblib'
        
        # Sauvegarde du modèle
        success = self.classifier_unsupervised.save_model(temp_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(temp_file))
        
        # Chargement du modèle
        loaded_classifier = EconomicCycleClassifier.load_model(temp_file)
        self.assertIsNotNone(loaded_classifier)
        self.assertFalse(loaded_classifier.supervised)
        
        # Vérification que le modèle chargé donne les mêmes prédictions
        original_predictions = self.classifier_unsupervised.predict(self.data)
        loaded_predictions = loaded_classifier.predict(self.data)
        
        pd.testing.assert_series_equal(
            original_predictions,
            loaded_predictions,
            check_names=False
        )
        
        # Nettoyage
        os.remove(temp_file)
    
    def test_plot_cycle_distribution(self):
        """
        Test de la méthode plot_cycle_distribution.
        """
        # Entrainement du modèle
        self.classifier_unsupervised.fit(self.data)
        
        # Appel de la méthode
        fig = self.classifier_unsupervised.plot_cycle_distribution(self.data)
        
        # Vérifications
        self.assertIsInstance(fig, plt.Figure)
        
        # Nettoyage
        plt.close(fig)
    
    def test_plot_cycle_characteristics(self):
        """
        Test de la méthode plot_cycle_characteristics.
        """
        # Entrainement du modèle
        self.classifier_unsupervised.fit(self.data)
        
        # Appel de la méthode
        fig = self.classifier_unsupervised.plot_cycle_characteristics(self.data)
        
        # Vérifications
        self.assertIsInstance(fig, plt.Figure)
        
        # Nettoyage
        plt.close(fig)
    
    def test_edge_case_empty_data(self):
        """
        Test avec un DataFrame vide.
        """
        # Création d'un DataFrame vide
        empty_data = pd.DataFrame()
        
        # Vérification qu'une exception est levée lors de l'appel à fit
        with self.assertRaises(Exception):
            self.classifier_unsupervised.fit(empty_data)
    
    def test_edge_case_missing_indicators(self):
        """
        Test avec des données manquant les indicateurs clés.
        """
        # Création de données avec des indicateurs manquants
        incomplete_data = pd.DataFrame({
            'RANDOM_COLUMN': np.random.rand(10)
        }, index=pd.date_range(start='2000-01-01', periods=10, freq='M'))
        
        # Vérification qu'une exception est levée lors de l'appel à fit
        with self.assertRaises(Exception):
            self.classifier_unsupervised.fit(incomplete_data)


if __name__ == '__main__':
    unittest.main()
