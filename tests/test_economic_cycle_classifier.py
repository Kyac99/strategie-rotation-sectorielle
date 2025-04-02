"""
Tests unitaires pour le module de classification des cycles économiques.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile

# Ajout du répertoire racine au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.economic_cycle_classifier import EconomicCycleClassifier


class TestEconomicCycleClassifier(unittest.TestCase):
    """Tests pour la classe EconomicCycleClassifier."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Création de données de test
        dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='M')
        
        # Création de données macroéconomiques de test avec des cycles simulés
        np.random.seed(42)  # Pour la reproductibilité
        self.macro_data = pd.DataFrame(index=dates)
        
        # Simulation de 5 cycles économiques sur la période
        n_periods = len(dates)
        cycle_length = n_periods // 5
        
        # Création d'une tendance cyclique pour les indicateurs
        cycle = np.sin(np.linspace(0, 10 * np.pi, n_periods))
        
        # PIB avec tendance haussière et cycles
        self.macro_data['GDPC1'] = 100 + np.linspace(0, 50, n_periods) + cycle * 10
        
        # Taux de chômage avec cycles inversés par rapport au PIB (contre-cyclique)
        self.macro_data['UNRATE'] = 5 - cycle * 2 + np.random.normal(0, 0.2, n_periods)
        self.macro_data['UNRATE'] = self.macro_data['UNRATE'].clip(lower=3)  # Le taux de chômage ne descend pas trop bas
        
        # Inflation avec cycles légèrement décalés
        inflation_cycle = np.sin(np.linspace(np.pi/4, 10 * np.pi + np.pi/4, n_periods))
        self.macro_data['CPIAUCSL'] = 100 + np.linspace(0, 30, n_periods) + inflation_cycle * 5
        
        # Taux d'intérêt suivant l'inflation avec un léger décalage
        interest_cycle = np.sin(np.linspace(np.pi/3, 10 * np.pi + np.pi/3, n_periods))
        self.macro_data['FEDFUNDS'] = 2 + interest_cycle * 3 + np.random.normal(0, 0.1, n_periods)
        self.macro_data['FEDFUNDS'] = self.macro_data['FEDFUNDS'].clip(lower=0.25)  # Le taux ne descend pas trop bas
        
        # Spread de taux
        self.macro_data['T10Y2Y'] = 1 + cycle * 0.5 + np.random.normal(0, 0.1, n_periods)
        
        # Volatilité du marché (contre-cyclique avec un décalage)
        vix_cycle = -np.sin(np.linspace(-np.pi/6, 10 * np.pi - np.pi/6, n_periods))
        self.macro_data['VIXCLS'] = 15 + vix_cycle * 10 + np.random.normal(0, 2, n_periods)
        self.macro_data['VIXCLS'] = self.macro_data['VIXCLS'].clip(lower=10)  # Le VIX ne descend pas trop bas
        
        # Calcul des indicateurs dérivés
        for col in ['GDPC1', 'UNRATE', 'CPIAUCSL']:
            # Variation annuelle (12 mois) en pourcentage
            self.macro_data[f'{col}_YOY'] = self.macro_data[col].pct_change(12) * 100
        
        # Suppression des valeurs manquantes (premiers 12 mois pour les variations annuelles)
        self.macro_data = self.macro_data.iloc[12:].copy()
    
    def test_init(self):
        """Test de l'initialisation du classifieur."""
        # Test du mode non supervisé
        classifier = EconomicCycleClassifier(supervised=False)
        self.assertFalse(classifier.supervised)
        self.assertIsNotNone(classifier.model)
        self.assertEqual(len(classifier.cycle_labels), 5)  # 5 phases du cycle
        
        # Test du mode supervisé
        classifier = EconomicCycleClassifier(supervised=True)
        self.assertTrue(classifier.supervised)
        self.assertIsNotNone(classifier.model)
    
    def test_fit_predict_unsupervised(self):
        """Test des méthodes fit et predict en mode non supervisé."""
        # Initialisation du classifieur
        classifier = EconomicCycleClassifier(supervised=False)
        
        # Entraînement du modèle
        classifier.fit(self.macro_data)
        
        # Prédiction des phases
        phases = classifier.predict(self.macro_data)
        
        # Vérification des dimensions
        self.assertEqual(len(phases), len(self.macro_data))
        
        # Vérification que les phases sont parmi les valeurs attendues
        for phase in phases:
            self.assertIn(phase, classifier.cycle_labels.values())
        
        # Vérification que toutes les phases sont représentées
        unique_phases = set(phases)
        self.assertTrue(len(unique_phases) >= 3)  # Au moins 3 phases différentes identifiées
    
    def test_save_load_model(self):
        """Test des méthodes de sauvegarde et chargement du modèle."""
        # Création d'un fichier temporaire pour la sauvegarde
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Initialisation et entraînement du classifieur
            classifier = EconomicCycleClassifier(supervised=False)
            classifier.fit(self.macro_data)
            
            # Prédiction des phases avant sauvegarde
            phases_before = classifier.predict(self.macro_data.iloc[:10])
            
            # Sauvegarde du modèle
            classifier.save_model(model_path)
            
            # Vérification que le fichier existe
            self.assertTrue(os.path.exists(model_path))
            
            # Chargement du modèle
            loaded_classifier = EconomicCycleClassifier.load_model(model_path)
            
            # Vérification que le modèle chargé est du même type
            self.assertEqual(classifier.supervised, loaded_classifier.supervised)
            
            # Vérification que les prédictions sont identiques
            phases_after = loaded_classifier.predict(self.macro_data.iloc[:10])
            np.testing.assert_array_equal(phases_before, phases_after)
        
        finally:
            # Suppression du fichier temporaire
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_plot_functions(self):
        """Test des méthodes de visualisation."""
        # Initialisation et entraînement du classifieur
        classifier = EconomicCycleClassifier(supervised=False)
        classifier.fit(self.macro_data)
        
        # Test de plot_cycle_distribution
        fig = classifier.plot_cycle_distribution(self.macro_data)
        self.assertIsNotNone(fig)
        
        # Test de plot_cycle_characteristics
        fig = classifier.plot_cycle_characteristics(self.macro_data)
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
