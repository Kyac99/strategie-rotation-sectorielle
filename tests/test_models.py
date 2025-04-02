"""
Tests unitaires pour les modèles.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile

# Ajout du répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.economic_cycle_classifier import EconomicCycleClassifier
from src.models.sector_selector import SectorSelector


class TestEconomicCycleClassifier(unittest.TestCase):
    """
    Tests pour le classifieur de cycles économiques.
    """
    
    def setUp(self):
        """Configuration des tests."""
        # Création d'un DataFrame de test pour les données macroéconomiques
        dates = pd.date_range(start='2022-01-01', periods=24, freq='M')
        
        # Création de données avec des tendances claires pour chaque indicateur
        gdp_data = np.concatenate([
            np.linspace(2, 4, 8),    # Croissance (phase 1)
            np.linspace(4, 0, 8),    # Décroissance (phase 2)
            np.linspace(0, 2, 8)     # Reprise (phase 3)
        ])
        
        inflation_data = np.concatenate([
            np.linspace(1, 3, 8),    # Inflation croissante (phase 1)
            np.linspace(3, 2, 8),    # Inflation décroissante (phase 2)
            np.linspace(2, 1, 8)     # Inflation faible (phase 3)
        ])
        
        unemployment_data = np.concatenate([
            np.linspace(4, 3, 8),    # Chômage décroissant (phase 1)
            np.linspace(3, 7, 8),    # Chômage croissant (phase 2)
            np.linspace(7, 5, 8)     # Chômage décroissant (phase 3)
        ])
        
        self.test_data = pd.DataFrame({
            'GDPC1_YOY': gdp_data,
            'CPIAUCSL_YOY': inflation_data,
            'UNRATE': unemployment_data
        }, index=dates)
        
        # Création du classifieur
        self.classifier = EconomicCycleClassifier(supervised=False)
    
    def test_initialization(self):
        """Test de l'initialisation du classifieur."""
        # Vérification que le classifieur est correctement initialisé
        self.assertFalse(self.classifier.supervised)
        self.assertIsNotNone(self.classifier.scaler)
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.cycle_labels)
        self.assertIsNotNone(self.classifier.key_indicators)
    
    def test_fit_and_predict(self):
        """Test de l'entraînement et de la prédiction."""
        # Entraînement du modèle
        self.classifier.fit(self.test_data)
        
        # Prédiction des phases
        phases = self.classifier.predict(self.test_data)
        
        # Vérification que les phases sont une Series
        self.assertIsInstance(phases, pd.Series)
        
        # Vérification que le nombre de phases correspond au nombre d'observations
        self.assertEqual(len(phases), len(self.test_data))
        
        # Vérification que les phases sont des chaînes de caractères correspondant aux labels
        for phase in phases:
            self.assertIn(phase, self.classifier.cycle_labels.values())
    
    def test_save_and_load_model(self):
        """Test de la sauvegarde et du chargement du modèle."""
        # Entraînement du modèle
        self.classifier.fit(self.test_data)
        
        # Sauvegarde du modèle dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp_file:
            temp_path = temp_file.name
        
        self.classifier.save_model(temp_path)
        
        # Chargement du modèle
        loaded_classifier = EconomicCycleClassifier.load_model(temp_path)
        
        # Vérification que le modèle chargé est un EconomicCycleClassifier
        self.assertIsInstance(loaded_classifier, EconomicCycleClassifier)
        
        # Vérification que les attributs sont préservés
        self.assertEqual(loaded_classifier.supervised, self.classifier.supervised)
        self.assertEqual(loaded_classifier.cycle_labels, self.classifier.cycle_labels)
        self.assertEqual(loaded_classifier.key_indicators, self.classifier.key_indicators)
        
        # Nettoyage
        os.unlink(temp_path)
    
    def test_plot_cycle_distribution(self):
        """Test de la visualisation de la distribution des cycles."""
        # Entraînement du modèle
        self.classifier.fit(self.test_data)
        
        # Génération du graphique
        fig = self.classifier.plot_cycle_distribution(self.test_data)
        
        # Vérification que la figure est retournée
        self.assertIsNotNone(fig)
    
    def test_plot_cycle_characteristics(self):
        """Test de la visualisation des caractéristiques des cycles."""
        # Entraînement du modèle
        self.classifier.fit(self.test_data)
        
        # Génération du graphique
        fig = self.classifier.plot_cycle_characteristics(self.test_data)
        
        # Vérification que la figure est retournée
        self.assertIsNotNone(fig)


class TestSectorSelector(unittest.TestCase):
    """
    Tests pour le sélecteur de secteurs.
    """
    
    def setUp(self):
        """Configuration des tests."""
        # Mock du classifieur de cycles
        self.mock_classifier = MagicMock()
        self.mock_classifier.predict.return_value = pd.Series(
            ['Expansion'] * 12,
            index=pd.date_range(start='2022-01-01', periods=12, freq='M')
        )
        
        # Création de données macro de test
        self.macro_data = pd.DataFrame({
            'GDPC1_YOY': np.linspace(2, 3, 12),
            'CPIAUCSL_YOY': np.linspace(1, 2, 12),
            'UNRATE': np.linspace(4, 3, 12)
        }, index=pd.date_range(start='2022-01-01', periods=12, freq='M'))
        
        # Création de données sectorielles de test
        sectors = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLRE', 'XLC']
        
        # Prix de base pour chaque secteur
        base_prices = {
            'XLY': 100, 'XLP': 70, 'XLE': 50, 'XLF': 40, 'XLV': 130,
            'XLI': 90, 'XLB': 80, 'XLK': 150, 'XLU': 60, 'XLRE': 110, 'XLC': 120
        }
        
        # Création des prix avec tendances différentes
        sector_data = {}
        dates = pd.date_range(start='2022-01-01', periods=12, freq='M')
        
        for sector in sectors:
            base = base_prices[sector]
            # Tendance aléatoire pour chaque secteur
            trend = np.random.choice([1.0, 1.5, 0.8, 1.2])
            prices = np.array([base * (1 + 0.02 * i * trend) for i in range(12)])
            sector_data[sector] = prices
        
        self.sector_data = pd.DataFrame(sector_data, index=dates)
        
        # Création du sélecteur
        self.selector = SectorSelector(cycle_classifier=self.mock_classifier)
    
    def test_initialization(self):
        """Test de l'initialisation du sélecteur."""
        # Vérification que le sélecteur est correctement initialisé
        self.assertEqual(self.selector.cycle_classifier, self.mock_classifier)
        self.assertIsNotNone(self.selector.sector_cycle_performance)
        self.assertIsNotNone(self.selector.momentum_factors)
    
    def test_identify_current_cycle(self):
        """Test de l'identification du cycle actuel."""
        # Identification du cycle
        cycle = self.selector.identify_current_cycle(self.macro_data)
        
        # Vérification que le classifieur a été appelé
        self.mock_classifier.predict.assert_called_once()
        
        # Vérification que le cycle retourné est 'Expansion'
        self.assertEqual(cycle, 'Expansion')
    
    def test_calculate_momentum_score(self):
        """Test du calcul du score de momentum."""
        # Création de données de test avec des rendements pour les différentes périodes
        data = self.sector_data.copy()
        
        # Création manuelle de colonnes de rendement
        for sector in data.columns:
            data[f"{sector}_monthly"] = 0.5  # Rendement mensuel fictif
            data[f"{sector}_1_month"] = 1.0  # Rendement sur 1 mois fictif
            data[f"{sector}_3_months"] = 3.0  # Rendement sur 3 mois fictif
            data[f"{sector}_6_months"] = 6.0  # Rendement sur 6 mois fictif
            data[f"{sector}_12_months"] = 12.0  # Rendement sur 12 mois fictif
        
        # Calcul du score de momentum
        momentum_scores = self.selector.calculate_momentum_score(data)
        
        # Vérification que les scores sont une Series
        self.assertIsInstance(momentum_scores, pd.Series)
        
        # Vérification que les scores sont calculés pour tous les secteurs
        self.assertEqual(len(momentum_scores), len(self.sector_data.columns))
        
        # Vérification que les scores sont égaux car nous avons mis les mêmes rendements fictifs pour tous les secteurs
        expected_score = 1.0 * 0.2 + 3.0 * 0.3 + 6.0 * 0.3 + 12.0 * 0.2  # Pondération selon momentum_factors
        for score in momentum_scores:
            self.assertAlmostEqual(score, expected_score)
    
    def test_select_sectors(self):
        """Test de la sélection des secteurs."""
        # Création de données de test avec des rendements pour les différentes périodes
        data = self.sector_data.copy()
        
        # Création manuelle de colonnes de rendement avec des valeurs différentes pour différencier les secteurs
        for i, sector in enumerate(data.columns):
            data[f"{sector}_monthly"] = 0.5 + 0.1 * i  # Rendement mensuel croissant
            data[f"{sector}_1_month"] = 1.0 + 0.2 * i  # Rendement sur 1 mois croissant
            data[f"{sector}_3_months"] = 3.0 + 0.3 * i  # Rendement sur 3 mois croissant
            data[f"{sector}_6_months"] = 6.0 + 0.4 * i  # Rendement sur 6 mois croissant
            data[f"{sector}_12_months"] = 12.0 + 0.5 * i  # Rendement sur 12 mois croissant
        
        # Sélection des secteurs
        weights = self.selector.select_sectors(
            self.macro_data, data, num_sectors=3, momentum_weight=0.5
        )
        
        # Vérification que les poids sont une Series
        self.assertIsInstance(weights, pd.Series)
        
        # Vérification qu'il y a 3 secteurs sélectionnés (num_sectors=3)
        self.assertEqual(sum(weights > 0), 3)
        
        # Vérification que la somme des poids est égale à 1
        self.assertAlmostEqual(weights.sum(), 1.0)
    
    def test_backtest_strategy(self):
        """Test du backtesting de la stratégie."""
        # Création de données de test avec des rendements pour les différentes périodes
        data = self.sector_data.copy()
        
        # Création manuelle de colonnes de rendement
        for sector in data.columns:
            data[f"{sector}_daily"] = 0.001  # Rendement journalier fictif
            data[f"{sector}_monthly"] = 0.02  # Rendement mensuel fictif
        
        # Ajout du benchmark
        data['SPY'] = 200.0  # Prix fictif du SPY
        data['SPY_daily'] = 0.0005  # Rendement journalier fictif du SPY
        
        # Backtesting de la stratégie
        results, allocations = self.selector.backtest_strategy(
            self.macro_data, data,
            start_date='2022-01-01', end_date='2022-12-31',
            rebalance_freq='M', num_sectors=3, momentum_weight=0.5
        )
        
        # Vérification que les résultats sont un DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        
        # Vérification que les allocations sont un DataFrame
        self.assertIsInstance(allocations, pd.DataFrame)
        
        # Vérification que les résultats contiennent les colonnes attendues
        expected_columns = ['Portfolio_Value', 'Portfolio_Return', 'Benchmark_Value', 'Benchmark_Return', 'Transaction_Costs']
        self.assertEqual(set(results.columns), set(expected_columns))
        
        # Vérification que les allocations sont bien définies pour les secteurs
        self.assertEqual(set(allocations.columns), set(self.sector_data.columns))
    
    def test_calculate_performance_metrics(self):
        """Test du calcul des métriques de performance."""
        # Création de données de test pour les résultats
        index = pd.date_range(start='2022-01-01', periods=12, freq='M')
        
        # Croissance constante de 1% par mois
        portfolio_values = np.array([10000 * (1.01) ** i for i in range(12)])
        
        # Croissance constante de 0.5% par mois
        benchmark_values = np.array([10000 * (1.005) ** i for i in range(12)])
        
        # Rendements mensuels
        portfolio_returns = np.array([0.01] * 12)
        benchmark_returns = np.array([0.005] * 12)
        
        results = pd.DataFrame({
            'Portfolio_Value': portfolio_values,
            'Benchmark_Value': benchmark_values,
            'Portfolio_Return': portfolio_returns,
            'Benchmark_Return': benchmark_returns,
            'Transaction_Costs': np.array([5.0] * 12)
        }, index=index)
        
        # Calcul des métriques
        metrics = self.selector.calculate_performance_metrics(results)
        
        # Vérification que les métriques sont un dictionnaire
        self.assertIsInstance(metrics, dict)
        
        # Vérification que les métriques contiennent les clés attendues
        expected_keys = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Vérification des valeurs calculées
        self.assertAlmostEqual(metrics['total_return'], 0.1268, places=4)  # (1.01)^12 - 1 ≈ 0.1268
        self.assertAlmostEqual(metrics['annual_return'], 0.1268, places=4)  # Période d'exactement 1 an
        self.assertEqual(metrics['volatility'], 0.0)  # Rendements constants, donc volatilité nulle


if __name__ == '__main__':
    unittest.main()
