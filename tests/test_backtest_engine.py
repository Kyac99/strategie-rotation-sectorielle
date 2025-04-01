"""
Tests unitaires pour le moteur de backtesting.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

# Ajout du répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.backtest_engine import BacktestEngine, simple_momentum_strategy, cycle_based_strategy


class TestBacktestEngine(unittest.TestCase):
    """
    Classe de tests pour BacktestEngine.
    """

    def setUp(self):
        """
        Configuration des tests.
        """
        # Création d'un DataFrame de test pour les données sectorielles
        index = pd.date_range(start='2020-01-01', periods=24, freq='M')
        
        # Création de données de prix synthétiques pour 5 secteurs
        sectors = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV']
        sector_data = {}
        
        # Données de base avec une tendance haussière
        base_prices = 100 + np.arange(24) * 2
        
        # Variation des prix pour chaque secteur
        for i, sector in enumerate(sectors):
            # Chaque secteur a un comportement légèrement différent
            multiplier = 0.9 + i * 0.1  # 0.9, 1.0, 1.1, 1.2, 1.3
            noise = np.random.normal(0, 5, 24)  # Bruit aléatoire
            sector_data[sector] = base_prices * multiplier + noise
        
        # Ajout du benchmark
        sector_data['SPY'] = base_prices * 1.1 + np.random.normal(0, 3, 24)
        
        self.sector_data = pd.DataFrame(sector_data, index=index)
        
        # Création d'un DataFrame de test pour les données macroéconomiques
        macro_data = {
            'GDPC1_YOY': np.concatenate([
                np.linspace(3.0, 4.0, 8),      # Expansion
                np.linspace(4.0, 0.5, 8),      # Ralentissement
                np.linspace(0.5, -2.0, 8)      # Récession
            ])[:24],
            'UNRATE': np.concatenate([
                np.linspace(4.0, 3.5, 8),      # Expansion
                np.linspace(3.5, 5.0, 8),      # Ralentissement
                np.linspace(5.0, 7.0, 8)       # Récession
            ])[:24],
            'CPIAUCSL_YOY': np.concatenate([
                np.linspace(2.0, 2.5, 8),      # Expansion
                np.linspace(2.5, 3.5, 8),      # Ralentissement
                np.linspace(3.5, 1.5, 8)       # Récession
            ])[:24]
        }
        
        self.macro_data = pd.DataFrame(macro_data, index=index)
        
        # Création du moteur de backtesting
        self.backtest = BacktestEngine(
            sector_data=self.sector_data,
            macro_data=self.macro_data,
            benchmark='SPY',
            risk_free_rate=0.02
        )
        
        # Mock du classifieur de cycles économiques
        self.mock_classifier = MagicMock()
        self.mock_classifier.predict.return_value = pd.Series(
            ['Expansion'] * 8 + ['Ralentissement'] * 8 + ['Récession'] * 8,
            index=index
        )

    def test_init(self):
        """
        Test de l'initialisation du moteur de backtesting.
        """
        self.assertEqual(self.backtest.sector_data.equals(self.sector_data), True)
        self.assertEqual(self.backtest.macro_data.equals(self.macro_data), True)
        self.assertEqual(self.backtest.benchmark, 'SPY')
        self.assertEqual(self.backtest.risk_free_rate, 0.02)

    def test_prepare_data(self):
        """
        Test de la préparation des données pour le backtest.
        """
        # Préparation des données avec des dates spécifiques
        start_date = '2020-03-01'
        end_date = '2020-10-01'
        
        sector_bt, macro_bt = self.backtest.prepare_data(
            start_date=start_date,
            end_date=end_date,
            frequency='M'
        )
        
        # Vérification des dates
        self.assertEqual(sector_bt.index[0], pd.Timestamp(start_date))
        self.assertTrue(sector_bt.index[-1] <= pd.Timestamp(end_date))
        
        # Vérification que toutes les colonnes sont présentes
        for col in self.sector_data.columns:
            self.assertIn(col, sector_bt.columns)
        
        # Vérification des données macroéconomiques
        self.assertEqual(macro_bt.index[0], pd.Timestamp(start_date))
        self.assertTrue(macro_bt.index[-1] <= pd.Timestamp(end_date))

    def test_run_simple_strategy(self):
        """
        Test de l'exécution d'une stratégie simple.
        """
        # Définition d'une stratégie de test simple
        def test_strategy(sector_data, macro_data, current_date, param1=1, param2=2):
            # Stratégie qui alloue également entre les deux premiers secteurs
            allocations = pd.Series(0, index=sector_data.columns)
            sectors = [col for col in sector_data.columns if col != 'SPY']
            if len(sectors) >= 2:
                allocations[sectors[0]] = 0.5
                allocations[sectors[1]] = 0.5
            return allocations
        
        # Exécution de la stratégie
        results, allocations = self.backtest.run_simple_strategy(
            strategy_func=test_strategy,
            strategy_params={'param1': 1, 'param2': 2},
            start_date='2020-02-01',
            end_date='2020-10-01',
            frequency='M'
        )
        
        # Vérification des résultats
        self.assertIsNotNone(results)
        self.assertIsNotNone(allocations)
        
        # Vérification des colonnes des résultats
        self.assertIn('Portfolio_Value', results.columns)
        self.assertIn('Portfolio_Return', results.columns)
        self.assertIn('Benchmark_Value', results.columns)
        self.assertIn('Benchmark_Return', results.columns)
        
        # Vérification des allocations
        self.assertEqual(allocations.iloc[0]['XLY'], 0.5)
        self.assertEqual(allocations.iloc[0]['XLP'], 0.5)
        
        # Vérification que la valeur du portefeuille change au fil du temps
        self.assertNotEqual(results['Portfolio_Value'].iloc[0], results['Portfolio_Value'].iloc[-1])

    def test_calculate_performance_metrics(self):
        """
        Test du calcul des métriques de performance.
        """
        # Création de données de résultats synthétiques
        index = pd.date_range(start='2020-01-01', periods=12, freq='M')
        
        # Portefeuille avec une croissance linéaire
        portfolio_values = 10000 * (1 + np.linspace(0, 0.2, 12))
        portfolio_returns = np.concatenate([[0], np.diff(portfolio_values) / portfolio_values[:-1]])
        
        # Benchmark avec une croissance moins rapide
        benchmark_values = 10000 * (1 + np.linspace(0, 0.1, 12))
        benchmark_returns = np.concatenate([[0], np.diff(benchmark_values) / benchmark_values[:-1]])
        
        results = pd.DataFrame({
            'Portfolio_Value': portfolio_values,
            'Portfolio_Return': portfolio_returns,
            'Benchmark_Value': benchmark_values,
            'Benchmark_Return': benchmark_returns
        }, index=index)
        
        # Calcul des métriques
        metrics = self.backtest.calculate_performance_metrics(results)
        
        # Vérification des métriques principales
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # Vérification des valeurs
        self.assertAlmostEqual(metrics['total_return'], 0.2, delta=0.01)
        self.assertTrue(metrics['sharpe_ratio'] > 0)
        self.assertTrue(metrics['volatility'] > 0)

    def test_run_strategy_optimization(self):
        """
        Test de l'optimisation des paramètres d'une stratégie.
        """
        # Définition d'une stratégie simple pour le test
        def test_strategy(sector_data, macro_data, current_date, param=1):
            # Stratégie qui alloue uniquement au premier secteur
            allocations = pd.Series(0, index=sector_data.columns)
            sectors = [col for col in sector_data.columns if col != 'SPY']
            if len(sectors) > 0:
                allocations[sectors[0]] = 1.0
            return allocations
        
        # Définition d'une grille de paramètres simple
        param_grid = {'param': [1, 2, 3]}
        
        # Exécution de l'optimisation
        best_params, optimization_results = self.backtest.run_strategy_optimization(
            strategy_func=test_strategy,
            param_grid=param_grid,
            start_date='2020-02-01',
            end_date='2020-08-01',
            frequency='M'
        )
        
        # Vérification des résultats
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(optimization_results)
        
        # Vérification que le meilleur paramètre est dans la grille
        self.assertIn(best_params['param'], param_grid['param'])
        
        # Vérification que tous les paramètres ont été testés
        self.assertEqual(len(optimization_results), len(param_grid['param']))

    def test_generate_performance_report(self):
        """
        Test de la génération d'un rapport de performance.
        """
        # Exécution d'une stratégie simple pour obtenir des résultats
        results, allocations = self.backtest.run_simple_strategy(
            strategy_func=simple_momentum_strategy,
            strategy_params={'lookback_periods': 3, 'top_n': 2},
            start_date='2020-02-01',
            end_date='2020-10-01',
            frequency='M'
        )
        
        # Génération du rapport sans sauvegarde
        report = self.backtest.generate_performance_report(results, allocations)
        
        # Vérification des sections du rapport
        self.assertIn('summary', report)
        self.assertIn('detailed_metrics', report)
        self.assertIn('allocation_summary', report)
        self.assertIn('figures', report)
        
        # Vérification des métriques dans le résumé
        summary = report['summary']
        self.assertIn('total_return', summary)
        self.assertIn('annualized_return', summary)
        self.assertIn('volatility', summary)
        self.assertIn('sharpe_ratio', summary)
        
        # Test de la sauvegarde du rapport
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            report_path = tmp.name
        
        try:
            # Génération du rapport avec sauvegarde
            self.backtest.generate_performance_report(results, allocations, output_file=report_path)
            
            # Vérification que le fichier a été créé
            self.assertTrue(os.path.exists(report_path))
            
        finally:
            # Nettoyage
            if os.path.exists(report_path):
                os.remove(report_path)

    def test_simple_momentum_strategy(self):
        """
        Test de la stratégie simple basée sur le momentum.
        """
        # Exécution de la stratégie
        allocations = simple_momentum_strategy(
            self.sector_data,
            None,  # Pas besoin de données macro
            self.sector_data.index[-1],  # Date actuelle
            lookback_periods=3,
            top_n=2
        )
        
        # Vérification du format des allocations
        self.assertIsInstance(allocations, pd.Series)
        self.assertEqual(len(allocations), len(self.sector_data.columns))
        
        # Vérification que la somme des allocations est égale à 1
        self.assertAlmostEqual(allocations.sum(), 1.0, delta=0.01)
        
        # Vérification que seulement top_n secteurs ont une allocation non nulle
        non_zero = allocations[allocations > 0]
        self.assertEqual(len(non_zero), 2)
        
        # Vérification que les allocations sont équitablement réparties
        for alloc in non_zero:
            self.assertAlmostEqual(alloc, 0.5, delta=0.01)

    def test_cycle_based_strategy(self):
        """
        Test de la stratégie basée sur les cycles économiques.
        """
        # Exécution de la stratégie
        allocations = cycle_based_strategy(
            self.sector_data,
            self.macro_data,
            self.sector_data.index[-1],  # Date actuelle
            cycle_classifier=self.mock_classifier,
            top_n=2,
            momentum_weight=0.5
        )
        
        # Vérification du format des allocations
        self.assertIsInstance(allocations, pd.Series)
        self.assertEqual(len(allocations), len(self.sector_data.columns))
        
        # Vérification que la somme des allocations est égale à 1
        self.assertAlmostEqual(allocations.sum(), 1.0, delta=0.01)
        
        # Vérification que seulement top_n secteurs ont une allocation non nulle
        non_zero = allocations[allocations > 0]
        self.assertEqual(len(non_zero), 2)
        
        # Vérification que les allocations sont équitablement réparties
        for alloc in non_zero:
            self.assertAlmostEqual(alloc, 0.5, delta=0.01)
        
        # Vérification que le classifieur a été appelé
        self.mock_classifier.predict.assert_called_once()


if __name__ == '__main__':
    unittest.main()
