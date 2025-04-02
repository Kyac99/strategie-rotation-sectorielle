"""
Tests unitaires pour le module de backtesting.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajout du répertoire racine au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.backtest_engine import BacktestEngine, simple_momentum_strategy


class TestBacktestEngine(unittest.TestCase):
    """Tests pour la classe BacktestEngine."""

    def setUp(self):
        """Initialisation avant chaque test."""
        # Création de données de test
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        
        # Création de données sectorielles de test
        np.random.seed(42)  # Pour la reproductibilité
        self.sector_data = pd.DataFrame(index=dates)
        
        # Ajout de quelques secteurs fictifs
        sectors = ['XLY', 'XLP', 'XLE', 'SPY']
        for sector in sectors:
            # Prix simulés avec une légère tendance haussière
            self.sector_data[sector] = 100 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates)))
        
        # Création de données macroéconomiques de test
        self.macro_data = pd.DataFrame(index=dates)
        self.macro_data['GDPC1'] = 100 + np.cumsum(np.random.normal(0.0002, 0.002, len(dates)))
        self.macro_data['UNRATE'] = 5 + np.random.normal(0, 0.1, len(dates))
        self.macro_data['CPIAUCSL'] = 100 + np.cumsum(np.random.normal(0.0001, 0.001, len(dates)))
        
        # Calcul de quelques indicateurs dérivés
        self.macro_data['GDPC1_YOY'] = self.macro_data['GDPC1'].pct_change(252) * 100
        self.macro_data['UNRATE_YOY'] = self.macro_data['UNRATE'].diff(252)
        self.macro_data['CPIAUCSL_YOY'] = self.macro_data['CPIAUCSL'].pct_change(252) * 100
        
        # Création du moteur de backtesting
        self.backtest = BacktestEngine(
            sector_data=self.sector_data,
            macro_data=self.macro_data,
            benchmark='SPY',
            risk_free_rate=0.02
        )
    
    def test_prepare_data(self):
        """Test de la méthode prepare_data."""
        # Préparation des données
        sector_bt, macro_bt = self.backtest.prepare_data(
            start_date='2020-02-01',
            end_date='2020-11-30',
            frequency='M'
        )
        
        # Vérification des dimensions
        self.assertEqual(len(sector_bt), 10)  # 10 mois de février à novembre
        self.assertEqual(len(macro_bt), 10)
        
        # Vérification des colonnes
        for sector in ['XLY', 'XLP', 'XLE', 'SPY']:
            self.assertIn(sector, sector_bt.columns)
        
        for indicator in ['GDPC1', 'UNRATE', 'CPIAUCSL']:
            self.assertIn(indicator, macro_bt.columns)
    
    def test_run_simple_strategy(self):
        """Test de la méthode run_simple_strategy avec une stratégie simple."""
        # Exécution du backtest
        results, allocations = self.backtest.run_simple_strategy(
            strategy_func=simple_momentum_strategy,
            strategy_params={'lookback_periods': 20, 'top_n': 2},
            start_date='2020-03-01',
            end_date='2020-10-31',
            frequency='M',
            initial_capital=10000,
            transaction_cost=0.001
        )
        
        # Vérification des dimensions
        self.assertEqual(len(results), 8)  # 8 mois de mars à octobre
        self.assertEqual(len(allocations), 8)
        
        # Vérification des colonnes des résultats
        for col in ['Portfolio_Value', 'Portfolio_Return', 'Benchmark_Value', 'Benchmark_Return', 'Transaction_Costs']:
            self.assertIn(col, results.columns)
        
        # Vérification des allocations
        for sector in ['XLY', 'XLP', 'XLE']:
            self.assertIn(sector, allocations.columns)
        
        # Vérification que les allocations somment à 1 (ou 0)
        for date, row in allocations.iterrows():
            self.assertTrue(abs(row.sum() - 1.0) < 0.01 or abs(row.sum()) < 0.01)
    
    def test_calculate_performance_metrics(self):
        """Test de la méthode calculate_performance_metrics."""
        # Création de résultats de test
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
        results = pd.DataFrame(index=dates)
        
        # Simulation de la valeur du portefeuille et du benchmark
        initial_value = 10000
        results['Portfolio_Value'] = initial_value * (1 + np.cumsum(np.random.normal(0.01, 0.03, len(dates))))
        results['Benchmark_Value'] = initial_value * (1 + np.cumsum(np.random.normal(0.005, 0.025, len(dates))))
        
        # Calcul des rendements
        results['Portfolio_Return'] = results['Portfolio_Value'].pct_change()
        results['Benchmark_Return'] = results['Benchmark_Value'].pct_change()
        
        # Ajout des coûts de transaction
        results['Transaction_Costs'] = initial_value * 0.001 * np.ones(len(dates))
        
        # Calcul des métriques
        metrics = self.backtest.calculate_performance_metrics(results)
        
        # Vérification de la présence des métriques principales
        for metric in ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']:
            self.assertIn(metric, metrics)
        
        # Vérification des valeurs calculées
        self.assertTrue(isinstance(metrics['total_return'], float))
        self.assertTrue(isinstance(metrics['annualized_return'], float))
        self.assertTrue(isinstance(metrics['volatility'], float))
        self.assertTrue(isinstance(metrics['sharpe_ratio'], float))
        self.assertTrue(isinstance(metrics['max_drawdown'], float))
        
        # Vérification des bornes des métriques
        self.assertTrue(-1.0 <= metrics['max_drawdown'] <= 0.0)  # Le drawdown est négatif ou nul
        self.assertTrue(metrics['volatility'] >= 0.0)  # La volatilité est positive


if __name__ == '__main__':
    unittest.main()
