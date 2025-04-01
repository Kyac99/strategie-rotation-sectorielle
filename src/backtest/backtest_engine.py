"""
Module de backtesting pour les stratégies de rotation sectorielle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
import sys
from itertools import product

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Moteur de backtesting pour évaluer les stratégies de rotation sectorielle.
    
    Cette classe permet de:
    - Simuler différentes stratégies de rotation sectorielle
    - Comparer les performances avec des benchmarks
    - Calculer des métriques de performance détaillées
    - Générer des rapports d'attribution de performance
    """

    def __init__(self, sector_data, macro_data=None, benchmark='SPY', risk_free_rate=0.02):
        """
        Initialise le moteur de backtesting.

        Args:
            sector_data (pd.DataFrame): Données historiques des secteurs (prix et/ou rendements).
            macro_data (pd.DataFrame, optional): Données macroéconomiques si nécessaires.
            benchmark (str, optional): Ticker du benchmark à utiliser. Par défaut 'SPY'.
            risk_free_rate (float, optional): Taux sans risque annualisé. Par défaut 0.02 (2%).
        """
        self.sector_data = sector_data
        self.macro_data = macro_data
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        
        # Vérification de la présence du benchmark dans les données
        if benchmark not in sector_data.columns:
            logger.warning(f"Le benchmark {benchmark} n'est pas dans les données. Utilisation de la moyenne des secteurs comme approximation.")
            
        logger.info(f"BacktestEngine initialisé avec {len(sector_data)} observations de données sectorielles.")
    
    def prepare_data(self, start_date=None, end_date=None, frequency='M'):
        """
        Prépare les données pour le backtest.

        Args:
            start_date (str or datetime, optional): Date de début du backtest.
            end_date (str or datetime, optional): Date de fin du backtest.
            frequency (str, optional): Fréquence de rééquilibrage ('D', 'W', 'M', 'Q').
                D=Daily, W=Weekly, M=Monthly, Q=Quarterly.

        Returns:
            tuple: (sector_data, macro_data) préparés pour le backtest.
        """
        # Définition des dates par défaut si non spécifiées
        if start_date is None:
            start_date = self.sector_data.index[0]
        if end_date is None:
            end_date = self.sector_data.index[-1]
        
        # Conversion en timestamps si nécessaire
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Filtrage des données selon la période
        sector_bt = self.sector_data.loc[start_date:end_date].copy()
        
        # Rééchantillonnage selon la fréquence si nécessaire
        if frequency != 'D':
            sector_bt = sector_bt.resample(frequency).last()
        
        # Traitement des données macroéconomiques si disponibles
        if self.macro_data is not None:
            macro_bt = self.macro_data.loc[start_date:end_date].copy()
            if frequency != 'D':
                macro_bt = macro_bt.resample(frequency).last()
        else:
            macro_bt = None
        
        logger.info(f"Données préparées pour le backtest: {len(sector_bt)} observations de {start_date} à {end_date} avec fréquence {frequency}.")
        return sector_bt, macro_bt
    
    def run_simple_strategy(self, strategy_func, strategy_params, start_date=None, end_date=None, 
                           frequency='M', initial_capital=10000, transaction_cost=0.0005):
        """
        Exécute un backtest pour une stratégie simple.

        Args:
            strategy_func (function): Fonction qui génère les allocations à chaque période.
            strategy_params (dict): Paramètres de la stratégie.
            start_date (str or datetime, optional): Date de début du backtest.
            end_date (str or datetime, optional): Date de fin du backtest.
            frequency (str, optional): Fréquence de rééquilibrage ('D', 'W', 'M', 'Q').
            initial_capital (float, optional): Capital initial. Par défaut 10000.
            transaction_cost (float, optional): Coût de transaction par transaction (%).
                Par défaut 0.0005 (0.05%).

        Returns:
            tuple: (results, allocations) du backtest.
        """
        # Préparation des données
        sector_bt, macro_bt = self.prepare_data(start_date, end_date, frequency)
        
        # Extraction des tickers sectoriels (en ignorant les colonnes spéciales)
        sector_columns = [col for col in sector_bt.columns if '_' not in col]
        
        # Initialisation des résultats
        dates = sector_bt.index
        portfolio_values = pd.Series(initial_capital, index=dates)
        allocations = pd.DataFrame(0, index=dates, columns=sector_columns)
        transaction_costs = pd.Series(0, index=dates)
        
        # Exécution du backtest
        previous_allocation = pd.Series(0, index=sector_columns)
        
        for i in range(1, len(dates)):
            current_date = dates[i]
            previous_date = dates[i-1]
            
            # Données disponibles jusqu'à la date précédente pour prendre la décision
            available_sector_data = sector_bt.iloc[:i].copy()
            if macro_bt is not None:
                available_macro_data = macro_bt.iloc[:i].copy()
            else:
                available_macro_data = None
            
            # Calcul des allocations selon la stratégie
            try:
                current_allocation = strategy_func(
                    available_sector_data, available_macro_data, 
                    current_date, **strategy_params
                )
                
                # Normalisation des allocations si nécessaire
                if current_allocation.sum() > 0:
                    current_allocation = current_allocation / current_allocation.sum()
                
                # Enregistrement des allocations
                for sector in current_allocation.index:
                    if sector in allocations.columns:
                        allocations.loc[current_date, sector] = current_allocation[sector]
            except Exception as e:
                logger.error(f"Erreur lors du calcul des allocations à la date {current_date}: {e}")
                # En cas d'erreur, conserver l'allocation précédente
                current_allocation = previous_allocation
            
            # Calcul du coût de transaction
            if i > 1:
                # Différence d'allocation par rapport à la période précédente
                allocation_diff = (current_allocation - previous_allocation).abs().sum() / 2
                cost = allocation_diff * transaction_cost * portfolio_values[previous_date]
                transaction_costs[current_date] = cost
            else:
                # Premier rééquilibrage
                allocation_diff = current_allocation.sum()
                cost = allocation_diff * transaction_cost * initial_capital
                transaction_costs[current_date] = cost
            
            # Mise à jour de la valeur du portefeuille
            if i > 0:
                # Calcul des rendements des secteurs entre la date précédente et la date actuelle
                sector_returns = {}
                for sector in sector_columns:
                    if sector in sector_bt.columns:
                        prev_price = sector_bt.loc[previous_date, sector]
                        curr_price = sector_bt.loc[current_date, sector]
                        sector_returns[sector] = curr_price / prev_price - 1
                
                # Calcul du rendement du portefeuille
                portfolio_return = sum(previous_allocation[sector] * sector_returns.get(sector, 0) 
                                      for sector in previous_allocation.index)
                
                # Mise à jour de la valeur du portefeuille (en tenant compte des coûts de transaction)
                portfolio_values[current_date] = portfolio_values[previous_date] * (1 + portfolio_return) - transaction_costs[current_date]
            
            # Mise à jour de l'allocation pour la prochaine itération
            previous_allocation = current_allocation
        
        # Calcul des rendements du portefeuille
        portfolio_returns = portfolio_values.pct_change()
        
        # Calcul des rendements du benchmark s'il est disponible
        if self.benchmark in sector_bt.columns:
            benchmark_prices = sector_bt[self.benchmark]
            benchmark_values = initial_capital * (benchmark_prices / benchmark_prices.iloc[0])
            benchmark_returns = benchmark_values.pct_change()
        else:
            # Utilisation de la moyenne des secteurs comme approximation
            avg_prices = sector_bt[sector_columns].mean(axis=1)
            benchmark_values = initial_capital * (avg_prices / avg_prices.iloc[0])
            benchmark_returns = benchmark_values.pct_change()
        
        # Compilation des résultats
        results = pd.DataFrame({
            'Portfolio_Value': portfolio_values,
            'Portfolio_Return': portfolio_returns,
            'Benchmark_Value': benchmark_values,
            'Benchmark_Return': benchmark_returns,
            'Transaction_Costs': transaction_costs
        })
        
        return results, allocations
    
    def run_strategy_optimization(self, strategy_func, param_grid, start_date=None, end_date=None,
                                 frequency='M', initial_capital=10000, transaction_cost=0.0005,
                                 metric='sharpe_ratio'):
        """
        Optimise les paramètres d'une stratégie par grid search.

        Args:
            strategy_func (function): Fonction qui génère les allocations à chaque période.
            param_grid (dict): Grille de paramètres à tester (format sklearn).
            start_date (str or datetime, optional): Date de début du backtest.
            end_date (str or datetime, optional): Date de fin du backtest.
            frequency (str, optional): Fréquence de rééquilibrage.
            initial_capital (float, optional): Capital initial.
            transaction_cost (float, optional): Coût de transaction par transaction (%).
            metric (str, optional): Métrique à optimiser ('sharpe_ratio', 'sortino_ratio',
                                  'calmar_ratio', 'total_return').

        Returns:
            tuple: (best_params, optimization_results) de l'optimisation.
        """
        # Génération de toutes les combinaisons de paramètres
        param_names = list(param_grid.keys())
        param_values = list(product(*[param_grid[name] for name in param_names]))
        
        # Liste pour stocker les résultats
        optimization_results = []
        
        # Exécution du backtest pour chaque combinaison de paramètres
        for params in param_values:
            strategy_params = {name: value for name, value in zip(param_names, params)}
            
            # Exécution du backtest
            results, _ = self.run_simple_strategy(
                strategy_func, 
                strategy_params, 
                start_date, end_date,
                frequency, initial_capital, 
                transaction_cost
            )
            
            # Calcul des métriques de performance
            performance_metrics = self.calculate_performance_metrics(results)
            
            # Ajout des paramètres et de la métrique à optimiser
            strategy_params['performance'] = performance_metrics.get(metric, 0)
            optimization_results.append(strategy_params)
        
        # Conversion en DataFrame
        optimization_df = pd.DataFrame(optimization_results)
        
        # Tri selon la métrique à optimiser
        optimization_df = optimization_df.sort_values('performance', ascending=False)
        
        # Meilleurs paramètres
        best_params = optimization_df.iloc[0].drop('performance').to_dict()
        
        return best_params, optimization_df
    
    def calculate_performance_metrics(self, results):
        """
        Calcule les métriques de performance détaillées pour les résultats du backtest.

        Args:
            results (pd.DataFrame): Résultats du backtest avec les colonnes Portfolio_Value,
                                   Portfolio_Return, Benchmark_Value, Benchmark_Return.

        Returns:
            dict: Métriques de performance.
        """
        # Extraction des rendements
        portfolio_returns = results['Portfolio_Return'].dropna()
        benchmark_returns = results['Benchmark_Return'].dropna()
        
        # Période d'analyse
        start_date = results.index[0]
        end_date = results.index[-1]
        total_days = (end_date - start_date).days
        years = total_days / 365.25
        
        # Valeurs de début et de fin
        initial_value = results['Portfolio_Value'].iloc[0]
        final_value = results['Portfolio_Value'].iloc[-1]
        
        # Métriques de base
        total_return = (final_value / initial_value) - 1
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatilité et drawdown
        volatility = portfolio_returns.std() * np.sqrt(252 if 'D' in str(portfolio_returns.index.freq) else 12)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        # Durée du drawdown maximum
        is_drawdown = drawdown < 0
        if is_drawdown.any():
            drawdown_periods = []
            current_period = 0
            for is_dd in is_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            max_drawdown_period = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_drawdown_period = 0
        
        # Ratios de performance
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_return = portfolio_returns - daily_rf
        sharpe_ratio = (excess_return.mean() / excess_return.std()) * np.sqrt(252 if 'D' in str(portfolio_returns.index.freq) else 12)
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252 if 'D' in str(portfolio_returns.index.freq) else 12)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_dev if downside_dev != 0 else float('inf')
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        # Métriques vs benchmark
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252 if 'D' in str(portfolio_returns.index.freq) else 12)
        information_ratio = ((portfolio_returns.mean() - benchmark_returns.mean()) / tracking_error) * np.sqrt(252 if 'D' in str(portfolio_returns.index.freq) else 12) if tracking_error != 0 else 0
        
        beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = (portfolio_returns.mean() - self.risk_free_rate) - beta * (benchmark_returns.mean() - self.risk_free_rate)
        alpha_annualized = alpha * (252 if 'D' in str(portfolio_returns.index.freq) else 12)
        
        # Compilation des métriques
        metrics = {
            'start_date': start_date,
            'end_date': end_date,
            'years': years,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_period': max_drawdown_period,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'alpha_annualized': alpha_annualized,
            'win_rate': (portfolio_returns > 0).mean(),
            'loss_rate': (portfolio_returns < 0).mean(),
            'best_return': portfolio_returns.max(),
            'worst_return': portfolio_returns.min(),
            'avg_win': portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0,
            'avg_loss': portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0,
        }
        
        # Calcul du ratio gain-perte
        if metrics['avg_loss'] != 0:
            metrics['gain_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['gain_loss_ratio'] = float('inf')
        
        # Indicateurs de market timing
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        if up_market.any():
            metrics['up_capture'] = (portfolio_returns[up_market].mean() / benchmark_returns[up_market].mean()) if benchmark_returns[up_market].mean() != 0 else float('inf')
        else:
            metrics['up_capture'] = 0
            
        if down_market.any():
            metrics['down_capture'] = (portfolio_returns[down_market].mean() / benchmark_returns[down_market].mean()) if benchmark_returns[down_market].mean() != 0 else float('inf')
        else:
            metrics['down_capture'] = 0
        
        # Métriques annuelles
        annual_returns = {}
        try:
            portfolio_annual = (1 + portfolio_returns).groupby(portfolio_returns.index.year).prod() - 1
            benchmark_annual = (1 + benchmark_returns).groupby(benchmark_returns.index.year).prod() - 1
            
            for year in portfolio_annual.index:
                annual_returns[year] = {
                    'portfolio': portfolio_annual.get(year, 0),
                    'benchmark': benchmark_annual.get(year, 0),
                    'excess': portfolio_annual.get(year, 0) - benchmark_annual.get(year, 0)
                }
        except Exception as e:
            logger.error(f"Erreur lors du calcul des rendements annuels: {e}")
        
        metrics['annual_returns'] = annual_returns
        
        # Calcul des coûts de transaction
        if 'Transaction_Costs' in results:
            total_transaction_costs = results['Transaction_Costs'].sum()
            metrics['total_transaction_costs'] = total_transaction_costs
            metrics['avg_transaction_costs_per_period'] = total_transaction_costs / len(results)
            metrics['transaction_costs_impact'] = total_transaction_costs / initial_value
        
        return metrics
    
    def generate_performance_report(self, results, allocations, output_file=None):
        """
        Génère un rapport de performance complet pour les résultats du backtest.

        Args:
            results (pd.DataFrame): Résultats du backtest.
            allocations (pd.DataFrame): Allocations du backtest.
            output_file (str, optional): Chemin du fichier de sortie.

        Returns:
            dict: Rapport de performance.
        """
        # Calcul des métriques de performance
        metrics = self.calculate_performance_metrics(results)
        
        # Préparation du rapport
        report = {
            'summary': {
                'start_date': metrics['start_date'].strftime('%Y-%m-%d'),
                'end_date': metrics['end_date'].strftime('%Y-%m-%d'),
                'years': metrics['years'],
                'final_value': metrics['final_value'],
                'total_return': metrics['total_return'],
                'annualized_return': metrics['annualized_return'],
                'volatility': metrics['volatility'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'transaction_costs_impact': metrics.get('transaction_costs_impact', 0)
            },
            'detailed_metrics': metrics,
            'allocation_summary': {
                'average_allocation': allocations.mean().to_dict(),
                'turnover': self._calculate_turnover(allocations),
                'concentration': self._calculate_concentration(allocations)
            }
        }
        
        # Génération des figures pour le rapport
        report['figures'] = self._generate_report_figures(results, allocations, metrics)
        
        # Sauvegarde du rapport si un chemin est spécifié
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            logger.info(f"Rapport de performance sauvegardé dans {output_file}")
        
        return report
    
    def _calculate_turnover(self, allocations):
        """Calcule le turnover des allocations."""
        turnover = []
        for i in range(1, len(allocations)):
            prev_alloc = allocations.iloc[i-1]
            curr_alloc = allocations.iloc[i]
            changes = (curr_alloc - prev_alloc).abs().sum() / 2
            turnover.append(changes)
        return {
            'average': np.mean(turnover),
            'max': np.max(turnover) if turnover else 0,
            'min': np.min(turnover) if turnover else 0
        }
    
    def _calculate_concentration(self, allocations):
        """Calcule la concentration des allocations."""
        # Indice de Herfindahl-Hirschman (HHI)
        hhi = (allocations ** 2).sum(axis=1)
        return {
            'average_hhi': hhi.mean(),
            'max_hhi': hhi.max(),
            'min_hhi': hhi.min(),
            'average_sectors': (allocations > 0).sum(axis=1).mean()
        }
    
    def _generate_report_figures(self, results, allocations, metrics):
        """Génère les figures pour le rapport de performance."""
        figures = {}
        
        # Figure 1: Performance cumulée
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['Portfolio_Value'], 'b-', linewidth=2, label='Stratégie')
        plt.plot(results.index, results['Benchmark_Value'], 'r--', linewidth=2, label='Benchmark')
        plt.title('Performance cumulée', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Valeur du portefeuille')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        figures['cumulative_performance'] = fig1
        
        # Figure 2: Drawdowns
        portfolio_returns = results['Portfolio_Return'].dropna()
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        
        fig2 = plt.figure(figsize=(10, 4))
        plt.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
        plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
        plt.title('Drawdowns (%)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['drawdowns'] = fig2
        
        # Figure 3: Rendements annuels
        annual_returns = pd.DataFrame(metrics['annual_returns']).T
        
        fig3 = plt.figure(figsize=(10, 6))
        ax = annual_returns[['portfolio', 'benchmark']].plot(kind='bar', figsize=(10, 6))
        plt.title('Rendements annuels (%)', fontsize=14)
        plt.xlabel('Année')
        plt.ylabel('Rendement (%)')
        plt.grid(True, alpha=0.3)
        ax.set_xticklabels([str(year) for year in annual_returns.index], rotation=45)
        for i, v in enumerate(annual_returns['portfolio']):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
        plt.tight_layout()
        figures['annual_returns'] = fig3
        
        # Figure 4: Allocations au fil du temps
        fig4 = plt.figure(figsize=(12, 6))
        allocations.plot.area(stacked=True, alpha=0.7, cmap='viridis', figsize=(12, 6))
        plt.title('Allocations sectorielles au fil du temps', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Allocation (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        figures['allocations'] = fig4
        
        # Figure 5: Matrice de corrélation
        sector_returns = {}
        for sector in allocations.columns:
            if sector in self.sector_data.columns:
                sector_returns[sector] = self.sector_data[sector].pct_change()
        
        if sector_returns:
            returns_df = pd.DataFrame(sector_returns)
            corr_matrix = returns_df.corr()
            
            fig5 = plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Matrice de corrélation des rendements sectoriels', fontsize=14)
            plt.tight_layout()
            figures['correlation_matrix'] = fig5
        
        return figures


# Fonctions de stratégie d'exemple
def simple_momentum_strategy(sector_data, macro_data, current_date, lookback_periods=6, top_n=3):
    """
    Stratégie simple basée sur le momentum.
    
    Args:
        sector_data (pd.DataFrame): Données sectorielles historiques.
        macro_data (pd.DataFrame): Données macroéconomiques historiques.
        current_date (datetime): Date actuelle pour le calcul des allocations.
        lookback_periods (int): Nombre de périodes pour calculer le momentum.
        top_n (int): Nombre de secteurs à sélectionner.
        
    Returns:
        pd.Series: Allocations sectorielles.
    """
    # Liste des secteurs (en ignorant les colonnes spéciales)
    sectors = [col for col in sector_data.columns if '_' not in col]
    
    # Calcul du momentum
    momentum = {}
    for sector in sectors:
        if sector in sector_data.columns:
            # Calcul du rendement sur la période de lookback
            start_price = sector_data[sector].iloc[-lookback_periods-1] if len(sector_data) > lookback_periods else sector_data[sector].iloc[0]
            end_price = sector_data[sector].iloc[-1]
            momentum[sector] = end_price / start_price - 1
    
    # Conversion en Series
    momentum_series = pd.Series(momentum)
    
    # Sélection des top_n secteurs
    selected_sectors = momentum_series.nlargest(top_n)
    
    # Création des allocations (égales pour les secteurs sélectionnés)
    allocations = pd.Series(0, index=sectors)
    for sector in selected_sectors.index:
        allocations[sector] = 1 / len(selected_sectors)
    
    return allocations


def cycle_based_strategy(sector_data, macro_data, current_date, cycle_classifier, top_n=3, momentum_weight=0.5):
    """
    Stratégie basée sur les cycles économiques et le momentum.
    
    Args:
        sector_data (pd.DataFrame): Données sectorielles historiques.
        macro_data (pd.DataFrame): Données macroéconomiques historiques.
        current_date (datetime): Date actuelle pour le calcul des allocations.
        cycle_classifier: Classifieur de cycles économiques.
        top_n (int): Nombre de secteurs à sélectionner.
        momentum_weight (float): Poids du momentum dans la sélection (0-1).
        
    Returns:
        pd.Series: Allocations sectorielles.
    """
    # Nécessite des données macroéconomiques et un classifieur de cycles
    if macro_data is None or cycle_classifier is None:
        return simple_momentum_strategy(sector_data, None, current_date, lookback_periods=6, top_n=top_n)
    
    # Liste des secteurs (en ignorant les colonnes spéciales)
    sectors = [col for col in sector_data.columns if '_' not in col]
    
    # Identification de la phase économique actuelle
    try:
        current_phase = cycle_classifier.predict(macro_data).iloc[-1]
    except Exception as e:
        logger.error(f"Erreur lors de l'identification de la phase économique: {e}")
        return simple_momentum_strategy(sector_data, None, current_date, lookback_periods=6, top_n=top_n)
    
    # Définition des scores sectoriels par phase (exemple simplifié)
    sector_cycle_performance = {
        'Expansion': {s: 1.3 if s in ['XLK', 'XLY', 'XLI'] else 1.0 if s in ['XLB', 'XLF'] else 0.8 for s in sectors},
        'Surchauffe': {s: 1.3 if s in ['XLE', 'XLB', 'XLF'] else 1.0 if s in ['XLI', 'XLRE'] else 0.8 for s in sectors},
        'Ralentissement': {s: 1.3 if s in ['XLU', 'XLP', 'XLV'] else 1.0 if s in ['XLRE', 'XLE'] else 0.8 for s in sectors},
        'Récession': {s: 1.3 if s in ['XLU', 'XLP', 'XLV'] else 1.0 if s in ['XLC'] else 0.8 for s in sectors},
        'Reprise': {s: 1.3 if s in ['XLY', 'XLK', 'XLI'] else 1.0 if s in ['XLB', 'XLF'] else 0.8 for s in sectors}
    }
    
    # Scores de cycle pour la phase actuelle
    if current_phase in sector_cycle_performance:
        cycle_scores = pd.Series(sector_cycle_performance[current_phase])
    else:
        cycle_scores = pd.Series(1, index=sectors)
    
    # Normalisation des scores de cycle (0-1)
    cycle_scores = (cycle_scores - cycle_scores.min()) / (cycle_scores.max() - cycle_scores.min()) if cycle_scores.max() > cycle_scores.min() else pd.Series(1/len(sectors), index=sectors)
    
    # Calcul du momentum (sur 6 mois)
    momentum_scores = pd.Series(0, index=sectors)
    lookback_periods = 6  # 6 périodes pour le momentum
    
    for sector in sectors:
        if sector in sector_data.columns and len(sector_data) > lookback_periods:
            # Calcul du rendement sur la période de lookback
            start_price = sector_data[sector].iloc[-lookback_periods-1]
            end_price = sector_data[sector].iloc[-1]
            momentum_scores[sector] = end_price / start_price - 1
    
    # Normalisation des scores de momentum (0-1)
    if momentum_scores.max() > momentum_scores.min():
        momentum_scores = (momentum_scores - momentum_scores.min()) / (momentum_scores.max() - momentum_scores.min())
    else:
        momentum_scores = pd.Series(1/len(sectors), index=sectors)
    
    # Combinaison des scores
    combined_scores = cycle_scores * (1 - momentum_weight) + momentum_scores * momentum_weight
    
    # Sélection des top_n secteurs
    selected_sectors = combined_scores.nlargest(top_n)
    
    # Création des allocations (égales pour les secteurs sélectionnés)
    allocations = pd.Series(0, index=sectors)
    for sector in selected_sectors.index:
        allocations[sector] = 1 / len(selected_sectors)
    
    return allocations


if __name__ == "__main__":
    # Exemple d'utilisation du framework de backtesting
    import os
    
    # Chemins des données
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed"))
    sector_data_path = os.path.join(data_dir, "sector_data.csv")
    macro_data_path = os.path.join(data_dir, "macro_data.csv")
    
    # Chargement des données
    if os.path.exists(sector_data_path):
        sector_data = pd.read_csv(sector_data_path, index_col=0, parse_dates=True)
        print(f"Données sectorielles chargées: {len(sector_data)} observations")
    else:
        print(f"Fichier {sector_data_path} non trouvé")
        sector_data = None
    
    if os.path.exists(macro_data_path):
        macro_data = pd.read_csv(macro_data_path, index_col=0, parse_dates=True)
        print(f"Données macroéconomiques chargées: {len(macro_data)} observations")
    else:
        print(f"Fichier {macro_data_path} non trouvé")
        macro_data = None
    
    if sector_data is not None:
        # Création du moteur de backtesting
        backtest = BacktestEngine(sector_data, macro_data)
        
        # Exécution d'un backtest simple avec une stratégie de momentum
        start_date = "2010-01-01"
        end_date = "2020-12-31"
        
        print(f"Exécution du backtest de {start_date} à {end_date}...")
        results, allocations = backtest.run_simple_strategy(
            simple_momentum_strategy,
            {'lookback_periods': 6, 'top_n': 3},
            start_date=start_date,
            end_date=end_date,
            frequency='M',
            initial_capital=10000
        )
        
        # Calcul des métriques de performance
        metrics = backtest.calculate_performance_metrics(results)
        
        # Affichage des métriques principales
        print("\nMétriques de performance principales:")
        print(f"Rendement total: {metrics['total_return']:.2%}")
        print(f"Rendement annualisé: {metrics['annualized_return']:.2%}")
        print(f"Volatilité: {metrics['volatility']:.2%}")
        print(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"Drawdown maximum: {metrics['max_drawdown']:.2%}")
        
        # Génération d'un rapport de performance
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "backtest_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report = backtest.generate_performance_report(results, allocations, output_file=report_path)
        print(f"Rapport de performance généré et sauvegardé dans {report_path}")
        
        # Optimisation des paramètres
        print("\nOptimisation des paramètres...")
        param_grid = {
            'lookback_periods': [3, 6, 9, 12],
            'top_n': [2, 3, 4, 5]
        }
        
        best_params, optimization_results = backtest.run_strategy_optimization(
            simple_momentum_strategy,
            param_grid,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Meilleurs paramètres: {best_params}")
        print("Top 5 combinaisons de paramètres:")
        print(optimization_results.head())
