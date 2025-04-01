"""
Module de visualisation des performances de stratégies de rotation sectorielle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import logging
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """
    Classe pour générer des visualisations détaillées des performances d'une stratégie.
    """

    def __init__(self, style='seaborn', theme='light', interactive=True):
        """
        Initialise le visualiseur de performances.

        Args:
            style (str, optional): Style des graphiques matplotlib ('seaborn', 'ggplot', etc.).
            theme (str, optional): Thème des graphiques ('light' ou 'dark').
            interactive (bool, optional): Si True, utilise plotly pour les graphiques interactifs.
        """
        self.style = style
        self.theme = theme
        self.interactive = interactive
        
        # Configuration du style
        if not interactive:
            plt.style.use(style)
        
        # Configuration des thèmes
        self.colors = {
            'light': {
                'portfolio': '#1f77b4',  # Bleu
                'benchmark': '#ff7f0e',  # Orange
                'background': '#ffffff',
                'text': '#333333',
                'grid': '#dddddd',
                'profit': '#2ca02c',     # Vert
                'loss': '#d62728',       # Rouge
                'neutral': '#7f7f7f'     # Gris
            },
            'dark': {
                'portfolio': '#1f77b4',  # Bleu
                'benchmark': '#ff7f0e',  # Orange
                'background': '#333333',
                'text': '#ffffff',
                'grid': '#555555',
                'profit': '#2ca02c',     # Vert
                'loss': '#d62728',       # Rouge
                'neutral': '#7f7f7f'     # Gris
            }
        }
        
        self.current_colors = self.colors[theme]
        
        logger.info(f"PerformanceVisualizer initialisé avec style={style}, theme={theme}, interactive={interactive}")
    
    def plot_cumulative_performance(self, results, title="Performance cumulée", 
                                    height=500, width=800, include_benchmark=True,
                                    log_scale=False, show_drawdowns=False):
        """
        Visualise la performance cumulée du portefeuille.

        Args:
            results (pd.DataFrame): DataFrame contenant les colonnes Portfolio_Value, 
                                   Benchmark_Value, etc.
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            include_benchmark (bool, optional): Si True, inclut le benchmark.
            log_scale (bool, optional): Si True, utilise une échelle logarithmique.
            show_drawdowns (bool, optional): Si True, affiche les drawdowns.

        Returns:
            Figure: Graphique de performance cumulée.
        """
        if self.interactive:
            fig = go.Figure()
            
            # Ajout de la performance du portefeuille
            fig.add_trace(
                go.Scatter(
                    x=results.index,
                    y=results['Portfolio_Value'],
                    mode='lines',
                    name='Portefeuille',
                    line=dict(color=self.current_colors['portfolio'], width=2)
                )
            )
            
            # Ajout du benchmark
            if include_benchmark and 'Benchmark_Value' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=results.index,
                        y=results['Benchmark_Value'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.current_colors['benchmark'], width=2, dash='dash')
                    )
                )
            
            # Ajout des drawdowns si demandé
            if show_drawdowns and 'Portfolio_Return' in results.columns:
                portfolio_returns = results['Portfolio_Return'].dropna()
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1
                
                # Seuil de drawdown pour coloration
                drawdown_threshold = -0.1  # 10% drawdown
                
                # Création des zones de drawdown
                for i in range(1, len(drawdown)):
                    if drawdown.iloc[i] <= drawdown_threshold and drawdown.iloc[i-1] > drawdown_threshold:
                        # Début d'un drawdown significatif
                        start_idx = i
                    elif drawdown.iloc[i] > drawdown_threshold and drawdown.iloc[i-1] <= drawdown_threshold:
                        # Fin d'un drawdown significatif
                        end_idx = i
                        
                        fig.add_shape(
                            type="rect",
                            x0=drawdown.index[start_idx],
                            x1=drawdown.index[end_idx],
                            y0=0,
                            y1=results['Portfolio_Value'].max() * 1.1,
                            fillcolor="red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
            
            # Configuration de l'échelle logarithmique si demandée
            if log_scale:
                fig.update_layout(yaxis_type="log")
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Valeur du portefeuille",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Plot du portefeuille
            ax.plot(results.index, results['Portfolio_Value'], '-', 
                   color=self.current_colors['portfolio'], linewidth=2, label='Portefeuille')
            
            # Plot du benchmark
            if include_benchmark and 'Benchmark_Value' in results.columns:
                ax.plot(results.index, results['Benchmark_Value'], '--', 
                       color=self.current_colors['benchmark'], linewidth=2, label='Benchmark')
            
            # Ajout des drawdowns si demandé
            if show_drawdowns and 'Portfolio_Return' in results.columns:
                portfolio_returns = results['Portfolio_Return'].dropna()
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1
                
                # Seuil de drawdown pour coloration
                drawdown_threshold = -0.1  # 10% drawdown
                
                # Création des zones de drawdown
                for i in range(1, len(drawdown)):
                    if drawdown.iloc[i] <= drawdown_threshold and drawdown.iloc[i-1] > drawdown_threshold:
                        # Début d'un drawdown significatif
                        start_idx = i
                    elif drawdown.iloc[i] > drawdown_threshold and drawdown.iloc[i-1] <= drawdown_threshold:
                        # Fin d'un drawdown significatif
                        end_idx = i
                        
                        ax.axvspan(drawdown.index[start_idx], drawdown.index[end_idx], 
                                   alpha=0.2, color=self.current_colors['loss'])
            
            # Configuration de l'échelle logarithmique si demandée
            if log_scale:
                ax.set_yscale('log')
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Valeur du portefeuille")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            ax.legend()
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            # Formatage des valeurs sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            
            plt.tight_layout()
            return fig
    
    def plot_drawdowns(self, results, title="Drawdowns", height=400, width=800, 
                       top_n_drawdowns=3, show_recovery=True):
        """
        Visualise les drawdowns du portefeuille.

        Args:
            results (pd.DataFrame): DataFrame avec les rendements du portefeuille.
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            top_n_drawdowns (int, optional): Nombre de drawdowns principaux à analyser.
            show_recovery (bool, optional): Si True, affiche les périodes de récupération.

        Returns:
            Figure: Graphique des drawdowns.
        """
        if 'Portfolio_Return' not in results.columns:
            logger.warning("La colonne 'Portfolio_Return' est nécessaire pour calculer les drawdowns.")
            return None
        
        # Calcul des drawdowns
        portfolio_returns = results['Portfolio_Return'].dropna()
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        
        # Identification des drawdowns significatifs
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_drawdown = {}
        
        for i in range(1, len(drawdown)):
            if drawdown.iloc[i] < 0 and drawdown.iloc[i-1] >= 0:
                # Début d'un drawdown
                current_drawdown = {
                    'start_idx': i,
                    'start_date': drawdown.index[i],
                    'max_drawdown': 0,
                    'max_drawdown_idx': i,
                    'max_drawdown_date': drawdown.index[i]
                }
            
            if drawdown.iloc[i] < current_drawdown.get('max_drawdown', 0):
                # Nouveau maximum drawdown
                current_drawdown['max_drawdown'] = drawdown.iloc[i]
                current_drawdown['max_drawdown_idx'] = i
                current_drawdown['max_drawdown_date'] = drawdown.index[i]
            
            if drawdown.iloc[i] >= 0 and drawdown.iloc[i-1] < 0:
                # Fin d'un drawdown
                current_drawdown['end_idx'] = i
                current_drawdown['end_date'] = drawdown.index[i]
                current_drawdown['duration'] = (current_drawdown['end_date'] - current_drawdown['start_date']).days
                current_drawdown['recovery'] = (current_drawdown['end_date'] - current_drawdown['max_drawdown_date']).days
                
                drawdown_periods.append(current_drawdown)
                current_drawdown = {}
        
        # Si un drawdown est en cours à la fin de la période
        if current_drawdown and 'start_idx' in current_drawdown:
            current_drawdown['end_idx'] = len(drawdown) - 1
            current_drawdown['end_date'] = drawdown.index[-1]
            current_drawdown['duration'] = (current_drawdown['end_date'] - current_drawdown['start_date']).days
            current_drawdown['recovery'] = 'En cours'
            
            drawdown_periods.append(current_drawdown)
        
        # Tri des drawdowns par ampleur
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        top_drawdowns = drawdown_periods[:top_n_drawdowns]
        
        if self.interactive:
            fig = go.Figure()
            
            # Graphique des drawdowns
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.current_colors['loss'], width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba({','.join(str(int(c * 255)) for c in plt.colors.to_rgb(self.current_colors['loss']))},0.3)"
                )
            )
            
            # Ajout des annotations pour les drawdowns significatifs
            for i, dd in enumerate(top_drawdowns):
                fig.add_annotation(
                    x=dd['max_drawdown_date'],
                    y=dd['max_drawdown'] * 100,
                    text=f"{dd['max_drawdown']:.1%}",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor=self.current_colors['text'],
                    arrowwidth=2,
                    arrowsize=1,
                    font=dict(
                        color=self.current_colors['text']
                    )
                )
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                yaxis=dict(
                    tickformat=".1%",
                    range=[min(drawdown) * 100 * 1.1, 5]  # Marge de 10% en dessous, 5% au-dessus
                ),
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Plot des drawdowns
            ax.plot(drawdown.index, drawdown * 100, '-', 
                   color=self.current_colors['loss'], linewidth=2)
            ax.fill_between(drawdown.index, drawdown * 100, 0, 
                           color=self.current_colors['loss'], alpha=0.3)
            
            # Ajout des annotations pour les drawdowns significatifs
            for i, dd in enumerate(top_drawdowns):
                ax.annotate(
                    f"{dd['max_drawdown']:.1%}",
                    xy=(dd['max_drawdown_date'], dd['max_drawdown'] * 100),
                    xytext=(dd['max_drawdown_date'] + timedelta(days=30), dd['max_drawdown'] * 100 * 0.8),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=self.current_colors['text']
                    ),
                    color=self.current_colors['text']
                )
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown (%)")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            # Formatage des pourcentages sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
            
            # Limites de l'axe y
            ax.set_ylim([min(drawdown) * 100 * 1.1, 5])  # Marge de 10% en dessous, 5% au-dessus
            
            plt.tight_layout()
            return fig
    
    def plot_annual_returns(self, results, title="Rendements annuels", height=500, width=800,
                           include_benchmark=True, show_excess_return=True):
        """
        Visualise les rendements annuels.

        Args:
            results (pd.DataFrame): DataFrame avec les rendements du portefeuille et du benchmark.
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            include_benchmark (bool, optional): Si True, inclut le benchmark.
            show_excess_return (bool, optional): Si True, affiche l'excès de rendement.

        Returns:
            Figure: Graphique des rendements annuels.
        """
        if 'Portfolio_Return' not in results.columns:
            logger.warning("La colonne 'Portfolio_Return' est nécessaire pour calculer les rendements annuels.")
            return None
        
        # Calcul des rendements annuels
        portfolio_returns = results['Portfolio_Return'].dropna()
        portfolio_annual = (1 + portfolio_returns).groupby(portfolio_returns.index.year).prod() - 1
        
        if include_benchmark and 'Benchmark_Return' in results.columns:
            benchmark_returns = results['Benchmark_Return'].dropna()
            benchmark_annual = (1 + benchmark_returns).groupby(benchmark_returns.index.year).prod() - 1
            
            if show_excess_return:
                excess_annual = portfolio_annual - benchmark_annual
        
        # Création du DataFrame pour le plotting
        annual_df = pd.DataFrame({'Portfolio': portfolio_annual})
        if include_benchmark and 'Benchmark_Return' in results.columns:
            annual_df['Benchmark'] = benchmark_annual
            if show_excess_return:
                annual_df['Excess'] = excess_annual
        
        if self.interactive:
            fig = go.Figure()
            
            # Ajout des barres pour le portefeuille
            fig.add_trace(
                go.Bar(
                    x=annual_df.index,
                    y=annual_df['Portfolio'] * 100,
                    name='Portefeuille',
                    marker_color=self.current_colors['portfolio']
                )
            )
            
            # Ajout des barres pour le benchmark
            if include_benchmark and 'Benchmark' in annual_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=annual_df.index,
                        y=annual_df['Benchmark'] * 100,
                        name='Benchmark',
                        marker_color=self.current_colors['benchmark']
                    )
                )
            
            # Ajout des barres pour l'excès de rendement
            if show_excess_return and 'Excess' in annual_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=annual_df.index,
                        y=annual_df['Excess'] * 100,
                        name='Excès',
                        marker_color=self.current_colors['profit']
                    )
                )
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Année",
                yaxis_title="Rendement (%)",
                barmode='group',
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Nombre de barres et largeur
            n_bars = len(annual_df.columns)
            bar_width = 0.8 / n_bars
            
            # Positions des barres
            positions = np.arange(len(annual_df))
            
            # Plot des barres pour chaque série
            for i, (col, color) in enumerate(zip(
                annual_df.columns, 
                [self.current_colors['portfolio'], self.current_colors['benchmark'], self.current_colors['profit']]
            )):
                ax.bar(
                    positions + (i - n_bars/2 + 0.5) * bar_width,
                    annual_df[col] * 100,
                    bar_width,
                    label=col,
                    color=color
                )
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Année")
            ax.set_ylabel("Rendement (%)")
            ax.set_xticks(positions)
            ax.set_xticklabels(annual_df.index)
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'], axis='y')
            ax.legend()
            
            # Rotation des étiquettes d'année
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Formatage des pourcentages sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
            
            # Ajout d'une ligne horizontale à 0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            plt.tight_layout()
            return fig
    
    def plot_rolling_returns(self, results, window=252, title=None, height=500, width=800,
                            include_benchmark=True, annualized=True):
        """
        Visualise les rendements glissants.

        Args:
            results (pd.DataFrame): DataFrame avec les rendements du portefeuille et du benchmark.
            window (int, optional): Fenêtre de calcul des rendements (en jours/observations).
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            include_benchmark (bool, optional): Si True, inclut le benchmark.
            annualized (bool, optional): Si True, annualise les rendements.

        Returns:
            Figure: Graphique des rendements glissants.
        """
        if 'Portfolio_Return' not in results.columns:
            logger.warning("La colonne 'Portfolio_Return' est nécessaire pour calculer les rendements glissants.")
            return None
        
        # Détermination du titre si non spécifié
        if title is None:
            period_str = f"{window} jours"
            if window == 252:
                period_str = "1 an"
            elif window == 252 * 3:
                period_str = "3 ans"
            elif window == 252 * 5:
                period_str = "5 ans"
                
            title = f"Rendements glissants ({period_str})"
        
        # Calcul des rendements glissants
        portfolio_returns = results['Portfolio_Return'].dropna()
        rolling_returns = (1 + portfolio_returns).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True
        )
        
        if include_benchmark and 'Benchmark_Return' in results.columns:
            benchmark_returns = results['Benchmark_Return'].dropna()
            benchmark_rolling = (1 + benchmark_returns).rolling(window).apply(
                lambda x: x.prod() - 1, raw=True
            )
        
        # Annualisation des rendements si demandé
        annual_factor = 252 / window if window < 252 else 1
        if annualized and window != 252:
            rolling_returns = (1 + rolling_returns) ** annual_factor - 1
            if include_benchmark and 'Benchmark_Return' in results.columns:
                benchmark_rolling = (1 + benchmark_rolling) ** annual_factor - 1
        
        if self.interactive:
            fig = go.Figure()
            
            # Ajout des rendements glissants du portefeuille
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns * 100,
                    mode='lines',
                    name='Portefeuille',
                    line=dict(color=self.current_colors['portfolio'], width=2)
                )
            )
            
            # Ajout des rendements glissants du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_rolling.index,
                        y=benchmark_rolling * 100,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.current_colors['benchmark'], width=2, dash='dash')
                    )
                )
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Rendement (%)",
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            # Ajout d'une ligne horizontale à 0
            fig.add_shape(
                type="line",
                x0=rolling_returns.index[0],
                x1=rolling_returns.index[-1],
                y0=0,
                y1=0,
                line=dict(color="black", width=1, dash="dash")
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Plot des rendements glissants du portefeuille
            ax.plot(rolling_returns.index, rolling_returns * 100, '-', 
                   color=self.current_colors['portfolio'], linewidth=2, label='Portefeuille')
            
            # Plot des rendements glissants du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                ax.plot(benchmark_rolling.index, benchmark_rolling * 100, '--', 
                       color=self.current_colors['benchmark'], linewidth=2, label='Benchmark')
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Rendement (%)")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            ax.legend()
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            # Formatage des pourcentages sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
            
            # Ajout d'une ligne horizontale à 0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            return fig
    
    def plot_rolling_volatility(self, results, window=63, title=None, height=500, width=800,
                               include_benchmark=True, annualized=True):
        """
        Visualise la volatilité glissante.

        Args:
            results (pd.DataFrame): DataFrame avec les rendements du portefeuille et du benchmark.
            window (int, optional): Fenêtre de calcul de la volatilité (en jours/observations).
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            include_benchmark (bool, optional): Si True, inclut le benchmark.
            annualized (bool, optional): Si True, annualise la volatilité.

        Returns:
            Figure: Graphique de la volatilité glissante.
        """
        if 'Portfolio_Return' not in results.columns:
            logger.warning("La colonne 'Portfolio_Return' est nécessaire pour calculer la volatilité glissante.")
            return None
        
        # Détermination du titre si non spécifié
        if title is None:
            period_str = f"{window} jours"
            if window == 21:
                period_str = "1 mois"
            elif window == 63:
                period_str = "3 mois"
            elif window == 126:
                period_str = "6 mois"
                
            title = f"Volatilité glissante ({period_str})"
        
        # Calcul de la volatilité glissante
        portfolio_returns = results['Portfolio_Return'].dropna()
        rolling_vol = portfolio_returns.rolling(window).std()
        
        if include_benchmark and 'Benchmark_Return' in results.columns:
            benchmark_returns = results['Benchmark_Return'].dropna()
            benchmark_vol = benchmark_returns.rolling(window).std()
        
        # Annualisation de la volatilité si demandé
        if annualized:
            annual_factor = np.sqrt(252)  # Jours de trading par an
            rolling_vol = rolling_vol * annual_factor
            if include_benchmark and 'Benchmark_Return' in results.columns:
                benchmark_vol = benchmark_vol * annual_factor
        
        if self.interactive:
            fig = go.Figure()
            
            # Ajout de la volatilité glissante du portefeuille
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol * 100,
                    mode='lines',
                    name='Portefeuille',
                    line=dict(color=self.current_colors['portfolio'], width=2)
                )
            )
            
            # Ajout de la volatilité glissante du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_vol.index,
                        y=benchmark_vol * 100,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.current_colors['benchmark'], width=2, dash='dash')
                    )
                )
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Volatilité (%)",
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Plot de la volatilité glissante du portefeuille
            ax.plot(rolling_vol.index, rolling_vol * 100, '-', 
                   color=self.current_colors['portfolio'], linewidth=2, label='Portefeuille')
            
            # Plot de la volatilité glissante du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                ax.plot(benchmark_vol.index, benchmark_vol * 100, '--', 
                       color=self.current_colors['benchmark'], linewidth=2, label='Benchmark')
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Volatilité (%)")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            ax.legend()
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            # Formatage des pourcentages sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
            
            plt.tight_layout()
            return fig
    
    def plot_rolling_sharpe(self, results, window=252, risk_free_rate=0.02, title=None, 
                           height=500, width=800, include_benchmark=True):
        """
        Visualise le ratio de Sharpe glissant.

        Args:
            results (pd.DataFrame): DataFrame avec les rendements du portefeuille et du benchmark.
            window (int, optional): Fenêtre de calcul du ratio (en jours/observations).
            risk_free_rate (float, optional): Taux sans risque annualisé.
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            include_benchmark (bool, optional): Si True, inclut le benchmark.

        Returns:
            Figure: Graphique du ratio de Sharpe glissant.
        """
        if 'Portfolio_Return' not in results.columns:
            logger.warning("La colonne 'Portfolio_Return' est nécessaire pour calculer le ratio de Sharpe glissant.")
            return None
        
        # Détermination du titre si non spécifié
        if title is None:
            period_str = f"{window} jours"
            if window == 252:
                period_str = "1 an"
            elif window == 252 * 3:
                period_str = "3 ans"
            elif window == 252 * 5:
                period_str = "5 ans"
                
            title = f"Ratio de Sharpe glissant ({period_str})"
        
        # Conversion du taux sans risque annuel en taux par période
        rf_period = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calcul du ratio de Sharpe glissant pour le portefeuille
        portfolio_returns = results['Portfolio_Return'].dropna()
        excess_returns = portfolio_returns - rf_period
        
        def rolling_sharpe(x):
            return (x.mean() / x.std()) * np.sqrt(252)
        
        rolling_sharpe_portfolio = excess_returns.rolling(window).apply(rolling_sharpe, raw=True)
        
        # Calcul du ratio de Sharpe glissant pour le benchmark
        if include_benchmark and 'Benchmark_Return' in results.columns:
            benchmark_returns = results['Benchmark_Return'].dropna()
            excess_returns_benchmark = benchmark_returns - rf_period
            rolling_sharpe_benchmark = excess_returns_benchmark.rolling(window).apply(rolling_sharpe, raw=True)
        
        if self.interactive:
            fig = go.Figure()
            
            # Ajout du ratio de Sharpe glissant du portefeuille
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe_portfolio.index,
                    y=rolling_sharpe_portfolio,
                    mode='lines',
                    name='Portefeuille',
                    line=dict(color=self.current_colors['portfolio'], width=2)
                )
            )
            
            # Ajout du ratio de Sharpe glissant du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_sharpe_benchmark.index,
                        y=rolling_sharpe_benchmark,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.current_colors['benchmark'], width=2, dash='dash')
                    )
                )
            
            # Configuration du layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Ratio de Sharpe",
                height=height,
                width=width,
                template="plotly_white" if self.theme == 'light' else "plotly_dark"
            )
            
            # Ajout d'une ligne horizontale à 0
            fig.add_shape(
                type="line",
                x0=rolling_sharpe_portfolio.index[0],
                x1=rolling_sharpe_portfolio.index[-1],
                y0=0,
                y1=0,
                line=dict(color="black", width=1, dash="dash")
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            # Plot du ratio de Sharpe glissant du portefeuille
            ax.plot(rolling_sharpe_portfolio.index, rolling_sharpe_portfolio, '-', 
                   color=self.current_colors['portfolio'], linewidth=2, label='Portefeuille')
            
            # Plot du ratio de Sharpe glissant du benchmark
            if include_benchmark and 'Benchmark_Return' in results.columns:
                ax.plot(rolling_sharpe_benchmark.index, rolling_sharpe_benchmark, '--', 
                       color=self.current_colors['benchmark'], linewidth=2, label='Benchmark')
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Ratio de Sharpe")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            ax.legend()
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            # Ajout d'une ligne horizontale à 0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            return fig
    
    def plot_sector_allocations(self, allocations, title="Allocations sectorielles", 
                               height=500, width=800, format='area', top_n=None,
                               smooth=True, window=3):
        """
        Visualise les allocations sectorielles au fil du temps.

        Args:
            allocations (pd.DataFrame): DataFrame contenant les allocations par secteur.
            title (str, optional): Titre du graphique.
            height (int, optional): Hauteur du graphique.
            width (int, optional): Largeur du graphique.
            format (str, optional): Format du graphique ('area', 'line', 'bar').
            top_n (int, optional): Limite le nombre de secteurs affichés.
            smooth (bool, optional): Si True, lisse les allocations.
            window (int, optional): Fenêtre de lissage.

        Returns:
            Figure: Graphique des allocations sectorielles.
        """
        # Copie des allocations pour éviter de modifier l'original
        alloc = allocations.copy()
        
        # Regroupement des secteurs peu alloués si top_n est spécifié
        if top_n is not None and top_n < len(alloc.columns):
            # Calcul de l'allocation moyenne par secteur
            avg_alloc = alloc.mean().sort_values(ascending=False)
            
            # Sélection des top_n secteurs
            top_sectors = avg_alloc.head(top_n).index.tolist()
            
            # Regroupement des autres secteurs
            other_sectors = [col for col in alloc.columns if col not in top_sectors]
            alloc['Autres'] = alloc[other_sectors].sum(axis=1)
            
            # Filtrage des colonnes
            alloc = alloc[top_sectors + ['Autres']]
        
        # Lissage des allocations si demandé
        if smooth and len(alloc) > window:
            alloc = alloc.rolling(window=window, min_periods=1).mean()
        
        if self.interactive:
            if format == 'area':
                fig = px.area(
                    alloc,
                    title=title,
                    height=height,
                    width=width,
                    template="plotly_white" if self.theme == 'light' else "plotly_dark"
                )
            elif format == 'line':
                fig = px.line(
                    alloc,
                    title=title,
                    height=height,
                    width=width,
                    template="plotly_white" if self.theme == 'light' else "plotly_dark"
                )
            elif format == 'bar':
                fig = px.bar(
                    alloc,
                    title=title,
                    height=height,
                    width=width,
                    template="plotly_white" if self.theme == 'light' else "plotly_dark"
                )
            else:
                logger.warning(f"Format '{format}' non reconnu. Utilisation du format 'area'.")
                fig = px.area(
                    alloc,
                    title=title,
                    height=height,
                    width=width,
                    template="plotly_white" if self.theme == 'light' else "plotly_dark"
                )
            
            # Configuration du layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Allocation (%)",
                yaxis=dict(
                    tickformat=".0%",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            
            if format == 'area':
                alloc.plot.area(ax=ax, stacked=True, alpha=0.7, colormap='viridis')
            elif format == 'line':
                alloc.plot.line(ax=ax, linewidth=2, colormap='viridis')
            elif format == 'bar':
                alloc.plot.bar(ax=ax, stacked=True, alpha=0.7, colormap='viridis')
            else:
                logger.warning(f"Format '{format}' non reconnu. Utilisation du format 'area'.")
                alloc.plot.area(ax=ax, stacked=True, alpha=0.7, colormap='viridis')
            
            # Configuration du graphique
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Allocation (%)")
            ax.grid(True, alpha=0.3, color=self.current_colors['grid'])
            
            # Formatage des pourcentages sur l'axe y
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
            
            # Si format 'bar', rotation des étiquettes
            if format == 'bar':
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Légende
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            return fig
    
    def generate_performance_dashboard(self, results, allocations=None, output_dir=None,
                                      filename_prefix="performance", format='png',
                                      dpi=300, include_metrics=True):
        """
        Génère un tableau de bord complet de performance.

        Args:
            results (pd.DataFrame): DataFrame contenant les résultats du backtest.
            allocations (pd.DataFrame, optional): DataFrame contenant les allocations.
            output_dir (str, optional): Répertoire de sortie pour les graphiques.
            filename_prefix (str, optional): Préfixe pour les noms de fichiers.
            format (str, optional): Format des fichiers ('png', 'pdf', 'svg', 'html').
            dpi (int, optional): Résolution des images (pour les formats raster).
            include_metrics (bool, optional): Si True, inclut un résumé des métriques.

        Returns:
            dict: Dictionnaire contenant les figures générées et les métriques.
        """
        # Création du répertoire de sortie si nécessaire
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Dictionnaire pour stocker les figures et les métriques
        dashboard = {
            'figures': {},
            'metrics': {}
        }
        
        # Génération des figures principales
        dashboard['figures']['cumulative_performance'] = self.plot_cumulative_performance(
            results, title="Performance cumulée", show_drawdowns=True
        )
        
        dashboard['figures']['drawdowns'] = self.plot_drawdowns(
            results, title="Drawdowns", top_n_drawdowns=5
        )
        
        dashboard['figures']['annual_returns'] = self.plot_annual_returns(
            results, title="Rendements annuels"
        )
        
        dashboard['figures']['rolling_returns'] = self.plot_rolling_returns(
            results, window=252, title="Rendements glissants (1 an)"
        )
        
        dashboard['figures']['rolling_volatility'] = self.plot_rolling_volatility(
            results, window=63, title="Volatilité glissante (3 mois)"
        )
        
        dashboard['figures']['rolling_sharpe'] = self.plot_rolling_sharpe(
            results, window=252, title="Ratio de Sharpe glissant (1 an)"
        )
        
        # Ajout des allocations si disponibles
        if allocations is not None:
            dashboard['figures']['allocations'] = self.plot_sector_allocations(
                allocations, title="Allocations sectorielles", smooth=True
            )
        
        # Calcul des métriques si demandé
        if include_metrics:
            # Calcul des métriques de base
            portfolio_returns = results['Portfolio_Return'].dropna()
            benchmark_returns = results['Benchmark_Return'].dropna() if 'Benchmark_Return' in results.columns else None
            
            # Période analysée
            start_date = results.index[0]
            end_date = results.index[-1]
            years = (end_date - start_date).days / 365.25
            
            # Rendements
            total_return = results['Portfolio_Value'].iloc[-1] / results['Portfolio_Value'].iloc[0] - 1
            annual_return = (1 + total_return) ** (1 / years) - 1
            
            # Volatilité
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Ratio de Sharpe
            risk_free_rate = 0.02  # 2% par défaut
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
            
            # Drawdown maximum
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Ratio de Calmar
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
            
            # Métriques vs benchmark
            if benchmark_returns is not None:
                # Rendement et volatilité du benchmark
                benchmark_total_return = results['Benchmark_Value'].iloc[-1] / results['Benchmark_Value'].iloc[0] - 1
                benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                
                # Comparaison avec le benchmark
                excess_return = annual_return - benchmark_annual_return
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
                
                # Beta et alpha
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
                alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
                
                # Ajout des métriques du benchmark
                dashboard['metrics']['benchmark'] = {
                    'total_return': benchmark_total_return,
                    'annual_return': benchmark_annual_return,
                    'volatility': benchmark_volatility
                }
                
                # Ajout des métriques de comparaison
                dashboard['metrics']['comparison'] = {
                    'excess_return': excess_return,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha
                }
            
            # Compilation des métriques principales
            dashboard['metrics']['portfolio'] = {
                'start_date': start_date,
                'end_date': end_date,
                'years': years,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio
            }
        
        # Sauvegarde des figures si un répertoire est spécifié
        if output_dir is not None:
            for name, fig in dashboard['figures'].items():
                filename = f"{filename_prefix}_{name}.{format}"
                filepath = os.path.join(output_dir, filename)
                
                if self.interactive:
                    # Sauvegarde des figures Plotly
                    if format == 'html':
                        fig.write_html(filepath)
                    else:
                        fig.write_image(filepath)
                else:
                    # Sauvegarde des figures Matplotlib
                    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                
                logger.info(f"Figure '{name}' sauvegardée dans {filepath}")
        
        return dashboard


if __name__ == "__main__":
    # Exemple d'utilisation
    import os
    import pandas as pd
    
    # Chemins des données
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed"))
    results_path = os.path.join(data_dir, "backtest_results.csv")
    allocations_path = os.path.join(data_dir, "sector_allocations.csv")
    
    # Vérification de l'existence des fichiers
    if os.path.exists(results_path) and os.path.exists(allocations_path):
        # Chargement des données
        results = pd.read_csv(results_path, index_col=0, parse_dates=True)
        allocations = pd.read_csv(allocations_path, index_col=0, parse_dates=True)
        
        # Création du visualiseur
        visualizer = PerformanceVisualizer(interactive=True)
        
        # Génération du tableau de bord
        output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "visualizations"))
        
        dashboard = visualizer.generate_performance_dashboard(
            results,
            allocations,
            output_dir=output_dir,
            format='html'
        )
        
        print(f"Tableau de bord généré avec {len(dashboard['figures'])} figures")
        print(f"Métriques calculées pour {len(dashboard['metrics'])} catégories")
        print(f"Figures sauvegardées dans {output_dir}")
    else:
        print(f"Fichiers de résultats non trouvés. Veuillez d'abord exécuter un backtest.")
