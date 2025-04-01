"""
Module contenant des fonctions utilitaires communes pour le projet de rotation sectorielle.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_project_root():
    """
    Renvoie le chemin racine du projet.

    Returns:
        Path: Chemin racine du projet.
    """
    # On cherche le répertoire contenant les dossiers src/, data/, notebooks/, etc.
    current_path = Path(__file__).resolve().parent
    while current_path.name != 'strategie-rotation-sectorielle' and current_path != current_path.parent:
        current_path = current_path.parent
    
    # Si on n'a pas trouvé le répertoire du projet, on utilise le parent de src/
    if current_path == current_path.parent:
        return Path(__file__).resolve().parent.parent.parent
    
    return current_path


def load_config(config_path=None):
    """
    Charge la configuration du projet depuis un fichier YAML.

    Args:
        config_path (str, optional): Chemin du fichier de configuration.
            Si None, utilise le fichier par défaut dans le répertoire config/.

    Returns:
        dict: Configuration du projet.
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de la configuration: {e}")
        logger.info("Utilisation de la configuration par défaut")
        return {}


def ensure_dir(directory):
    """
    Crée un répertoire s'il n'existe pas.

    Args:
        directory (str): Chemin du répertoire à créer.

    Returns:
        str: Chemin du répertoire créé.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé: {directory}")
    return directory


def save_to_json(data, filepath):
    """
    Sauvegarde des données dans un fichier JSON.

    Args:
        data (dict or list): Données à sauvegarder.
        filepath (str): Chemin du fichier de sortie.

    Returns:
        bool: True si la sauvegarde a réussi, False sinon.
    """
    try:
        # Création du répertoire parent si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Conversion des données non sérialisables
        def json_serial(obj):
            if isinstance(obj, (datetime, np.datetime64)):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=json_serial)
        
        logger.info(f"Données sauvegardées dans {filepath}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données: {e}")
        return False


def load_from_json(filepath):
    """
    Charge des données depuis un fichier JSON.

    Args:
        filepath (str): Chemin du fichier JSON.

    Returns:
        dict or list: Données chargées.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Données chargées depuis {filepath}")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None


def convert_frequency(data, frequency='M'):
    """
    Convertit les données à une fréquence spécifique.

    Args:
        data (pd.DataFrame): DataFrame avec un index de type datetime.
        frequency (str, optional): Fréquence cible ('D', 'W', 'M', 'Q', 'A').

    Returns:
        pd.DataFrame: DataFrame à la fréquence spécifiée.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("L'index doit être de type DatetimeIndex pour la conversion de fréquence.")
        return data
    
    freq_map = {
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'Q': 'quarterly',
        'A': 'yearly'
    }
    
    logger.info(f"Conversion des données à une fréquence {freq_map.get(frequency, frequency)}")
    
    # Pour les données de prix/valeurs, on utilise la dernière valeur de la période
    return data.resample(frequency).last()


def annualize_returns(returns, frequency='M'):
    """
    Annualise des rendements selon leur fréquence.

    Args:
        returns (pd.Series or pd.DataFrame): Rendements à annualiser.
        frequency (str, optional): Fréquence des rendements ('D', 'W', 'M', 'Q', 'A').

    Returns:
        pd.Series or pd.DataFrame: Rendements annualisés.
    """
    # Facteur d'annualisation
    ann_factor = {
        'D': 252,    # Jours de trading
        'W': 52,     # Semaines
        'M': 12,     # Mois
        'Q': 4,      # Trimestres
        'A': 1       # Années
    }
    
    factor = ann_factor.get(frequency, 12)  # Par défaut: mensuel
    
    # Calcul du rendement annualisé
    annual_return = (1 + returns.mean()) ** factor - 1
    
    return annual_return


def annualize_volatility(returns, frequency='M'):
    """
    Annualise la volatilité selon la fréquence des rendements.

    Args:
        returns (pd.Series or pd.DataFrame): Rendements pour calculer la volatilité.
        frequency (str, optional): Fréquence des rendements ('D', 'W', 'M', 'Q', 'A').

    Returns:
        pd.Series or pd.DataFrame: Volatilité annualisée.
    """
    # Facteur d'annualisation
    ann_factor = {
        'D': np.sqrt(252),    # Jours de trading
        'W': np.sqrt(52),     # Semaines
        'M': np.sqrt(12),     # Mois
        'Q': np.sqrt(4),      # Trimestres
        'A': 1                # Années
    }
    
    factor = ann_factor.get(frequency, np.sqrt(12))  # Par défaut: mensuel
    
    # Calcul de la volatilité annualisée
    annual_vol = returns.std() * factor
    
    return annual_vol


def calculate_drawdowns(returns):
    """
    Calcule les drawdowns à partir des rendements.

    Args:
        returns (pd.Series): Série de rendements.

    Returns:
        tuple: (drawdowns, max_drawdown, max_drawdown_duration)
    """
    # Calcul des rendements cumulés
    cumulative = (1 + returns).cumprod()
    
    # Calcul des drawdowns
    running_max = cumulative.cummax()
    drawdowns = (cumulative / running_max) - 1
    
    # Calcul du drawdown maximum
    max_drawdown = drawdowns.min()
    
    # Calcul de la durée du drawdown maximum
    is_drawdown = drawdowns < 0
    durations = []
    current_duration = 0
    
    for i, is_dd in enumerate(is_drawdown):
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    max_drawdown_duration = max(durations) if durations else 0
    
    return drawdowns, max_drawdown, max_drawdown_duration


def calculate_performance_metrics(returns, benchmark_returns=None, risk_free_rate=0.02, frequency='M'):
    """
    Calcule les principales métriques de performance.

    Args:
        returns (pd.Series): Rendements du portefeuille.
        benchmark_returns (pd.Series, optional): Rendements du benchmark.
        risk_free_rate (float, optional): Taux sans risque annualisé.
        frequency (str, optional): Fréquence des rendements ('D', 'W', 'M', 'Q', 'A').

    Returns:
        dict: Métriques de performance.
    """
    # Conversion du taux sans risque à la fréquence des rendements
    freq_factor = {
        'D': 252,
        'W': 52,
        'M': 12,
        'Q': 4,
        'A': 1
    }
    period_rf = (1 + risk_free_rate) ** (1 / freq_factor.get(frequency, 12)) - 1
    
    # Calcul du rendement annualisé
    total_return = (1 + returns).prod() - 1
    periods = len(returns)
    years = periods / freq_factor.get(frequency, 12)
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # Calcul de la volatilité annualisée
    volatility = annualize_volatility(returns, frequency)
    
    # Calcul du ratio de Sharpe
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0
    
    # Calcul des drawdowns
    drawdowns, max_drawdown, max_drawdown_duration = calculate_drawdowns(returns)
    
    # Calcul du ratio de Sortino
    downside_returns = returns[returns < 0]
    downside_deviation = annualize_volatility(downside_returns, frequency) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
    
    # Calcul du ratio de Calmar
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Compilation des métriques
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'calmar_ratio': calmar_ratio,
        'win_rate': (returns > 0).mean(),
        'periods': periods,
        'years': years
    }
    
    # Métriques vs benchmark
    if benchmark_returns is not None:
        # Calcul du rendement annualisé du benchmark
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1
        
        # Calcul de la volatilité annualisée du benchmark
        benchmark_volatility = annualize_volatility(benchmark_returns, frequency)
        
        # Calcul du ratio d'information
        tracking_error = annualize_volatility(returns - benchmark_returns, frequency)
        information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error != 0 else 0
        
        # Calcul de beta et alpha
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        # Alpha de Jensen
        jensen_alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        # Capture up/down
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        up_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean()) if up_market.any() and benchmark_returns[up_market].mean() != 0 else 0
        down_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()) if down_market.any() and benchmark_returns[down_market].mean() != 0 else 0
        
        # Ajout des métriques vs benchmark
        metrics.update({
            'benchmark_return': benchmark_annual_return,
            'benchmark_volatility': benchmark_volatility,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': jensen_alpha,
            'up_capture': up_capture,
            'down_capture': down_capture
        })
    
    return metrics


def create_correlation_matrix(returns, method='pearson', title='Matrice de corrélation'):
    """
    Crée une matrice de corrélation des rendements.

    Args:
        returns (pd.DataFrame): DataFrame des rendements.
        method (str, optional): Méthode de corrélation ('pearson', 'spearman', 'kendall').
        title (str, optional): Titre du graphique.

    Returns:
        matplotlib.figure.Figure: Figure de la matrice de corrélation.
    """
    # Calcul de la matrice de corrélation
    corr_matrix = returns.corr(method=method)
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Création de la heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    # Configuration du graphique
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_rolling_metric(returns, func, window, frequency='M', benchmark_returns=None, 
                        title=None, ylabel=None):
    """
    Crée un graphique de métrique glissante.

    Args:
        returns (pd.Series): Rendements du portefeuille.
        func (function): Fonction à appliquer à la fenêtre glissante.
        window (int): Taille de la fenêtre glissante.
        frequency (str, optional): Fréquence des rendements ('D', 'W', 'M', 'Q', 'A').
        benchmark_returns (pd.Series, optional): Rendements du benchmark.
        title (str, optional): Titre du graphique.
        ylabel (str, optional): Label de l'axe y.

    Returns:
        matplotlib.figure.Figure: Figure de la métrique glissante.
    """
    # Calcul de la métrique glissante
    rolling_metric = returns.rolling(window=window).apply(func, raw=True)
    
    if title is None:
        window_str = f"{window} "
        if frequency == 'D':
            window_str += "jours"
        elif frequency == 'W':
            window_str += "semaines"
        elif frequency == 'M':
            window_str += "mois"
        elif frequency == 'Q':
            window_str += "trimestres"
        elif frequency == 'A':
            window_str += "ans"
        
        title = f"{func.__name__.replace('_', ' ').title()} ({window_str})"
    
    if ylabel is None:
        ylabel = func.__name__.replace('_', ' ').title()
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot du portefeuille
    ax.plot(rolling_metric.index, rolling_metric, label='Portefeuille')
    
    # Plot du benchmark si disponible
    if benchmark_returns is not None:
        benchmark_metric = benchmark_returns.rolling(window=window).apply(func, raw=True)
        ax.plot(benchmark_metric.index, benchmark_metric, label='Benchmark', linestyle='--')
    
    # Configuration du graphique
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Auto-format des dates
    fig.autofmt_xdate()
    
    return fig


def rolling_sharpe(returns, window=12, risk_free_rate=0.02, frequency='M'):
    """
    Calcule le ratio de Sharpe glissant.

    Args:
        returns (pd.Series): Rendements.
        window (int, optional): Taille de la fenêtre en périodes.
        risk_free_rate (float, optional): Taux sans risque annualisé.
        frequency (str, optional): Fréquence des rendements ('D', 'W', 'M', 'Q', 'A').

    Returns:
        pd.Series: Ratio de Sharpe glissant.
    """
    # Conversion du taux sans risque à la fréquence des rendements
    freq_factor = {
        'D': 252,
        'W': 52,
        'M': 12,
        'Q': 4,
        'A': 1
    }
    period_rf = (1 + risk_free_rate) ** (1 / freq_factor.get(frequency, 12)) - 1
    
    # Calcul du Sharpe ratio glissant
    def sharpe_for_window(window_returns):
        excess = window_returns - period_rf
        sharpe = excess.mean() / excess.std() if excess.std() != 0 else 0
        # Annualisation du Sharpe
        sharpe *= np.sqrt(freq_factor.get(frequency, 12))
        return sharpe
    
    return returns.rolling(window=window).apply(sharpe_for_window, raw=True)


def extract_drawdown_periods(returns, threshold=-0.1):
    """
    Extrait les périodes de drawdown significatives.

    Args:
        returns (pd.Series): Rendements.
        threshold (float, optional): Seuil de drawdown (ex: -0.1 pour -10%).

    Returns:
        list: Liste des périodes de drawdown avec leurs caractéristiques.
    """
    # Calcul des drawdowns
    drawdowns, _, _ = calculate_drawdowns(returns)
    
    # Identification des périodes de drawdown
    drawdown_periods = []
    in_drawdown = False
    start_idx = None
    
    for i, (date, dd) in enumerate(drawdowns.items()):
        if dd <= threshold and not in_drawdown:
            # Début d'un drawdown significatif
            in_drawdown = True
            start_idx = i
            start_date = date
        elif dd > threshold and in_drawdown:
            # Fin d'un drawdown significatif
            in_drawdown = False
            end_date = date
            
            # Calcul des caractéristiques de ce drawdown
            duration = (end_date - start_date).days
            min_dd = drawdowns.iloc[start_idx:i+1].min()
            min_dd_date = drawdowns.iloc[start_idx:i+1].idxmin()
            
            # Calcul du temps de récupération
            recovery_duration = (end_date - min_dd_date).days
            
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'min_drawdown': min_dd,
                'min_drawdown_date': min_dd_date,
                'duration': duration,
                'recovery_duration': recovery_duration
            })
    
    # Si on est encore en drawdown à la fin de la série
    if in_drawdown:
        end_date = drawdowns.index[-1]
        duration = (end_date - start_date).days
        min_dd = drawdowns.iloc[start_idx:].min()
        min_dd_date = drawdowns.iloc[start_idx:].idxmin()
        
        drawdown_periods.append({
            'start_date': start_date,
            'end_date': end_date,
            'min_drawdown': min_dd,
            'min_drawdown_date': min_dd_date,
            'duration': duration,
            'recovery_duration': 'En cours'
        })
    
    return drawdown_periods


if __name__ == "__main__":
    # Test des fonctions
    logger.info("Test des fonctions utilitaires")
    
    # Test de get_project_root
    root = get_project_root()
    logger.info(f"Racine du projet: {root}")
    
    # Test de load_config
    config = load_config()
    logger.info(f"Configuration: {config}")
    
    # Test de ensure_dir
    test_dir = "test_dir"
    ensure_dir(test_dir)
    
    # Test de save_to_json et load_from_json
    test_data = {'test': 'data', 'date': datetime.now()}
    save_to_json(test_data, os.path.join(test_dir, 'test.json'))
    loaded_data = load_from_json(os.path.join(test_dir, 'test.json'))
    logger.info(f"Données chargées: {loaded_data}")
    
    # Nettoyage
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("Tests terminés")
