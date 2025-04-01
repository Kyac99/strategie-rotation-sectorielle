#!/usr/bin/env python3
"""
Script principal pour la stratégie de rotation sectorielle.
Ce script permet d'exécuter le pipeline complet, de la collecte des données
à la génération des recommandations sectorielles et des rapports.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime

# Ajout du répertoire courant au path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import des modules du projet
from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector
from src.models.economic_cycle_classifier import EconomicCycleClassifier
from src.models.sector_selector import SectorSelector
from src.backtest.backtest_engine import BacktestEngine
from src.visualization.performance_visualizer import PerformanceVisualizer
from src.utils.common_utils import load_config, ensure_dir

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/strategie_rotation.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Création du dossier logs si nécessaire
os.makedirs("logs", exist_ok=True)


def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Stratégie de rotation sectorielle")
    parser.add_argument(
        "--collect", action="store_true",
        help="Collecter de nouvelles données"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Entraîner le modèle de classification des cycles économiques"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Exécuter le backtesting de la stratégie"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Optimiser les paramètres de la stratégie"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Générer un rapport de performance"
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Générer des recommandations sectorielles actuelles"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Exécuter l'ensemble du pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Chemin du fichier de configuration (par défaut: config/config.yaml)"
    )
    
    return parser.parse_args()


def collect_data(config):
    """
    Collecte les données macroéconomiques et sectorielles.

    Args:
        config (dict): Configuration du projet.

    Returns:
        tuple: (macro_data, sector_data) - DataFrames contenant les données collectées.
    """
    logger.info("Début de la collecte des données")
    
    # Création des répertoires de données
    data_dir = config['paths']['data_processed']
    ensure_dir(data_dir)
    
    # Chemins des fichiers de données
    macro_path = os.path.join(data_dir, "macro_data.csv")
    sector_path = os.path.join(data_dir, "sector_data.csv")
    
    # Collecte des données macroéconomiques
    logger.info("Collecte des données macroéconomiques")
    macro_collector = MacroDataCollector()
    macro_data = macro_collector.get_all_series(start_date="2000-01-01", frequency='m')
    processed_macro = macro_collector.preprocess_data(macro_data)
    processed_macro.to_csv(macro_path)
    logger.info(f"Données macroéconomiques sauvegardées dans {macro_path}")
    
    # Collecte des données sectorielles
    logger.info("Collecte des données sectorielles")
    sector_collector = SectorDataCollector()
    etf_data = sector_collector.get_all_etf_data(start_date="2000-01-01")
    processed_sectors = sector_collector.preprocess_data(etf_data)
    processed_sectors.to_csv(sector_path)
    logger.info(f"Données sectorielles sauvegardées dans {sector_path}")
    
    logger.info("Collecte des données terminée")
    return processed_macro, processed_sectors


def load_data(config):
    """
    Charge les données macroéconomiques et sectorielles.

    Args:
        config (dict): Configuration du projet.

    Returns:
        tuple: (macro_data, sector_data) - DataFrames contenant les données.
    """
    logger.info("Chargement des données")
    
    # Chemins des fichiers de données
    data_dir = config['paths']['data_processed']
    macro_path = os.path.join(data_dir, "macro_data.csv")
    sector_path = os.path.join(data_dir, "sector_data.csv")
    
    # Vérification de l'existence des fichiers
    if not os.path.exists(macro_path) or not os.path.exists(sector_path):
        logger.warning("Données manquantes, collecte automatique")
        return collect_data(config)
    
    # Chargement des données
    macro_data = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    sector_data = pd.read_csv(sector_path, index_col=0, parse_dates=True)
    
    logger.info(f"Données macroéconomiques chargées: {len(macro_data)} observations")
    logger.info(f"Données sectorielles chargées: {len(sector_data)} observations")
    
    return macro_data, sector_data


def train_model(macro_data, config):
    """
    Entraîne le modèle de classification des cycles économiques.

    Args:
        macro_data (pd.DataFrame): Données macroéconomiques.
        config (dict): Configuration du projet.

    Returns:
        EconomicCycleClassifier: Modèle entraîné.
    """
    logger.info("Entraînement du modèle de classification des cycles économiques")
    
    # Création du répertoire des modèles
    models_dir = config['paths']['models']
    ensure_dir(models_dir)
    
    # Chemin du modèle
    model_path = os.path.join(models_dir, "economic_cycle_classifier.joblib")
    
    # Paramètres du modèle
    supervised = config['models']['economic_cycle']['supervised']
    
    # Création et entraînement du modèle
    classifier = EconomicCycleClassifier(supervised=supervised)
    classifier.fit(macro_data)
    
    # Sauvegarde du modèle
    classifier.save_model(model_path)
    logger.info(f"Modèle entraîné et sauvegardé dans {model_path}")
    
    # Prédiction et sauvegarde des phases économiques
    phases = classifier.predict(macro_data)
    phases_df = pd.DataFrame({'phase': phases}, index=phases.index)
    phases_path = os.path.join(config['paths']['data_processed'], "economic_phases.csv")
    phases_df.to_csv(phases_path)
    logger.info(f"Phases économiques sauvegardées dans {phases_path}")
    
    return classifier


def load_model(config):
    """
    Charge le modèle de classification des cycles économiques.

    Args:
        config (dict): Configuration du projet.

    Returns:
        EconomicCycleClassifier: Modèle chargé.
    """
    logger.info("Chargement du modèle de classification des cycles économiques")
    
    # Chemin du modèle
    model_path = os.path.join(config['paths']['models'], "economic_cycle_classifier.joblib")
    
    # Vérification de l'existence du modèle
    if not os.path.exists(model_path):
        logger.warning("Modèle non trouvé, entraînement automatique")
        macro_data, _ = load_data(config)
        return train_model(macro_data, config)
    
    # Chargement du modèle
    classifier = EconomicCycleClassifier.load_model(model_path)
    logger.info(f"Modèle chargé depuis {model_path}")
    
    return classifier


def run_backtest(macro_data, sector_data, classifier, config, optimize=False):
    """
    Exécute le backtesting de la stratégie.

    Args:
        macro_data (pd.DataFrame): Données macroéconomiques.
        sector_data (pd.DataFrame): Données sectorielles.
        classifier (EconomicCycleClassifier): Modèle de classification des cycles.
        config (dict): Configuration du projet.
        optimize (bool, optional): Si True, optimise les paramètres de la stratégie.

    Returns:
        tuple: (results, allocations, metrics) - Résultats du backtest.
    """
    logger.info("Exécution du backtesting de la stratégie")
    
    # Configuration du backtest
    backtest_config = config['backtest']
    start_date = backtest_config['start_date']
    end_date = backtest_config['end_date']
    initial_capital = backtest_config['initial_capital']
    rebalance_frequency = backtest_config['rebalance_frequency']
    transaction_cost = backtest_config['transaction_cost']
    benchmark = backtest_config['benchmark']
    
    # Création du moteur de backtesting
    backtest_engine = BacktestEngine(
        sector_data=sector_data,
        macro_data=macro_data,
        benchmark=benchmark,
        risk_free_rate=backtest_config['risk_free_rate']
    )
    
    # Configuration de la stratégie
    from src.backtest.backtest_engine import cycle_based_strategy
    
    strategy_params = {
        'cycle_classifier': classifier,
        'top_n': config['models']['sector_selection']['num_sectors'],
        'momentum_weight': config['models']['sector_selection']['momentum_weight']
    }
    
    # Optimisation des paramètres si demandée
    if optimize:
        logger.info("Optimisation des paramètres de la stratégie")
        
        param_grid = {
            'cycle_classifier': [classifier],
            'top_n': [2, 3, 4, 5],
            'momentum_weight': [0.3, 0.4, 0.5, 0.6, 0.7]
        }
        
        best_params, optimization_results = backtest_engine.run_strategy_optimization(
            cycle_based_strategy,
            param_grid,
            start_date=start_date,
            end_date=end_date,
            frequency=rebalance_frequency,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            metric='sharpe_ratio'
        )
        
        logger.info(f"Meilleurs paramètres: {best_params}")
        strategy_params = best_params
        
        # Sauvegarde des résultats d'optimisation
        results_dir = config['paths']['results']
        ensure_dir(results_dir)
        optimization_results.to_csv(os.path.join(results_dir, "optimization_results.csv"))
    
    # Exécution du backtest
    results, allocations = backtest_engine.run_simple_strategy(
        strategy_func=cycle_based_strategy,
        strategy_params=strategy_params,
        start_date=start_date,
        end_date=end_date,
        frequency=rebalance_frequency,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    # Calcul des métriques de performance
    metrics = backtest_engine.calculate_performance_metrics(results)
    
    # Affichage des métriques principales
    logger.info(f"Rendement annualisé: {metrics['annualized_return']:.2%}")
    logger.info(f"Volatilité annualisée: {metrics['volatility']:.2%}")
    logger.info(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Drawdown maximum: {metrics['max_drawdown']:.2%}")
    
    # Sauvegarde des résultats
    results_dir = config['paths']['results']
    ensure_dir(results_dir)
    results.to_csv(os.path.join(results_dir, "backtest_results.csv"))
    allocations.to_csv(os.path.join(results_dir, "sector_allocations.csv"))
    
    logger.info("Backtesting terminé")
    return results, allocations, metrics


def generate_report(results, allocations, config):
    """
    Génère un rapport de performance.

    Args:
        results (pd.DataFrame): Résultats du backtest.
        allocations (pd.DataFrame): Allocations sectorielles.
        config (dict): Configuration du projet.
    """
    logger.info("Génération du rapport de performance")
    
    # Création du répertoire des rapports
    reports_dir = config['paths']['reports']
    ensure_dir(reports_dir)
    visualizations_dir = os.path.join(reports_dir, 'visualizations')
    ensure_dir(visualizations_dir)
    
    # Création du backtest engine pour générer le rapport
    backtest_engine = BacktestEngine(
        sector_data=None,  # Non nécessaire pour la génération du rapport
        benchmark=config['backtest']['benchmark'],
        risk_free_rate=config['backtest']['risk_free_rate']
    )
    
    # Génération du rapport
    report = backtest_engine.generate_performance_report(
        results, 
        allocations,
        output_file=os.path.join(reports_dir, "performance_report.json")
    )
    
    # Création du visualiseur de performances
    visualizer = PerformanceVisualizer(
        style=config['visualization']['style'],
        theme=config['visualization']['theme'],
        interactive=config['visualization']['interactive']
    )
    
    # Génération des visualisations
    dashboard = visualizer.generate_performance_dashboard(
        results,
        allocations,
        output_dir=visualizations_dir,
        filename_prefix="strategie_rotation",
        format=config['visualization']['format'],
        dpi=config['visualization']['dpi'],
        include_metrics=True
    )
    
    logger.info(f"Rapport généré et sauvegardé dans {reports_dir}")
    logger.info(f"Visualisations sauvegardées dans {visualizations_dir}")


def generate_recommendations(macro_data, sector_data, classifier, config):
    """
    Génère des recommandations sectorielles actuelles.

    Args:
        macro_data (pd.DataFrame): Données macroéconomiques.
        sector_data (pd.DataFrame): Données sectorielles.
        classifier (EconomicCycleClassifier): Modèle de classification des cycles.
        config (dict): Configuration du projet.
    """
    logger.info("Génération des recommandations sectorielles actuelles")
    
    # Création du sélecteur de secteurs
    selector = SectorSelector(cycle_classifier=classifier)
    
    # Identification de la phase économique actuelle
    current_phase = classifier.predict(macro_data).iloc[-1]
    logger.info(f"Phase économique actuelle: {current_phase}")
    
    # Sélection des secteurs
    weights = selector.select_sectors(
        macro_data=macro_data,
        sector_data=sector_data,
        num_sectors=config['models']['sector_selection']['num_sectors'],
        momentum_weight=config['models']['sector_selection']['momentum_weight']
    )
    
    # Affichage des recommandations
    logger.info("Recommandations sectorielles:")
    for sector, weight in weights.items():
        logger.info(f"  {sector}: {weight:.2%}")
    
    # Sauvegarde des recommandations
    results_dir = config['paths']['results']
    ensure_dir(results_dir)
    
    recommendations = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'phase': current_phase,
        'sectors': {sector: float(weight) for sector, weight in weights.items()}
    }
    
    import json
    with open(os.path.join(results_dir, "current_recommendations.json"), 'w') as f:
        json.dump(recommendations, f, indent=4)
    
    logger.info(f"Recommandations sauvegardées dans {os.path.join(results_dir, 'current_recommendations.json')}")


def run_pipeline(args):
    """
    Exécute le pipeline complet.

    Args:
        args: Arguments de ligne de commande.
    """
    # Chargement de la configuration
    config = load_config(args.config)
    logger.info(f"Configuration chargée depuis {args.config}")
    
    # Collecte des données
    if args.collect or args.all:
        macro_data, sector_data = collect_data(config)
    else:
        macro_data, sector_data = load_data(config)
    
    # Entraînement du modèle
    if args.train or args.all:
        classifier = train_model(macro_data, config)
    else:
        classifier = load_model(config)
    
    # Backtesting de la stratégie
    if args.backtest or args.all:
        results, allocations, metrics = run_backtest(
            macro_data, sector_data, classifier, config, optimize=args.optimize
        )
    elif args.report or args.recommend:
        # Chargement des résultats du backtest si nécessaire pour le rapport ou les recommandations
        results_dir = config['paths']['results']
        results_path = os.path.join(results_dir, "backtest_results.csv")
        allocations_path = os.path.join(results_dir, "sector_allocations.csv")
        
        if os.path.exists(results_path) and os.path.exists(allocations_path):
            results = pd.read_csv(results_path, index_col=0, parse_dates=True)
            allocations = pd.read_csv(allocations_path, index_col=0, parse_dates=True)
        else:
            logger.warning("Résultats de backtest non trouvés, exécution du backtest")
            results, allocations, metrics = run_backtest(
                macro_data, sector_data, classifier, config, optimize=False
            )
    
    # Génération du rapport
    if args.report or args.all:
        generate_report(results, allocations, config)
    
    # Génération des recommandations actuelles
    if args.recommend or args.all:
        generate_recommendations(macro_data, sector_data, classifier, config)
    
    logger.info("Pipeline terminé")


if __name__ == "__main__":
    # Création du répertoire racine du projet si nécessaire
    project_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_root)
    
    # Parsing des arguments
    args = parse_arguments()
    
    # Si aucune action n'est spécifiée, utiliser --all
    if not any([args.collect, args.train, args.backtest, args.optimize, args.report, args.recommend, args.all]):
        logger.info("Aucune action spécifiée, exécution de l'ensemble du pipeline")
        args.all = True
    
    # Exécution du pipeline
    run_pipeline(args)
