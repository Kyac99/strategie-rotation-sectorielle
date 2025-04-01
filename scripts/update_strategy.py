#!/usr/bin/env python
"""
Script d'automatisation pour la mise à jour et le suivi de la stratégie de rotation sectorielle.

Ce script effectue les opérations suivantes:
1. Mise à jour des données macroéconomiques et sectorielles
2. Identification de la phase économique actuelle
3. Génération des recommandations sectorielles
4. Production d'un rapport résumant la situation actuelle
5. Envoi des résultats par email (optionnel)

Usage:
    python update_strategy.py --email --report

Options:
    --email     Envoyer les résultats par email
    --report    Générer un rapport détaillé (HTML)
    --backtest  Mettre à jour le backtest avec les nouvelles données
    --verbose   Afficher des informations détaillées pendant l'exécution
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Ajout du répertoire racine au path pour l'importation des modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importation des modules personnalisés
from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector
from src.models.economic_cycle_classifier import EconomicCycleClassifier
from src.models.sector_selector import SectorSelector
from src.backtest.backtest_engine import BacktestEngine
from src.visualization.performance_visualizer import PerformanceVisualizer
from src.utils.common_utils import (
    load_config, ensure_dir, save_to_json, load_from_json,
    calculate_performance_metrics
)


def setup_logging(verbose=False):
    """Configure le système de logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(project_root, 'logs', 'update_strategy.log'))
        ]
    )
    
    # Création du répertoire de logs s'il n'existe pas
    ensure_dir(os.path.join(project_root, 'logs'))
    
    return logging.getLogger(__name__)


def load_environment():
    """Charge les variables d'environnement et la configuration."""
    # Chargement des variables d'environnement
    load_dotenv(os.path.join(project_root, '.env'))
    
    # Chargement de la configuration
    config = load_config(os.path.join(project_root, 'config', 'config.yaml'))
    
    return config


def update_data(config, logger):
    """Met à jour les données macroéconomiques et sectorielles."""
    logger.info("Mise à jour des données...")
    
    # Chemins des données
    data_dir = os.path.join(project_root, config['paths']['data_processed'])
    ensure_dir(data_dir)
    
    macro_data_path = os.path.join(data_dir, "macro_data.csv")
    sector_data_path = os.path.join(data_dir, "sector_data.csv")
    
    # Collecte des données macroéconomiques
    logger.info("Collecte des données macroéconomiques...")
    macro_collector = MacroDataCollector(api_key=os.getenv('FRED_API_KEY'))
    
    # Utilisation des données existantes comme base si disponibles
    if os.path.exists(macro_data_path):
        existing_data = pd.read_csv(macro_data_path, index_col=0, parse_dates=True)
        start_date = existing_data.index[-1] + timedelta(days=1)
        logger.info(f"Dernière date des données macro: {existing_data.index[-1].strftime('%Y-%m-%d')}")
        logger.info(f"Collecte des nouvelles données à partir du {start_date.strftime('%Y-%m-%d')}")
        
        # Collecte des nouvelles données
        new_data = macro_collector.get_all_series(start_date=start_date.strftime('%Y-%m-%d'), frequency='m')
        
        if new_data is not None and len(new_data) > 0:
            # Prétraitement des nouvelles données
            new_processed = macro_collector.preprocess_data(new_data)
            
            # Fusion des données existantes et nouvelles
            combined_data = pd.concat([existing_data, new_processed])
            logger.info(f"Nouvelles données macro collectées: {len(new_processed)} observations")
        else:
            combined_data = existing_data
            logger.info("Aucune nouvelle donnée macro disponible")
        
        # Sauvegarde des données combinées
        combined_data.to_csv(macro_data_path)
        logger.info(f"Données macro mises à jour et sauvegardées dans {macro_data_path}")
        
        macro_data = combined_data
    else:
        # Collecte de toutes les données depuis 2000
        logger.info("Aucune donnée macro existante, collecte de toutes les données depuis 2000...")
        macro_data = macro_collector.get_all_series(start_date="2000-01-01", frequency='m')
        processed_macro = macro_collector.preprocess_data(macro_data)
        processed_macro.to_csv(macro_data_path)
        logger.info(f"Données macro collectées et sauvegardées: {len(processed_macro)} observations")
        
        macro_data = processed_macro
    
    # Collecte des données sectorielles
    logger.info("Collecte des données sectorielles...")
    sector_collector = SectorDataCollector()
    
    # Utilisation des données existantes comme base si disponibles
    if os.path.exists(sector_data_path):
        existing_sectors = pd.read_csv(sector_data_path, index_col=0, parse_dates=True)
        start_date = existing_sectors.index[-1] + timedelta(days=1)
        logger.info(f"Dernière date des données sectorielles: {existing_sectors.index[-1].strftime('%Y-%m-%d')}")
        logger.info(f"Collecte des nouvelles données à partir du {start_date.strftime('%Y-%m-%d')}")
        
        # Collecte des nouvelles données
        new_etf_data = sector_collector.get_all_etf_data(start_date=start_date.strftime('%Y-%m-%d'))
        
        if new_etf_data and any(len(df) > 0 for df in new_etf_data.values()):
            # Prétraitement des nouvelles données
            new_sectors = sector_collector.preprocess_data(new_etf_data)
            
            # Fusion des données existantes et nouvelles
            # Note: Cette fusion peut être complexe car preprocess_data génère des colonnes dérivées
            # Ici nous sauvegardons juste les nouvelles données pour éviter les problèmes
            logger.info(f"Nouvelles données sectorielles collectées")
            new_sectors.to_csv(sector_data_path)
            logger.info(f"Données sectorielles mises à jour et sauvegardées dans {sector_data_path}")
            
            sector_data = new_sectors
        else:
            sector_data = existing_sectors
            logger.info("Aucune nouvelle donnée sectorielle disponible")
    else:
        # Collecte de toutes les données depuis 2000
        logger.info("Aucune donnée sectorielle existante, collecte de toutes les données depuis 2000...")
        etf_data = sector_collector.get_all_etf_data(start_date="2000-01-01")
        processed_sectors = sector_collector.preprocess_data(etf_data)
        processed_sectors.to_csv(sector_data_path)
        logger.info(f"Données sectorielles collectées et sauvegardées: {len(processed_sectors)} observations")
        
        sector_data = processed_sectors
    
    return macro_data, sector_data


def identify_economic_phase(macro_data, config, logger):
    """Identifie la phase actuelle du cycle économique."""
    logger.info("Identification de la phase économique actuelle...")
    
    # Chemin du modèle
    models_dir = os.path.join(project_root, config['paths']['models'])
    ensure_dir(models_dir)
    cycle_model_path = os.path.join(models_dir, "economic_cycle_classifier.joblib")
    
    # Chargement ou entraînement du modèle
    if os.path.exists(cycle_model_path):
        cycle_model = EconomicCycleClassifier.load_model(cycle_model_path)
        logger.info(f"Modèle de classification des cycles économiques chargé depuis {cycle_model_path}")
    else:
        logger.info("Entraînement d'un nouveau modèle de classification des cycles économiques...")
        cycle_model = EconomicCycleClassifier(supervised=config['models']['economic_cycle']['supervised'])
        cycle_model.fit(macro_data)
        cycle_model.save_model(cycle_model_path)
        logger.info(f"Modèle entraîné et sauvegardé dans {cycle_model_path}")
    
    # Prédiction des phases
    phases = cycle_model.predict(macro_data)
    
    # Sauvegarde des phases
    phases_path = os.path.join(project_root, config['paths']['data_processed'], "economic_phases.csv")
    phases_df = pd.DataFrame({'phase': phases}, index=phases.index)
    phases_df.to_csv(phases_path)
    logger.info(f"Phases économiques identifiées et sauvegardées dans {phases_path}")
    
    # Phase actuelle
    current_phase = phases.iloc[-1]
    logger.info(f"Phase économique actuelle: {current_phase}")
    
    # Détermination de la durée de la phase actuelle
    phase_start = phases[phases != current_phase].index[-1] if any(phases != current_phase) else phases.index[0]
    phase_duration = (phases.index[-1] - phase_start).days
    logger.info(f"Durée de la phase actuelle: {phase_duration} jours")
    
    return cycle_model, current_phase, phase_duration


def generate_recommendations(macro_data, sector_data, cycle_model, config, logger):
    """Génère les recommandations sectorielles basées sur la phase économique et le momentum."""
    logger.info("Génération des recommandations sectorielles...")
    
    # Création du sélecteur de secteurs
    selector = SectorSelector(cycle_classifier=cycle_model)
    
    # Paramètres de sélection
    num_sectors = config['models']['sector_selection']['num_sectors']
    momentum_weight = config['models']['sector_selection']['momentum_weight']
    
    # Sélection des secteurs
    weights = selector.select_sectors(
        macro_data=macro_data,
        sector_data=sector_data,
        num_sectors=num_sectors,
        momentum_weight=momentum_weight
    )
    
    # Sauvegarde des recommandations
    recommendations_path = os.path.join(
        project_root, config['paths']['results'], "latest_recommendations.json"
    )
    ensure_dir(os.path.dirname(recommendations_path))
    
    # Conversion des poids en pourcentages pour le JSON
    weights_pct = {k: float(v * 100) for k, v in weights.items()}
    
    # Ajout des métadonnées
    recommendations = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'economic_phase': selector.identify_current_cycle(macro_data),
        'weights': weights_pct,
        'parameters': {
            'num_sectors': num_sectors,
            'momentum_weight': momentum_weight
        }
    }
    
    save_to_json(recommendations, recommendations_path)
    logger.info(f"Recommandations générées et sauvegardées dans {recommendations_path}")
    
    # Affichage des recommandations
    logger.info("Recommandations sectorielles:")
    for sector, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {sector}: {weight:.2%}")
    
    return weights, recommendations


def update_backtest(macro_data, sector_data, cycle_model, config, logger):
    """Met à jour le backtest avec les données les plus récentes."""
    logger.info("Mise à jour du backtest...")
    
    # Paramètres de backtesting
    start_date = config['backtest']['start_date']
    end_date = macro_data.index[-1].strftime('%Y-%m-%d')  # Utilisation de la dernière date disponible
    rebalance_frequency = config['backtest']['rebalance_frequency']
    initial_capital = config['backtest']['initial_capital']
    transaction_cost = config['backtest']['transaction_cost']
    
    # Création du moteur de backtesting
    backtest = BacktestEngine(
        sector_data=sector_data,
        macro_data=macro_data,
        benchmark=config['backtest']['benchmark'],
        risk_free_rate=config['backtest']['risk_free_rate']
    )
    
    # Importation de la stratégie
    from src.backtest.backtest_engine import cycle_based_strategy
    
    # Exécution du backtest
    results, allocations = backtest.run_simple_strategy(
        strategy_func=cycle_based_strategy,
        strategy_params={
            'cycle_classifier': cycle_model,
            'top_n': config['models']['sector_selection']['num_sectors'],
            'momentum_weight': config['models']['sector_selection']['momentum_weight']
        },
        start_date=start_date,
        end_date=end_date,
        frequency=rebalance_frequency,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    # Calcul des métriques de performance
    metrics = backtest.calculate_performance_metrics(results)
    
    # Sauvegarde des résultats
    results_path = os.path.join(project_root, config['paths']['results'], "backtest_results.csv")
    allocations_path = os.path.join(project_root, config['paths']['results'], "backtest_allocations.csv")
    metrics_path = os.path.join(project_root, config['paths']['results'], "backtest_metrics.json")
    
    ensure_dir(os.path.dirname(results_path))
    results.to_csv(results_path)
    allocations.to_csv(allocations_path)
    save_to_json(metrics, metrics_path)
    
    logger.info(f"Backtest mis à jour et résultats sauvegardés")
    logger.info(f"  Période: {start_date} à {end_date}")
    logger.info(f"  Rendement annualisé: {metrics['annualized_return']:.2%}")
    logger.info(f"  Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    return results, allocations, metrics


def generate_report(macro_data, sector_data, cycle_model, recommendations, backtest_results, config, logger):
    """Génère un rapport détaillé sur la situation actuelle et les recommandations."""
    logger.info("Génération du rapport...")
    
    # Création du visualiseur
    visualizer = PerformanceVisualizer(
        style=config['visualization']['style'],
        theme=config['visualization']['theme'],
        interactive=config['visualization']['interactive']
    )
    
    # Répertoire pour les visualisations
    visualizations_dir = os.path.join(project_root, config['paths']['reports'], 'visualizations')
    ensure_dir(visualizations_dir)
    
    # Visualisation des indicateurs économiques clés
    indicators = ['GDPC1_YOY', 'INDPRO_YOY', 'UNRATE', 'CPIAUCSL_YOY', 'FEDFUNDS', 'T10Y2Y']
    available_indicators = [ind for ind in indicators if ind in macro_data.columns]
    
    # Dernières valeurs des indicateurs
    latest_indicators = {ind: macro_data[ind].iloc[-1] for ind in available_indicators}
    
    # Détermination de la phase économique
    phases = cycle_model.predict(macro_data)
    current_phase = phases.iloc[-1]
    phase_history = phases.value_counts().to_dict()
    
    # Visualisation des performances sectorielles récentes
    sector_etfs = [s for s in sector_data.columns if '_' not in s and len(s) <= 4]
    recent_returns = {}
    
    for period in [1, 3, 6, 12]:  # 1, 3, 6, 12 mois
        if len(sector_data) > period:
            period_returns = {}
            for etf in sector_etfs:
                if etf in sector_data.columns:
                    start_price = sector_data[etf].iloc[-period-1] if period < len(sector_data) else sector_data[etf].iloc[0]
                    end_price = sector_data[etf].iloc[-1]
                    period_returns[etf] = (end_price / start_price - 1) * 100
            
            recent_returns[f"{period}m"] = period_returns
    
    # Génération du rapport HTML
    report_date = datetime.now().strftime('%Y-%m-%d')
    report_path = os.path.join(project_root, config['paths']['reports'], f"market_report_{report_date}.html")
    
    # Création du contenu HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport de Marché - {report_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            .indicator {{ display: flex; margin-bottom: 10px; }}
            .indicator-name {{ width: 250px; font-weight: bold; }}
            .indicator-value {{ width: 100px; text-align: right; }}
            .recommendations {{ margin-top: 20px; }}
            .recommendation {{ display: flex; margin-bottom: 10px; }}
            .sector {{ width: 100px; font-weight: bold; }}
            .weight {{ width: 100px; text-align: right; }}
            .progress-bar {{ flex-grow: 1; height: 20px; background-color: #ecf0f1; margin: 0 20px; }}
            .progress {{ height: 100%; background-color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport de Marché - {report_date}</h1>
            
            <div class="section">
                <h2>Situation Économique</h2>
                <p>Phase économique actuelle: <strong>{current_phase}</strong></p>
                
                <h3>Indicateurs Économiques Clés</h3>
                <div class="indicators">
    """
    
    # Ajout des indicateurs
    for ind, value in latest_indicators.items():
        html_content += f"""
                    <div class="indicator">
                        <div class="indicator-name">{ind}</div>
                        <div class="indicator-value">{value:.2f}</div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Recommandations Sectorielles</h2>
                <p>Basées sur la phase économique actuelle et le momentum récent.</p>
                
                <div class="recommendations">
    """
    
    # Ajout des recommandations
    for sector, weight in sorted(recommendations['weights'].items(), key=lambda x: x[1], reverse=True):
        html_content += f"""
                    <div class="recommendation">
                        <div class="sector">{sector}</div>
                        <div class="progress-bar">
                            <div class="progress" style="width: {weight}%;"></div>
                        </div>
                        <div class="weight">{weight:.1f}%</div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Sectorielle Récente</h2>
                <table>
                    <tr>
                        <th>Secteur</th>
    """
    
    # Ajout des entêtes de périodes
    for period in recent_returns.keys():
        html_content += f"<th>Rendement {period}</th>\n"
    
    html_content += """
                    </tr>
    """
    
    # Ajout des performances sectorielles
    for sector in sector_etfs:
        if all(sector in returns for returns in recent_returns.values()):
            html_content += f"<tr><td>{sector}</td>\n"
            
            for period, returns in recent_returns.items():
                value = returns[sector]
                css_class = "positive" if value > 0 else "negative"
                html_content += f"<td class='{css_class}'>{value:.2f}%</td>\n"
            
            html_content += "</tr>\n"
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Performance de la Stratégie</h2>
    """
    
    # Ajout des métriques de performance si disponibles
    if backtest_results is not None:
        metrics_path = os.path.join(project_root, config['paths']['results'], "backtest_metrics.json")
        if os.path.exists(metrics_path):
            metrics = load_from_json(metrics_path)
            
            html_content += f"""
                <p>Rendement annualisé: <strong>{metrics['annualized_return']*100:.2f}%</strong></p>
                <p>Volatilité annualisée: <strong>{metrics['volatility']*100:.2f}%</strong></p>
                <p>Ratio de Sharpe: <strong>{metrics['sharpe_ratio']:.2f}</strong></p>
                <p>Drawdown maximum: <strong>{metrics['max_drawdown']*100:.2f}%</strong></p>
            """
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>Ce rapport présente une analyse de la situation économique actuelle et des recommandations d'allocation sectorielle basées sur notre stratégie de rotation sectorielle dynamique.</p>
                <p>Les recommandations sont générées automatiquement à partir de l'analyse des cycles économiques et du momentum des secteurs.</p>
                <p>Pour plus de détails, veuillez consulter les notebooks d'analyse et de backtesting.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Sauvegarde du rapport
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Rapport généré et sauvegardé dans {report_path}")
    
    return report_path


def send_email_report(report_path, recommendations, config, logger):
    """Envoie le rapport par email."""
    logger.info("Envoi du rapport par email...")
    
    # Configuration de l'email
    email_config = config.get('email', {})
    sender_email = os.getenv('EMAIL_SENDER', email_config.get('sender'))
    receiver_email = os.getenv('EMAIL_RECEIVER', email_config.get('receiver'))
    password = os.getenv('EMAIL_PASSWORD')
    smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    
    if not all([sender_email, receiver_email, password]):
        logger.error("Configuration email incomplète. Impossible d'envoyer le rapport.")
        return False
    
    # Création du message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"Rapport de Rotation Sectorielle - {datetime.now().strftime('%Y-%m-%d')}"
    
    # Corps du message
    body = f"""
    Bonjour,
    
    Voici le rapport de rotation sectorielle du {datetime.now().strftime('%d/%m/%Y')}.
    
    Phase économique actuelle: {recommendations['economic_phase']}
    
    Recommandations sectorielles:
    """
    
    for sector, weight in sorted(recommendations['weights'].items(), key=lambda x: x[1], reverse=True):
        body += f"- {sector}: {weight:.1f}%\n"
    
    body += """
    Le rapport complet est joint à ce message.
    
    Cordialement,
    Système de Rotation Sectorielle Automatisé
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Pièce jointe (rapport HTML)
    with open(report_path, "rb") as f:
        attachment = MIMEApplication(f.read(), _subtype="html")
        attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_path))
        msg.attach(attachment)
    
    # Envoi de l'email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        logger.info(f"Rapport envoyé par email à {receiver_email}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email: {e}")
        return False


def main():
    """Fonction principale du script."""
    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Script d'automatisation pour la stratégie de rotation sectorielle")
    parser.add_argument('--email', action='store_true', help="Envoyer les résultats par email")
    parser.add_argument('--report', action='store_true', help="Générer un rapport détaillé")
    parser.add_argument('--backtest', action='store_true', help="Mettre à jour le backtest")
    parser.add_argument('--verbose', action='store_true', help="Afficher des informations détaillées")
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging(args.verbose)
    logger.info("Démarrage du script d'automatisation...")
    
    # Chargement de la configuration
    config = load_environment()
    
    # Mise à jour des données
    macro_data, sector_data = update_data(config, logger)
    
    # Identification de la phase économique
    cycle_model, current_phase, phase_duration = identify_economic_phase(macro_data, config, logger)
    
    # Génération des recommandations
    weights, recommendations = generate_recommendations(macro_data, sector_data, cycle_model, config, logger)
    
    # Mise à jour du backtest si demandé
    backtest_results = None
    if args.backtest:
        results, allocations, metrics = update_backtest(macro_data, sector_data, cycle_model, config, logger)
        backtest_results = results
    
    # Génération du rapport si demandé
    if args.report:
        report_path = generate_report(macro_data, sector_data, cycle_model, recommendations, backtest_results, config, logger)
        
        # Envoi du rapport par email si demandé
        if args.email:
            send_email_report(report_path, recommendations, config, logger)
    
    logger.info("Script d'automatisation terminé avec succès.")


if __name__ == "__main__":
    main()
