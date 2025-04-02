#!/usr/bin/env python
"""
Script pour la mise à jour automatique des données.

Ce script peut être exécuté quotidiennement, hebdomadairement ou mensuellement
pour maintenir les données à jour. Il peut être programmé avec cron ou équivalent.

Utilisation:
    python update_data.py [--force] [--log-level=INFO] [--config=config.yaml]

Arguments:
    --force: Forcer la mise à jour même si les données sont récentes
    --log-level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    --config: Chemin vers le fichier de configuration

Exemple:
    python update_data.py --force --log-level=DEBUG
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import time
import yaml
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector
from src.models.economic_cycle_classifier import EconomicCycleClassifier
from src.utils.common_utils import load_config, ensure_dir


def setup_logging(log_level='INFO', log_file=None):
    """
    Configure le logging.
    
    Args:
        log_level (str): Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Chemin du fichier de log
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )


def check_update_needed(file_path, max_age_days=1, force=False):
    """
    Vérifie si une mise à jour est nécessaire.
    
    Args:
        file_path (str): Chemin du fichier à vérifier
        max_age_days (int): Âge maximal du fichier en jours
        force (bool): Forcer la mise à jour
        
    Returns:
        bool: True si une mise à jour est nécessaire, False sinon
    """
    if force:
        return True
    
    if not os.path.exists(file_path):
        return True
    
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    now = datetime.now()
    age = now - file_time
    
    return age.days >= max_age_days


def update_macro_data(config, force=False):
    """
    Met à jour les données macroéconomiques.
    
    Args:
        config (dict): Configuration du projet
        force (bool): Forcer la mise à jour
        
    Returns:
        pd.DataFrame: Données macroéconomiques mises à jour
    """
    logger = logging.getLogger('update_macro_data')
    
    # Chemin du fichier de données
    data_path = os.path.join(parent_dir, config['paths']['data_processed'], 'macro_data.csv')
    
    # Vérification si la mise à jour est nécessaire
    if not check_update_needed(data_path, max_age_days=30, force=force):
        logger.info(f"Les données macroéconomiques sont à jour (max_age_days=30). Chargement depuis {data_path}")
        import pandas as pd
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Création du collecteur
    api_key = config.get('data_sources', {}).get('fred', {}).get('api_key', None)
    if not api_key:
        logger.warning("Aucune clé API FRED trouvée dans la configuration. Vérification de la variable d'environnement.")
        api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        logger.error("Aucune clé API FRED trouvée. Impossible de mettre à jour les données macroéconomiques.")
        return None
    
    try:
        collector = MacroDataCollector(api_key=api_key)
        
        # Collecte des données
        logger.info("Collecte des données macroéconomiques...")
        start_time = time.time()
        macro_data = collector.get_all_series(frequency='m')
        
        # Prétraitement des données
        logger.info("Prétraitement des données macroéconomiques...")
        processed_data = collector.preprocess_data(macro_data)
        
        # Sauvegarde des données
        ensure_dir(os.path.dirname(data_path))
        processed_data.to_csv(data_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Données macroéconomiques mises à jour en {elapsed_time:.1f} secondes.")
        logger.info(f"Données sauvegardées dans {data_path}")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des données macroéconomiques: {e}")
        return None


def update_sector_data(config, force=False):
    """
    Met à jour les données sectorielles.
    
    Args:
        config (dict): Configuration du projet
        force (bool): Forcer la mise à jour
        
    Returns:
        pd.DataFrame: Données sectorielles mises à jour
    """
    logger = logging.getLogger('update_sector_data')
    
    # Chemin du fichier de données
    data_path = os.path.join(parent_dir, config['paths']['data_processed'], 'sector_data.csv')
    
    # Vérification si la mise à jour est nécessaire
    if not check_update_needed(data_path, max_age_days=1, force=force):
        logger.info(f"Les données sectorielles sont à jour (max_age_days=1). Chargement depuis {data_path}")
        import pandas as pd
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    try:
        collector = SectorDataCollector()
        
        # Collecte des données
        logger.info("Collecte des données sectorielles...")
        start_time = time.time()
        etf_data = collector.get_all_etf_data()
        
        # Prétraitement des données
        logger.info("Prétraitement des données sectorielles...")
        processed_data = collector.preprocess_data(etf_data)
        
        # Sauvegarde des données
        ensure_dir(os.path.dirname(data_path))
        processed_data.to_csv(data_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Données sectorielles mises à jour en {elapsed_time:.1f} secondes.")
        logger.info(f"Données sauvegardées dans {data_path}")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des données sectorielles: {e}")
        return None


def update_economic_phases(config, macro_data, force=False):
    """
    Met à jour les phases du cycle économique.
    
    Args:
        config (dict): Configuration du projet
        macro_data (pd.DataFrame): Données macroéconomiques
        force (bool): Forcer la mise à jour
        
    Returns:
        pd.DataFrame: Phases économiques mises à jour
    """
    logger = logging.getLogger('update_economic_phases')
    
    if macro_data is None:
        logger.error("Impossible de mettre à jour les phases économiques: données macroéconomiques non disponibles.")
        return None
    
    # Chemins des fichiers
    phases_path = os.path.join(parent_dir, config['paths']['data_processed'], 'economic_phases.csv')
    model_path = os.path.join(parent_dir, config['paths']['models'], 'economic_cycle_classifier.joblib')
    
    # Vérification si la mise à jour est nécessaire
    if not force and os.path.exists(phases_path) and os.path.getmtime(phases_path) >= os.path.getmtime(
            os.path.join(parent_dir, config['paths']['data_processed'], 'macro_data.csv')):
        logger.info(f"Les phases économiques sont à jour. Chargement depuis {phases_path}")
        import pandas as pd
        return pd.read_csv(phases_path, index_col=0, parse_dates=True)
    
    try:
        # Chargement ou entraînement du modèle
        if os.path.exists(model_path) and not force:
            logger.info(f"Chargement du modèle depuis {model_path}")
            classifier = EconomicCycleClassifier.load_model(model_path)
        else:
            logger.info("Entraînement d'un nouveau modèle...")
            classifier = EconomicCycleClassifier(supervised=config['models']['economic_cycle']['supervised'])
            classifier.fit(macro_data)
            
            # Sauvegarde du modèle
            ensure_dir(os.path.dirname(model_path))
            classifier.save_model(model_path)
            logger.info(f"Modèle sauvegardé dans {model_path}")
        
        # Prédiction des phases
        logger.info("Identification des phases économiques...")
        phases = classifier.predict(macro_data)
        
        # Création du DataFrame
        import pandas as pd
        phases_df = pd.DataFrame({'phase': phases}, index=phases.index)
        
        # Sauvegarde des phases
        ensure_dir(os.path.dirname(phases_path))
        phases_df.to_csv(phases_path)
        logger.info(f"Phases économiques sauvegardées dans {phases_path}")
        
        return phases_df
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des phases économiques: {e}")
        return None


def main():
    """Fonction principale du script."""
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Mise à jour des données pour la stratégie de rotation sectorielle.')
    parser.add_argument('--force', action='store_true', help='Forcer la mise à jour même si les données sont récentes')
    parser.add_argument('--log-level', default='INFO', help='Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--config', default=os.path.join(parent_dir, 'config', 'config.yaml'),
                       help='Chemin vers le fichier de configuration')
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_file = os.path.join(parent_dir, 'logs', 'update_data.log')
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger('update_data')
    logger.info("Démarrage de la mise à jour des données...")
    
    # Chargement de la configuration
    config = load_config(args.config)
    if not config:
        logger.error(f"Impossible de charger la configuration depuis {args.config}")
        return 1
    
    # Mise à jour des données macroéconomiques
    macro_data = update_macro_data(config, args.force)
    
    # Mise à jour des données sectorielles
    sector_data = update_sector_data(config, args.force)
    
    # Mise à jour des phases économiques
    if macro_data is not None:
        phases = update_economic_phases(config, macro_data, args.force)
    
    # Résumé
    logger.info("Résumé de la mise à jour:")
    logger.info(f"- Données macroéconomiques: {'OK' if macro_data is not None else 'ÉCHEC'}")
    logger.info(f"- Données sectorielles: {'OK' if sector_data is not None else 'ÉCHEC'}")
    logger.info(f"- Phases économiques: {'OK' if 'phases' in locals() and phases is not None else 'ÉCHEC'}")
    
    # Statut de sortie
    if macro_data is None or sector_data is None or ('phases' in locals() and phases is None):
        logger.warning("La mise à jour n'est pas complète.")
        return 1
    
    logger.info("Mise à jour terminée avec succès.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
