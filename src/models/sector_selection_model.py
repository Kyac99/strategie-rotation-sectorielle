# -*- coding: utf-8 -*-
"""
Modèle de sélection sectorielle basé sur le cycle économique et le momentum.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sector_selection_model')

class SectorSelectionModel:
    """
    Modèle pour sélectionner les secteurs basé sur le cycle économique et le momentum.
    """
    
    def __init__(self, config_path='config/model_params.yaml', backtest_path='config/backtest_params.yaml'):
        """
        Initialise le modèle de sélection sectorielle avec les configurations nécessaires.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration des paramètres des modèles.
            backtest_path (str): Chemin vers le fichier de configuration des paramètres de backtest.
        """
        # Charger les configurations
        self.config_path = config_path
        self.backtest_path = backtest_path
        self.load_config()
        
        # Créer le répertoire de sauvegarde des modèles
        Path('models').mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """
        Charge les configurations depuis les fichiers YAML.
        """
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Configuration du modèle chargée depuis {self.config_path}")
            
            with open(self.backtest_path, 'r') as file:
                self.backtest_config = yaml.safe_load(file)
            logger.info(f"Configuration du backtest chargée depuis {self.backtest_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des configurations: {e}")
            raise
    
    def load_data(self):
        """
        Charge les données nécessaires pour la sélection sectorielle.
        
        Returns:
            tuple: Tuple contenant les données chargées.
        """
        try:
            # Charger les prédictions de cycle économique
            cycle_predictions = pd.read_parquet(Path('data/processed/cycle_predictions.parquet'))
            logger.info("Prédictions de cycle économique chargées")
            
            # Charger les features
            features = pd.read_parquet(Path('data/processed/engineered_features.parquet'))
            logger.info("Features chargées")
            
            # Charger les données brutes de prix pour le calcul des rendements
            processed_data = pd.read_parquet(Path('data/processed/merged_data.parquet'))
            logger.info("Données traitées chargées")
            
            return cycle_predictions, features, processed_data
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return None, None, None
    
    def select_sectors_by_cycle(self, cycle_phase):
        """
        Sélectionne les secteurs recommandés pour une phase donnée du cycle.
        
        Args:
            cycle_phase (str): Phase du cycle économique.
            
        Returns:
            list: Liste des secteurs recommandés pour cette phase.
        """
        try:
            # Obtenir le mapping des secteurs par cycle depuis la configuration
            sector_cycle_mapping = self.config['sector_cycle_mapping']
            
            # Vérifier si la phase existe dans le mapping
            if cycle_phase in sector_cycle_mapping:
                # Obtenir les secteurs recommandés pour cette phase
                recommended_sectors = sector_cycle_mapping[cycle_phase]['top_sectors']
                logger.debug(f"Secteurs recommandés pour la phase {cycle_phase}: {recommended_sectors}")
                return recommended_sectors
            else:
                logger.warning(f"Phase {cycle_phase} non trouvée dans le mapping des secteurs")
                return []
        
        except Exception as e:
            logger.error(f"Erreur lors de la sélection des secteurs par cycle: {e}")
            return []
    
    def rank_sectors_by_momentum(self, features, date):
        """
        Classe les secteurs par momentum à une date donnée.
        
        Args:
            features (pd.DataFrame): DataFrame contenant les features.
            date (pd.Timestamp): Date à laquelle classer les secteurs.
            
        Returns:
            list: Liste des secteurs classés par momentum décroissant.
        """
        try:
            # Extraire les colonnes de momentum composite
            momentum_cols = [col for col in features.columns if col.startswith('momentum_composite_')]
            
            if not momentum_cols:
                logger.error("Aucune colonne de momentum composite trouvée")
                return []
            
            # Extraire les valeurs de momentum à la date spécifiée
            if date in features.index:
                momentum_values = features.loc[date, momentum_cols]
                
                # Convertir les noms de colonnes en symboles d'ETF
                momentum_dict = {col.split('_')[-1]: momentum_values[col] for col in momentum_cols}
                
                # Trier les secteurs par momentum décroissant
                ranked_sectors = sorted(momentum_dict.keys(), key=lambda x: momentum_dict[x], reverse=True)
                
                logger.debug(f"Secteurs classés par momentum à {date}: {ranked_sectors}")
                return ranked_sectors
            else:
                logger.warning(f"Date {date} non trouvée dans les features")
                return []
        
        except Exception as e:
            logger.error(f"Erreur lors du classement des secteurs par momentum: {e}")
            return []
    
    def combine_cycle_and_momentum(self, cycle_sectors, momentum_sectors, alpha=0.7):
        """
        Combine les recommandations basées sur le cycle et le momentum.
        
        Args:
            cycle_sectors (list): Liste des secteurs recommandés par le cycle.
            momentum_sectors (list): Liste des secteurs classés par momentum.
            alpha (float): Pondération du cycle vs momentum (0 = momentum uniquement, 1 = cycle uniquement).
            
        Returns:
            list: Liste combinée des secteurs recommandés.
        """
        try:
            # Si l'une des listes est vide, retourner l'autre
            if not cycle_sectors:
                return momentum_sectors[:self.backtest_config['strategy_params']['num_sectors']]
            if not momentum_sectors:
                return cycle_sectors[:self.backtest_config['strategy_params']['num_sectors']]
            
            # Calculer les scores pour chaque secteur
            all_sectors = list(set(cycle_sectors + momentum_sectors))
            scores = {}
            
            for sector in all_sectors:
                # Score basé sur le cycle (position inverse dans la liste)
                cycle_score = 0
                if sector in cycle_sectors:
                    cycle_score = len(cycle_sectors) - cycle_sectors.index(sector)
                
                # Score basé sur le momentum (position inverse dans la liste)
                momentum_score = 0
                if sector in momentum_sectors:
                    momentum_score = len(momentum_sectors) - momentum_sectors.index(sector)
                
                # Score pondéré
                scores[sector] = alpha * cycle_score + (1 - alpha) * momentum_score
            
            # Trier les secteurs par score décroissant
            ranked_sectors = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
            # Sélectionner le nombre de secteurs spécifié dans la configuration
            num_sectors = self.backtest_config['strategy_params']['num_sectors']
            selected_sectors = ranked_sectors[:num_sectors]
            
            logger.debug(f"Secteurs sélectionnés après combinaison: {selected_sectors}")
            return selected_sectors
        
        except Exception as e:
            logger.error(f"Erreur lors de la combinaison des recommandations: {e}")
            return []
    
    def calculate_weights(self, selected_sectors, features, date):
        """
        Calcule les poids pour les secteurs sélectionnés.
        
        Args:
            selected_sectors (list): Liste des secteurs sélectionnés.
            features (pd.DataFrame): DataFrame contenant les features.
            date (pd.Timestamp): Date à laquelle calculer les poids.
            
        Returns:
            dict: Dictionnaire des poids par secteur.
        """
        try:
            # Obtenir le schéma de pondération depuis la configuration
            weighting_scheme = self.backtest_config['strategy_params']['weighting_scheme']
            
            # Initialiser le dictionnaire des poids
            weights = {}
            
            if weighting_scheme == 'equal':
                # Pondération égale
                equal_weight = 1.0 / len(selected_sectors)
                weights = {sector: equal_weight for sector in selected_sectors}
            
            elif weighting_scheme == 'momentum_weighted':
                # Pondération basée sur le momentum
                momentum_cols = [f'momentum_composite_{sector}' for sector in selected_sectors]
                momentum_values = features.loc[date, momentum_cols]
                
                # Convertir en valeurs positives en ajoutant une constante si nécessaire
                if min(momentum_values) < 0:
                    momentum_values = momentum_values - min(momentum_values) + 0.01
                
                # Normaliser pour que la somme soit 1
                total = momentum_values.sum()
                if total > 0:
                    weights = {sector: momentum_values[f'momentum_composite_{sector}'] / total 
                              for sector in selected_sectors}
                else:
                    # En cas de problème, utiliser une pondération égale
                    equal_weight = 1.0 / len(selected_sectors)
                    weights = {sector: equal_weight for sector in selected_sectors}
            
            elif weighting_scheme == 'volatility_weighted':
                # Pondération inverse à la volatilité
                volatility_cols = [f'vol_1m_{sector}' for sector in selected_sectors]
                volatility_values = features.loc[date, volatility_cols]
                
                # Calcul des poids inverses à la volatilité
                inverse_vol = 1.0 / volatility_values
                total = inverse_vol.sum()
                if total > 0:
                    weights = {sector: inverse_vol[f'vol_1m_{sector}'] / total 
                              for sector in selected_sectors}
                else:
                    # En cas de problème, utiliser une pondération égale
                    equal_weight = 1.0 / len(selected_sectors)
                    weights = {sector: equal_weight for sector in selected_sectors}
            
            else:
                # Par défaut, utiliser une pondération égale
                logger.warning(f"Schéma de pondération {weighting_scheme} non reconnu, utilisation de la pondération égale")
                equal_weight = 1.0 / len(selected_sectors)
                weights = {sector: equal_weight for sector in selected_sectors}
            
            logger.debug(f"Poids calculés pour les secteurs à {date}: {weights}")
            return weights
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul des poids: {e}")
            # En cas d'erreur, utiliser une pondération égale
            equal_weight = 1.0 / len(selected_sectors) if selected_sectors else 0
            return {sector: equal_weight for sector in selected_sectors}
    
    def generate_allocation(self, cycle_predictions, features, rebalancing_dates=None):
        """
        Génère l'allocation pour toutes les dates de rebalancement.
        
        Args:
            cycle_predictions (pd.DataFrame): DataFrame contenant les prédictions de cycle.
            features (pd.DataFrame): DataFrame contenant les features.
            rebalancing_dates (list, optional): Liste des dates de rebalancement.
            
        Returns:
            pd.DataFrame: DataFrame contenant l'allocation pour chaque date.
        """
        try:
            # Si les dates de rebalancement ne sont pas spécifiées, utiliser toutes les dates disponibles
            if rebalancing_dates is None:
                # Utiliser les dates communes aux deux DataFrames
                common_dates = sorted(set(cycle_predictions.index) & set(features.index))
                rebalancing_dates = common_dates
            
            # Initialiser le DataFrame pour stocker les allocations
            allocations = pd.DataFrame(index=rebalancing_dates, columns=['cycle_phase', 'selected_sectors', 'weights'])
            
            # Pour chaque date de rebalancement
            for date in rebalancing_dates:
                # Obtenir la phase du cycle économique
                if date in cycle_predictions.index:
                    cycle_phase = cycle_predictions.loc[date, 'cycle_phase']
                    allocations.loc[date, 'cycle_phase'] = cycle_phase
                    
                    # Sélectionner les secteurs recommandés pour cette phase
                    cycle_sectors = self.select_sectors_by_cycle(cycle_phase)
                    
                    # Classer les secteurs par momentum
                    momentum_sectors = self.rank_sectors_by_momentum(features, date)
                    
                    # Combiner les recommandations
                    alpha = 0.7  # Paramètre de pondération cycle vs momentum
                    selected_sectors = self.combine_cycle_and_momentum(cycle_sectors, momentum_sectors, alpha)
                    allocations.loc[date, 'selected_sectors'] = str(selected_sectors)
                    
                    # Calculer les poids
                    if selected_sectors:
                        weights = self.calculate_weights(selected_sectors, features, date)
                        allocations.loc[date, 'weights'] = str(weights)
                    else:
                        allocations.loc[date, 'weights'] = '{}'
            
            # Sauvegarder les allocations
            allocations.to_parquet(Path('data/processed/sector_allocations.parquet'))
            logger.info("Allocations sectorielles générées et sauvegardées")
            
            return allocations
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des allocations: {e}")
            return None
    
    def generate_signals(self, allocations, processed_data):
        """
        Génère les signaux d'achat/vente pour le backtest.
        
        Args:
            allocations (pd.DataFrame): DataFrame contenant les allocations.
            processed_data (pd.DataFrame): DataFrame contenant les données traitées.
            
        Returns:
            pd.DataFrame: DataFrame contenant les signaux.
        """
        try:
            # Extraire les colonnes de rendement des ETFs
            returns_cols = [col for col in processed_data.columns if col.startswith('returns_')]
            sectors = [col.split('_')[1] for col in returns_cols]
            
            # Initialiser le DataFrame pour les signaux (poids de chaque secteur à chaque date)
            signals = pd.DataFrame(index=allocations.index, columns=sectors).fillna(0.0)
            
            # Pour chaque date d'allocation
            for date in allocations.index:
                # Récupérer les secteurs sélectionnés et leurs poids
                selected_sectors_str = allocations.loc[date, 'selected_sectors']
                weights_str = allocations.loc[date, 'weights']
                
                # Convertir les chaînes en listes/dictionnaires
                if selected_sectors_str != '[]' and weights_str != '{}':
                    # Convertir la chaîne des secteurs sélectionnés en liste
                    selected_sectors = eval(selected_sectors_str)
                    
                    # Convertir la chaîne des poids en dictionnaire
                    weights = eval(weights_str)
                    
                    # Mettre à jour les signaux
                    for sector in sectors:
                        if sector in selected_sectors:
                            signals.loc[date, sector] = weights.get(sector, 0.0)
                        else:
                            signals.loc[date, sector] = 0.0
            
            # Sauvegarder les signaux
            signals.to_parquet(Path('data/processed/sector_signals.parquet'))
            logger.info("Signaux générés et sauvegardés")
            
            return signals
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {e}")
            return None
    
    def run_sector_selection(self):
        """
        Exécute le processus complet de sélection sectorielle.
        
        Returns:
            pd.DataFrame: DataFrame contenant les signaux.
        """
        logger.info("Début du processus de sélection sectorielle")
        
        # Charger les données
        cycle_predictions, features, processed_data = self.load_data()
        if cycle_predictions is None or features is None or processed_data is None:
            return None
        
        # Générer les allocations
        allocations = self.generate_allocation(cycle_predictions, features)
        if allocations is None:
            return None
        
        # Générer les signaux
        signals = self.generate_signals(allocations, processed_data)
        
        logger.info("Processus de sélection sectorielle terminé")
        return signals

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer l'instance du modèle
    model = SectorSelectionModel()
    
    # Exécuter le processus de sélection sectorielle
    signals = model.run_sector_selection()
    
    # Afficher un résumé des signaux
    if signals is not None:
        print("\nNombre de dates de rebalancement:", len(signals))
        print("\nDistribution des poids par secteur:")
        print(signals.mean().sort_values(ascending=False))