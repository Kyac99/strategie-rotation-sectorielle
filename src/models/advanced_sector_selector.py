"""
Module de sélection sectorielle avancée intégrant des facteurs de cycle économique, momentum,
valorisation et volatilité.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ajout du répertoire parent au path pour l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sector_selector import SectorSelector
from src.utils.common_utils import load_config, ensure_dir

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AdvancedSectorSelector(SectorSelector):
    """
    Classe pour la sélection sectorielle avancée intégrant plusieurs facteurs.
    
    Cette classe étend SectorSelector en ajoutant:
    - Facteurs de valorisation (P/E, P/B, etc.)
    - Facteurs de volatilité et risque
    - Métrique de tendance (MeanReversion, tendance à moyen terme)
    - Pondérations dynamiques des facteurs selon la phase économique
    """

    def __init__(self, cycle_classifier=None, config_path=None):
        """
        Initialise le sélecteur de secteurs avancé.

        Args:
            cycle_classifier: Instance de EconomicCycleClassifier ou chemin vers un modèle sauvegardé.
            config_path (str, optional): Chemin du fichier de configuration.
        """
        # Initialisation de la classe parente
        super().__init__(cycle_classifier)
        
        # Chargement de la configuration
        self.config = load_config(config_path)
        
        # Facteurs de valorisation par secteur (valeurs moyennes historiques)
        self.valuation_factors = {
            'XLY': {'pe': 22, 'pb': 4.2, 'ps': 1.5, 'dividend_yield': 0.012},  # Consommation discrétionnaire
            'XLP': {'pe': 20, 'pb': 3.8, 'ps': 1.4, 'dividend_yield': 0.025},  # Consommation de base
            'XLE': {'pe': 15, 'pb': 1.8, 'ps': 1.1, 'dividend_yield': 0.040},  # Énergie
            'XLF': {'pe': 13, 'pb': 1.3, 'ps': 2.3, 'dividend_yield': 0.022},  # Finance
            'XLV': {'pe': 18, 'pb': 3.5, 'ps': 1.7, 'dividend_yield': 0.016},  # Santé
            'XLI': {'pe': 17, 'pb': 3.0, 'ps': 1.6, 'dividend_yield': 0.018},  # Industrie
            'XLB': {'pe': 16, 'pb': 2.5, 'ps': 1.3, 'dividend_yield': 0.020},  # Matériaux
            'XLK': {'pe': 24, 'pb': 6.0, 'ps': 4.0, 'dividend_yield': 0.010},  # Technologie
            'XLU': {'pe': 17, 'pb': 1.9, 'ps': 2.0, 'dividend_yield': 0.035},  # Services publics
            'XLRE': {'pe': 35, 'pb': 2.2, 'ps': 6.0, 'dividend_yield': 0.033}, # Immobilier
            'XLC': {'pe': 20, 'pb': 3.6, 'ps': 2.0, 'dividend_yield': 0.008}   # Services de communication
        }
        
        # Pondérations des facteurs par phase du cycle économique
        self.factor_weights = {
            'Expansion': {
                'cycle': 0.35,
                'momentum': 0.35,
                'valuation': 0.15,
                'volatility': 0.15
            },
            'Surchauffe': {
                'cycle': 0.30,
                'momentum': 0.25,
                'valuation': 0.30,
                'volatility': 0.15
            },
            'Ralentissement': {
                'cycle': 0.40,
                'momentum': 0.20,
                'valuation': 0.25,
                'volatility': 0.15
            },
            'Récession': {
                'cycle': 0.35,
                'momentum': 0.15,
                'valuation': 0.20,
                'volatility': 0.30
            },
            'Reprise': {
                'cycle': 0.30,
                'momentum': 0.40,
                'valuation': 0.10,
                'volatility': 0.20
            }
        }
        
        # Facteurs de volatilité par phase du cycle
        self.volatility_preferences = {
            'Expansion': 'medium',     # Préférence pour volatilité moyenne
            'Surchauffe': 'low',       # Préférence pour faible volatilité
            'Ralentissement': 'low',   # Préférence pour faible volatilité
            'Récession': 'low',        # Préférence pour faible volatilité
            'Reprise': 'high'          # Préférence pour haute volatilité (opportunités)
        }
        
        # Métrique de tendance à moyen terme (Mean Reversion vs Momentum)
        self.trend_metrics = {
            'lookback_short': 1,       # Période courte (1 mois)
            'lookback_medium': 3,      # Période moyenne (3 mois)
            'lookback_long': 12,       # Période longue (12 mois)
            'mean_reversion_threshold': 0.15  # Seuil pour identifier une opportunité de mean reversion
        }
        
        logger.info("AdvancedSectorSelector initialisé")

    def calculate_valuation_score(self, sector_data, current_date, lookback=6):
        """
        Calcule un score de valorisation pour chaque secteur.
        
        Ce score compare les métriques de valorisation actuelles avec leurs
        moyennes historiques. Un score bas indique une valorisation attractive.

        Args:
            sector_data (pd.DataFrame): Données sectorielles.
            current_date (datetime): Date actuelle.
            lookback (int, optional): Nombre de périodes pour moyenner les données.

        Returns:
            pd.Series: Scores de valorisation par secteur.
        """
        # Liste des secteurs
        sectors = list(self.valuation_factors.keys())
        
        # Initialisation des scores
        valuation_scores = pd.Series(0.5, index=sectors)
        
        # Dans un environnement réel, on récupérerait les données de valorisation actuelles
        # Ici, on simule avec des données historiques et des variations aléatoires
        for sector in sectors:
            if sector in sector_data.columns:
                # Simulation de métriques de valorisation actuelles
                current_pe = self.valuation_factors[sector]['pe'] * (1 + np.random.normal(0, 0.1))
                current_pb = self.valuation_factors[sector]['pb'] * (1 + np.random.normal(0, 0.1))
                current_ps = self.valuation_factors[sector]['ps'] * (1 + np.random.normal(0, 0.1))
                current_dy = self.valuation_factors[sector]['dividend_yield'] * (1 + np.random.normal(0, 0.1))
                
                # Calcul des ratios par rapport aux moyennes historiques
                pe_ratio = self.valuation_factors[sector]['pe'] / current_pe
                pb_ratio = self.valuation_factors[sector]['pb'] / current_pb
                ps_ratio = self.valuation_factors[sector]['ps'] / current_ps
                dy_ratio = current_dy / self.valuation_factors[sector]['dividend_yield']
                
                # Combinaison des ratios en un score (0-1)
                # Plus le score est élevé, plus la valorisation est attractive
                score = (pe_ratio * 0.35 + pb_ratio * 0.25 + ps_ratio * 0.15 + dy_ratio * 0.25)
                
                # Normalisation du score (0-1)
                score = max(0, min(1, score / 2))
                
                valuation_scores[sector] = score
        
        return valuation_scores

    def calculate_volatility_score(self, sector_data, current_date, lookback_window=63):
        """
        Calcule un score de volatilité pour chaque secteur.
        
        En fonction de la phase économique, le score favorise soit une faible
        volatilité (phases de ralentissement/récession) soit une forte volatilité
        (phases de reprise/début d'expansion).

        Args:
            sector_data (pd.DataFrame): Données sectorielles.
            current_date (datetime): Date actuelle.
            lookback_window (int, optional): Fenêtre pour le calcul de la volatilité.

        Returns:
            pd.Series: Scores de volatilité par secteur.
        """
        # Liste des secteurs
        sectors = [col for col in sector_data.columns if '_' not in col]
        
        # Calcul de la volatilité
        volatility = {}
        date_idx = sector_data.index.get_loc(current_date)
        start_idx = max(0, date_idx - lookback_window)
        
        for sector in sectors:
            if sector in sector_data.columns:
                # Calcul de la volatilité sur la fenêtre de lookback
                sector_price = sector_data[sector].iloc[start_idx:date_idx+1]
                sector_returns = sector_price.pct_change().dropna()
                vol = sector_returns.std() * np.sqrt(252)  # Annualisation
                volatility[sector] = vol
        
        # Conversion en Series
        volatility_series = pd.Series(volatility)
        
        # Identification de la phase économique actuelle
        current_phase = self.identify_current_cycle(sector_data)
        preference = self.volatility_preferences.get(current_phase, 'medium')
        
        # Calcul du score selon la préférence de volatilité
        if preference == 'low':
            # Préférence pour faible volatilité: score inversement proportionnel à la volatilité
            volatility_scores = 1 - (volatility_series - volatility_series.min()) / (volatility_series.max() - volatility_series.min())
        elif preference == 'high':
            # Préférence pour forte volatilité: score proportionnel à la volatilité
            volatility_scores = (volatility_series - volatility_series.min()) / (volatility_series.max() - volatility_series.min())
        else:  # 'medium'
            # Préférence pour volatilité moyenne: score maximal au milieu
            normalized_vol = (volatility_series - volatility_series.min()) / (volatility_series.max() - volatility_series.min())
            volatility_scores = 1 - 2 * np.abs(normalized_vol - 0.5)
        
        return volatility_scores

    def calculate_trend_score(self, sector_data, current_date):
        """
        Calcule un score de tendance pour chaque secteur.
        
        Ce score identifie si un secteur est en tendance (momentum) ou en
        situation de retour à la moyenne (mean reversion).

        Args:
            sector_data (pd.DataFrame): Données sectorielles.
            current_date (datetime): Date actuelle.

        Returns:
            pd.Series: Scores de tendance par secteur.
        """
        # Liste des secteurs
        sectors = [col for col in sector_data.columns if '_' not in col]
        
        # Initialisation des scores
        trend_scores = pd.Series(0.5, index=sectors)
        
        # Paramètres
        short = self.trend_metrics['lookback_short']
        medium = self.trend_metrics['lookback_medium']
        long = self.trend_metrics['lookback_long']
        threshold = self.trend_metrics['mean_reversion_threshold']
        
        # Date de l'observation
        date_idx = sector_data.index.get_loc(current_date)
        
        for sector in sectors:
            if sector in sector_data.columns:
                # Extraction des prix
                sector_price = sector_data[sector]
                if date_idx < long:
                    continue
                
                # Calcul des rendements sur différentes périodes
                price_current = sector_price.iloc[date_idx]
                price_short = sector_price.iloc[date_idx - short]
                price_medium = sector_price.iloc[date_idx - medium]
                price_long = sector_price.iloc[date_idx - long]
                
                return_short = price_current / price_short - 1
                return_medium = price_current / price_medium - 1
                return_long = price_current / price_long - 1
                
                # Détection de tendance vs mean reversion
                if return_short > 0 and return_medium > 0 and return_long > 0:
                    # Tendance haussière forte
                    trend_scores[sector] = 0.8
                elif return_short < 0 and return_medium < 0 and return_long < 0:
                    # Tendance baissière forte
                    trend_scores[sector] = 0.2
                elif return_short < 0 and return_medium > 0 and return_long > 0:
                    # Correction à court terme dans une tendance haussière
                    # Potentiel mean reversion vers le haut
                    if abs(return_short) > threshold:
                        trend_scores[sector] = 0.9  # Forte opportunité de mean reversion
                    else:
                        trend_scores[sector] = 0.7  # Opportunité modérée
                elif return_short > 0 and return_medium < 0 and return_long < 0:
                    # Rebond à court terme dans une tendance baissière
                    # Potentiel mean reversion vers le bas
                    if abs(return_short) > threshold:
                        trend_scores[sector] = 0.1  # Forte opportunité de mean reversion (baissière)
                    else:
                        trend_scores[sector] = 0.3  # Opportunité modérée
                else:
                    # Situation mixte
                    trend_scores[sector] = 0.5
        
        return trend_scores

    def select_sectors_advanced(self, macro_data, sector_data, num_sectors=3, include_trend=True):
        """
        Sélectionne les secteurs en utilisant l'approche avancée multi-facteurs.

        Args:
            macro_data (pd.DataFrame): Données macroéconomiques.
            sector_data (pd.DataFrame): Données sectorielles.
            num_sectors (int, optional): Nombre de secteurs à sélectionner.
            include_trend (bool, optional): Si True, inclut la métrique de tendance.

        Returns:
            pd.Series: Pondérations recommandées pour chaque secteur.
        """
        # Obtention de la date actuelle (dernière date disponible)
        current_date = sector_data.index[-1]
        
        # Identification de la phase économique actuelle
        current_phase = self.identify_current_cycle(macro_data)
        logger.info(f"Phase économique actuelle: {current_phase}")
        
        # Obtention des pondérations des facteurs pour la phase actuelle
        weights = self.factor_weights.get(current_phase, {
            'cycle': 0.3,
            'momentum': 0.3,
            'valuation': 0.2,
            'volatility': 0.2
        })
        
        # Calcul des scores par facteur
        # 1. Score de cycle (basé sur la performance historique dans cette phase)
        cycle_scores = pd.Series(self.sector_cycle_performance.get(current_phase, {}))
        cycle_scores = (cycle_scores - cycle_scores.min()) / (cycle_scores.max() - cycle_scores.min())
        
        # 2. Score de momentum
        momentum_scores = self.calculate_momentum_score(sector_data)
        momentum_scores = (momentum_scores - momentum_scores.min()) / (momentum_scores.max() - momentum_scores.min())
        
        # 3. Score de valorisation
        valuation_scores = self.calculate_valuation_score(sector_data, current_date)
        
        # 4. Score de volatilité
        volatility_scores = self.calculate_volatility_score(sector_data, current_date)
        
        # 5. Score de tendance (optionnel)
        if include_trend:
            trend_scores = self.calculate_trend_score(sector_data, current_date)
            
            # Ajustement des pondérations pour inclure la tendance
            weights = {k: v * 0.8 for k, v in weights.items()}
            weights['trend'] = 0.2
        
        # Combinaison des scores avec leurs pondérations
        combined_scores = pd.Series(0, index=cycle_scores.index)
        
        for sector in combined_scores.index:
            score = 0
            if sector in cycle_scores:
                score += cycle_scores[sector] * weights['cycle']
            if sector in momentum_scores:
                score += momentum_scores[sector] * weights['momentum']
            if sector in valuation_scores:
                score += valuation_scores[sector] * weights['valuation']
            if sector in volatility_scores:
                score += volatility_scores[sector] * weights['volatility']
            if include_trend and sector in trend_scores:
                score += trend_scores[sector] * weights['trend']
            
            combined_scores[sector] = score
        
        # Sélection des meilleurs secteurs
        top_sectors = combined_scores.nlargest(num_sectors)
        
        # Normalisation des scores pour obtenir les pondérations
        weights = top_sectors / top_sectors.sum()
        
        # Log des secteurs sélectionnés
        logger.info(f"Secteurs sélectionnés pour la phase {current_phase}:")
        for sector, weight in weights.items():
            logger.info(f"  {sector}: {weight:.2%}")
        
        return weights

    def plot_factor_scores(self, macro_data, sector_data, include_trend=True):
        """
        Visualise les scores des différents facteurs pour tous les secteurs.

        Args:
            macro_data (pd.DataFrame): Données macroéconomiques.
            sector_data (pd.DataFrame): Données sectorielles.
            include_trend (bool, optional): Si True, inclut la métrique de tendance.

        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        # Obtention de la date actuelle
        current_date = sector_data.index[-1]
        
        # Identification de la phase économique actuelle
        current_phase = self.identify_current_cycle(macro_data)
        
        # Calcul des scores par facteur
        cycle_scores = pd.Series(self.sector_cycle_performance.get(current_phase, {}))
        cycle_scores = (cycle_scores - cycle_scores.min()) / (cycle_scores.max() - cycle_scores.min())
        
        momentum_scores = self.calculate_momentum_score(sector_data)
        momentum_scores = (momentum_scores - momentum_scores.min()) / (momentum_scores.max() - momentum_scores.min())
        
        valuation_scores = self.calculate_valuation_score(sector_data, current_date)
        volatility_scores = self.calculate_volatility_score(sector_data, current_date)
        
        if include_trend:
            trend_scores = self.calculate_trend_score(sector_data, current_date)
        
        # Création du DataFrame pour le graphique
        sectors = cycle_scores.index
        
        scores_df = pd.DataFrame(index=sectors)
        scores_df['Cycle'] = cycle_scores
        scores_df['Momentum'] = momentum_scores
        scores_df['Valorisation'] = valuation_scores
        scores_df['Volatilité'] = volatility_scores
        
        if include_trend:
            scores_df['Tendance'] = trend_scores
        
        # Tri par score global
        weights = self.factor_weights.get(current_phase, {
            'cycle': 0.3,
            'momentum': 0.3,
            'valuation': 0.2,
            'volatility': 0.2
        })
        
        if include_trend:
            weights = {k: v * 0.8 for k, v in weights.items()}
            weights['trend'] = 0.2
        
        global_score = (
            scores_df['Cycle'] * weights['cycle'] +
            scores_df['Momentum'] * weights['momentum'] +
            scores_df['Valorisation'] * weights['valuation'] +
            scores_df['Volatilité'] * weights['volatility']
        )
        
        if include_trend:
            global_score += scores_df['Tendance'] * weights['trend']
        
        scores_df['Score Global'] = global_score
        scores_df = scores_df.sort_values('Score Global', ascending=False)
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap des scores
        sns.heatmap(
            scores_df.iloc[:, :-1],  # Exclure la colonne Score Global
            annot=True,
            cmap='YlGnBu',
            linewidths=0.5,
            ax=ax,
            vmin=0,
            vmax=1,
            fmt='.2f'
        )
        
        # Titre et labels
        ax.set_title(f'Scores des facteurs par secteur - Phase: {current_phase}', fontsize=14)
        ax.set_ylabel('Secteur')
        
        # Ajout du score global sur la droite
        ax_right = ax.twinx()
        y_pos = np.arange(len(scores_df))
        ax_right.barh(
            y_pos,
            scores_df['Score Global'],
            height=0.8,
            alpha=0.3,
            color='red'
        )
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_ylabel('Score Global')
        
        # Formatage des étiquettes du score global
        for i, score in enumerate(scores_df['Score Global']):
            ax_right.text(
                score + 0.02,
                i,
                f'{score:.2f}',
                va='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        return fig

    def optimize_weights(self, macro_data, sector_data, target_risk=None, constraints=None):
        """
        Optimise les pondérations des secteurs selon un objectif de rendement/risque.
        
        Utilise une optimisation de portefeuille simplifiée pour obtenir des
        allocations optimales selon la frontière efficiente.

        Args:
            macro_data (pd.DataFrame): Données macroéconomiques.
            sector_data (pd.DataFrame): Données sectorielles.
            target_risk (float, optional): Niveau de risque cible.
            constraints (dict, optional): Contraintes supplémentaires.

        Returns:
            pd.Series: Pondérations optimisées pour chaque secteur.
        """
        # Identification de la phase économique actuelle
        current_phase = self.identify_current_cycle(macro_data)
        
        # Sélection des secteurs prometteurs avec l'approche multi-facteurs
        base_sectors = self.select_sectors_advanced(macro_data, sector_data, num_sectors=5)
        
        # Liste des secteurs sélectionnés
        selected_sectors = base_sectors.index.tolist()
        
        # Extraction des rendements historiques des secteurs sélectionnés
        returns = {}
        for sector in selected_sectors:
            sector_col = f"{sector}_monthly" if f"{sector}_monthly" in sector_data.columns else sector
            if sector_col in sector_data.columns:
                returns[sector] = sector_data[sector_col].dropna()
        
        # Création du DataFrame des rendements
        returns_df = pd.DataFrame(returns)
        
        # Calcul de la matrice de covariance
        cov_matrix = returns_df.cov()
        
        # Rendements espérés par secteur (basés sur la phase économique et les scores)
        expected_returns = pd.Series(index=selected_sectors)
        for sector in selected_sectors:
            # Combinaison du score du cycle et du rendement historique
            cycle_score = self.sector_cycle_performance.get(current_phase, {}).get(sector, 1.0)
            hist_return = returns_df[sector].mean() if sector in returns_df else 0
            expected_returns[sector] = hist_return * cycle_score
        
        # Optimisation simplifiée (allocation égale ajustée par le risque)
        if target_risk is None:
            # Allocation proportionnelle inverse à la volatilité
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol = 1 / volatilities
            optimized_weights = inv_vol / inv_vol.sum()
        else:
            # Ici on pourrait implémenter une optimisation plus complexe
            # comme l'algorithme d'optimisation de Markowitz
            # Pour simplifier, on utilise une allocation égale
            optimized_weights = pd.Series(1 / len(selected_sectors), index=selected_sectors)
        
        # Application de contraintes si spécifiées
        if constraints is not None:
            # Implémentation des contraintes (min/max par secteur, etc.)
            pass
        
        # Normalisation des poids
        optimized_weights = optimized_weights / optimized_weights.sum()
        
        return optimized_weights


if __name__ == "__main__":
    # Exemple d'utilisation
    import os
    
    # Importation du classifieur de cycles
    from src.models.economic_cycle_classifier import EconomicCycleClassifier
    
    # Chemins des données
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed"))
    macro_data_path = os.path.join(data_dir, "macro_data.csv")
    sector_data_path = os.path.join(data_dir, "sector_data.csv")
    
    # Chemin du modèle de classification des cycles
    models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"))
    cycle_model_path = os.path.join(models_dir, "economic_cycle_classifier.joblib")
    
    # Chargement du classifieur de cycles économiques
    if os.path.exists(cycle_model_path):
        classifier = EconomicCycleClassifier.load_model(cycle_model_path)
        print(f"Classifieur de cycles économiques chargé depuis {cycle_model_path}")
    else:
        classifier = None
        print("Modèle de classifieur non trouvé")
    
    # Chargement des données
    if os.path.exists(macro_data_path) and os.path.exists(sector_data_path):
        macro_data = pd.read_csv(macro_data_path, index_col=0, parse_dates=True)
        sector_data = pd.read_csv(sector_data_path, index_col=0, parse_dates=True)
        print(f"Données chargées: {len(macro_data)} observations macroéconomiques, {len(sector_data)} observations sectorielles")
        
        # Création du sélecteur avancé
        selector = AdvancedSectorSelector(cycle_classifier=classifier)
        
        # Sélection des secteurs avec l'approche avancée
        weights = selector.select_sectors_advanced(macro_data, sector_data, num_sectors=3)
        print("\nSecteurs recommandés (approche avancée):")
        for sector, weight in weights.items():
            print(f"  {sector}: {weight:.2%}")
        
        # Visualisation des scores des facteurs
        fig = selector.plot_factor_scores(macro_data, sector_data)
        plt.tight_layout()
        plt.show()
        
        # Optimisation des pondérations
        optimized_weights = selector.optimize_weights(macro_data, sector_data)
        print("\nPondérations optimisées:")
        for sector, weight in optimized_weights.items():
            print(f"  {sector}: {weight:.2%}")
    else:
        print(f"Fichiers de données non trouvés: {macro_data_path}, {sector_data_path}")
