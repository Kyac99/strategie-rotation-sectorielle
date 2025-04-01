"""
Module d'identification des cycles économiques.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class EconomicCycleClassifier:
    """
    Classe pour identifier les phases du cycle économique.
    
    Cette classe implémente deux approches pour identifier les phases du cycle économique:
    1. Approche non supervisée: utilisation de K-means pour regrouper les données en clusters
    2. Approche supervisée: utilisation d'un RandomForest pour classifier les phases
    
    Les phases du cycle économique sont généralement:
    - Expansion (croissance économique forte, inflation modérée)
    - Surchauffe (croissance forte, inflation élevée)
    - Ralentissement (croissance faible, inflation élevée)
    - Récession (croissance négative, inflation en baisse)
    - Reprise (croissance en hausse, inflation faible)
    """

    def __init__(self, supervised=False):
        """
        Initialise le classifieur de cycles économiques.

        Args:
            supervised (bool): Si True, utilise un modèle supervisé (RandomForest).
                              Si False, utilise un modèle non supervisé (K-means).
        """
        self.supervised = supervised
        self.scaler = StandardScaler()
        
        # Initialisation des modèles
        if supervised:
            # Modèle supervisé (Random Forest)
            self.model = RandomForestClassifier(random_state=42)
        else:
            # Modèle non supervisé (K-means avec 5 clusters)
            self.model = KMeans(n_clusters=5, random_state=42, n_init=10)
        
        # Labels des phases du cycle économique
        self.cycle_labels = {
            0: 'Expansion',
            1: 'Surchauffe',
            2: 'Ralentissement',
            3: 'Récession',
            4: 'Reprise'
        }
        
        # Indicateurs économiques importants pour l'identification des cycles
        self.key_indicators = [
            'GDPC1_YOY',         # Croissance du PIB (annuelle)
            'INDPRO_YOY',        # Croissance de la production industrielle (annuelle)
            'UNRATE',            # Taux de chômage
            'UNRATE_YOY',        # Variation du taux de chômage (annuelle)
            'CPIAUCSL_YOY',      # Inflation (annuelle)
            'FEDFUNDS',          # Taux d'intérêt directeur
            'T10Y2Y',            # Spread de taux 10 ans - 2 ans
            'BAMLH0A0HYM2',      # Spread de crédit à haut rendement
            'UMCSENT',           # Confiance des consommateurs
            'VIXCLS'             # Volatilité du marché
        ]
        
        logger.info(f"EconomicCycleClassifier initialisé en mode {'supervisé' if supervised else 'non supervisé'}")
    
    def _select_features(self, data):
        """
        Sélectionne les indicateurs clés pour l'identification des cycles.

        Args:
            data (pd.DataFrame): Données macroéconomiques.

        Returns:
            pd.DataFrame: DataFrame contenant uniquement les indicateurs sélectionnés.
        """
        # Vérification des colonnes disponibles
        available_columns = []
        for indicator in self.key_indicators:
            if indicator in data.columns:
                available_columns.append(indicator)
            else:
                logger.warning(f"L'indicateur {indicator} n'est pas disponible dans les données")
        
        if not available_columns:
            raise ValueError("Aucun des indicateurs clés n'est disponible dans les données")
        
        # Sélection des colonnes disponibles
        X = data[available_columns].copy()
        
        # Gestion des valeurs manquantes
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        return X
    
    def fit(self, data, labels=None):
        """
        Entraîne le modèle d'identification des cycles économiques.

        Args:
            data (pd.DataFrame): Données macroéconomiques.
            labels (pd.Series, optional): Labels des phases pour l'entraînement supervisé.

        Returns:
            self: Instance entraînée.
        """
        # Sélection des features
        X = self._select_features(data)
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement du modèle
        if self.supervised:
            if labels is None:
                raise ValueError("Les labels sont nécessaires pour l'entraînement supervisé")
            
            logger.info("Entraînement du modèle RandomForest")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, labels, test_size=0.2, random_state=42
            )
            
            # Recherche des hyperparamètres optimaux
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Utilisation des meilleurs paramètres
            self.model = grid_search.best_estimator_
            
            # Évaluation sur le test set
            y_pred = self.model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            logger.info(f"Précision sur le test set: {accuracy:.2f}")
            
            # Importance des features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            })
            logger.info("Importance des features:")
            for idx, row in feature_importance.sort_values('importance', ascending=False).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        else:
            logger.info("Entraînement du modèle K-means")
            
            # Recherche du nombre optimal de clusters (si nécessaire)
            inertia = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            
            # Entraînement du modèle final
            self.model.fit(X_scaled)
            
            # Obtention des clusters
            labels = self.model.predict(X_scaled)
            
            # Analyse des clusters
            cluster_stats = self._analyze_clusters(X, labels)
            
            # Assignation manuelle des labels aux clusters basée sur les statistiques
            self._assign_cluster_labels(cluster_stats)
        
        return self
    
    def _analyze_clusters(self, X, labels):
        """
        Analyse les clusters pour comprendre leurs caractéristiques.

        Args:
            X (pd.DataFrame): Données macroéconomiques.
            labels (np.array): Labels des clusters.

        Returns:
            pd.DataFrame: Statistiques des clusters.
        """
        # Ajout des labels au DataFrame
        X_with_labels = X.copy()
        X_with_labels['cluster'] = labels
        
        # Calcul des moyennes par cluster
        cluster_means = X_with_labels.groupby('cluster').mean()
        
        logger.info("Caractéristiques des clusters:")
        logger.info(cluster_means)
        
        return cluster_means
    
    def _assign_cluster_labels(self, cluster_stats):
        """
        Assigne les labels des phases économiques aux clusters.

        Args:
            cluster_stats (pd.DataFrame): Statistiques des clusters.
        """
        # Logique pour assigner les phases économiques aux clusters
        # Cette méthode est simplifiée et devrait être adaptée selon les données réelles
        
        # Exemple d'assignation basée sur la croissance du PIB et l'inflation
        if 'GDPC1_YOY' in cluster_stats.columns and 'CPIAUCSL_YOY' in cluster_stats.columns:
            # Expansion: Croissance PIB élevée, inflation modérée
            # Surchauffe: Croissance PIB élevée, inflation élevée
            # Ralentissement: Croissance PIB faible, inflation élevée
            # Récession: Croissance PIB négative, inflation en baisse
            # Reprise: Croissance PIB en hausse, inflation faible
            
            # Normalisation pour faciliter la comparaison
            normalized_stats = cluster_stats.copy()
            for col in ['GDPC1_YOY', 'CPIAUCSL_YOY']:
                normalized_stats[col] = (cluster_stats[col] - cluster_stats[col].min()) / (cluster_stats[col].max() - cluster_stats[col].min())
            
            # Assignation des labels
            cluster_phases = {}
            
            for cluster in normalized_stats.index:
                gdp_growth = normalized_stats.loc[cluster, 'GDPC1_YOY']
                inflation = normalized_stats.loc[cluster, 'CPIAUCSL_YOY']
                
                if gdp_growth < 0.2:  # Croissance faible ou négative
                    if inflation < 0.3:
                        cluster_phases[cluster] = 'Récession'
                    else:
                        cluster_phases[cluster] = 'Ralentissement'
                elif gdp_growth > 0.8:  # Croissance très élevée
                    if inflation > 0.7:
                        cluster_phases[cluster] = 'Surchauffe'
                    else:
                        cluster_phases[cluster] = 'Expansion'
                else:  # Croissance modérée
                    if inflation < 0.4:
                        cluster_phases[cluster] = 'Reprise'
                    else:
                        cluster_phases[cluster] = 'Expansion'
            
            # Mise à jour du dictionnaire de mapping
            for cluster, phase in cluster_phases.items():
                for idx, label in enumerate(self.cycle_labels.values()):
                    if label == phase:
                        self.cycle_labels[cluster] = phase
                        break
            
            logger.info("Assignation des phases économiques aux clusters:")
            for cluster, phase in self.cycle_labels.items():
                logger.info(f"  Cluster {cluster}: {phase}")
        else:
            logger.warning("Impossible d'assigner les phases: colonnes nécessaires non disponibles")
    
    def predict(self, data):
        """
        Prédit les phases du cycle économique pour les données fournies.

        Args:
            data (pd.DataFrame): Données macroéconomiques.

        Returns:
            pd.Series: Series contenant les phases prédites.
        """
        # Sélection des features
        X = self._select_features(data)
        
        # Normalisation des données
        X_scaled = self.scaler.transform(X)
        
        # Prédiction
        y_pred = self.model.predict(X_scaled)
        
        # Conversion des clusters en labels de phases
        phase_labels = pd.Series([self.cycle_labels[cluster] for cluster in y_pred], index=data.index)
        
        return phase_labels
    
    def save_model(self, file_path):
        """
        Sauvegarde le modèle entraîné.

        Args:
            file_path (str): Chemin du fichier de sauvegarde.

        Returns:
            bool: True si la sauvegarde a réussi, False sinon.
        """
        try:
            # Création du répertoire parent si nécessaire
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Sauvegarde du modèle
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'supervised': self.supervised,
                'cycle_labels': self.cycle_labels,
                'key_indicators': self.key_indicators
            }, file_path)
            
            logger.info(f"Modèle sauvegardé avec succès dans {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False
    
    @classmethod
    def load_model(cls, file_path):
        """
        Charge un modèle entraîné.

        Args:
            file_path (str): Chemin du fichier de sauvegarde.

        Returns:
            EconomicCycleClassifier: Instance avec le modèle chargé.
        """
        try:
            # Chargement du modèle
            data = joblib.load(file_path)
            
            # Création d'une nouvelle instance
            instance = cls(supervised=data['supervised'])
            
            # Chargement des attributs
            instance.model = data['model']
            instance.scaler = data['scaler']
            instance.cycle_labels = data['cycle_labels']
            instance.key_indicators = data['key_indicators']
            
            logger.info(f"Modèle chargé avec succès depuis {file_path}")
            return instance
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return None
    
    def plot_cycle_distribution(self, data):
        """
        Affiche la distribution des phases du cycle économique au fil du temps.

        Args:
            data (pd.DataFrame): Données macroéconomiques.

        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        # Prédiction des phases
        phases = self.predict(data)
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Conversion en numérique pour colormap
        phase_numeric = phases.map({phase: i for i, phase in enumerate(set(phases))})
        
        # Création du graphique
        scatter = ax.scatter(
            phases.index, 
            np.ones(len(phases)),
            c=phase_numeric, 
            cmap='viridis', 
            s=100, 
            alpha=0.7
        )
        
        # Légende
        unique_phases = list(set(phases))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=plt.cm.viridis(i / len(unique_phases)), 
                           markersize=10, label=phase)
                           for i, phase in enumerate(unique_phases)]
        ax.legend(handles=legend_elements, title="Phases économiques")
        
        # Formatage de l'axe y
        ax.set_yticks([])
        
        # Titre et labels
        ax.set_title('Distribution des phases du cycle économique', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        
        # Grille
        ax.grid(True, alpha=0.3)
        
        # Rotation des dates
        fig.autofmt_xdate()
        
        return fig
    
    def plot_cycle_characteristics(self, data):
        """
        Affiche les caractéristiques des différentes phases du cycle économique.

        Args:
            data (pd.DataFrame): Données macroéconomiques.

        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        # Prédiction des phases
        phases = self.predict(data)
        
        # Sélection des features
        X = self._select_features(data)
        
        # Ajout de la colonne de phase
        X_with_phases = X.copy()
        X_with_phases['Phase'] = phases
        
        # Calcul des moyennes par phase
        phase_means = X_with_phases.groupby('Phase').mean()
        
        # Normalisation des valeurs pour faciliter la visualisation
        normalized_means = phase_means.copy()
        for col in normalized_means.columns:
            normalized_means[col] = (phase_means[col] - phase_means[col].min()) / (phase_means[col].max() - phase_means[col].min())
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Heatmap des caractéristiques
        sns.heatmap(
            normalized_means, 
            annot=True, 
            cmap='YlGnBu', 
            linewidths=.5, 
            ax=ax,
            fmt='.2f'
        )
        
        # Titre
        ax.set_title('Caractéristiques des phases du cycle économique (normalisées)', fontsize=14)
        
        return fig


if __name__ == "__main__":
    # Exemple d'utilisation
    from datetime import datetime
    import os
    import sys
    
    # Ajout du répertoire parent au path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Import du collecteur de données
    from src.data.macro_data_collector import MacroDataCollector
    
    # Chemin des données prétraitées
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "processed", "macro_data.csv"
    )
    
    # Vérification si les données existent, sinon les collecter
    if not os.path.exists(data_path):
        # Collecte des données
        collector = MacroDataCollector()
        macro_data = collector.get_all_series(start_date="2000-01-01", frequency='m')
        processed_data = collector.preprocess_data(macro_data)
        
        # Sauvegarde des données
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        processed_data.to_csv(data_path)
    else:
        # Chargement des données
        processed_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Création et entraînement du modèle
    classifier = EconomicCycleClassifier(supervised=False)
    classifier.fit(processed_data)
    
    # Prédiction des phases
    phases = classifier.predict(processed_data)
    
    # Affichage de la distribution des phases
    fig = classifier.plot_cycle_distribution(processed_data)
    
    # Sauvegarde du modèle
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models", "economic_cycle_classifier.joblib"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)
