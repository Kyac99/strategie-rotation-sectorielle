"""
Tests unitaires pour le module d'identification des cycles économiques.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
import joblib

# Ajout du répertoire racine au path pour l'importation des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.economic_cycle_classifier import EconomicCycleClassifier


class TestEconomicCycleClassifier(unittest.TestCase):
    """Tests pour la classe EconomicCycleClassifier."""
    
    def setUp(self):
        """Configuration initiale pour les tests."""
        # Création de données simulées pour les tests
        dates = pd.date_range(start='2010-01-01', end='2020-12-31', freq='M')
        n_samples = len(dates)
        
        # Simulation de 5 phases économiques sur la période
        # Phase 1: Expansion (2010-2011)
        # Phase 2: Surchauffe (2012-2013)
        # Phase 3: Ralentissement (2014-2015)
        # Phase 4: Récession (2016-2017)
        # Phase 5: Reprise (2018-2020)
        
        # Création des données macroéconomiques simulées
        data = {
            # PIB: croissance forte en expansion et surchauffe, négative en récession
            'GDPC1_YOY': np.concatenate([
                np.random.uniform(3, 5, 24),    # Expansion: 3-5%
                np.random.uniform(4, 6, 24),    # Surchauffe: 4-6%
                np.random.uniform(0, 2, 24),    # Ralentissement: 0-2%
                np.random.uniform(-3, 0, 24),   # Récession: -3-0%
                np.random.uniform(1, 3, 36)     # Reprise: 1-3%
            ]),
            
            # Production industrielle: similaire au PIB mais plus volatile
            'INDPRO_YOY': np.concatenate([
                np.random.uniform(4, 7, 24),    # Expansion: 4-7%
                np.random.uniform(5, 8, 24),    # Surchauffe: 5-8%
                np.random.uniform(-1, 3, 24),   # Ralentissement: -1-3%
                np.random.uniform(-5, -1, 24),  # Récession: -5--1%
                np.random.uniform(2, 5, 36)     # Reprise: 2-5%
            ]),
            
            # Chômage: faible en expansion, élevé en récession
            'UNRATE': np.concatenate([
                np.random.uniform(4, 5, 24),    # Expansion: 4-5%
                np.random.uniform(3.5, 4.5, 24),# Surchauffe: 3.5-4.5%
                np.random.uniform(5, 6, 24),    # Ralentissement: 5-6%
                np.random.uniform(7, 9, 24),    # Récession: 7-9%
                np.random.uniform(5.5, 7, 36)   # Reprise: 5.5-7%
            ]),
            
            # Inflation: modérée en expansion, élevée en surchauffe, faible en récession
            'CPIAUCSL_YOY': np.concatenate([
                np.random.uniform(1.5, 2.5, 24),  # Expansion: 1.5-2.5%
                np.random.uniform(3, 5, 24),      # Surchauffe: 3-5%
                np.random.uniform(2, 4, 24),      # Ralentissement: 2-4%
                np.random.uniform(0, 1.5, 24),    # Récession: 0-1.5%
                np.random.uniform(1, 2, 36)       # Reprise: 1-2%
            ]),
            
            # Taux d'intérêt: bas en récession, élevés en surchauffe
            'FEDFUNDS': np.concatenate([
                np.random.uniform(2, 3, 24),    # Expansion: 2-3%
                np.random.uniform(3, 4.5, 24),  # Surchauffe: 3-4.5%
                np.random.uniform(2, 3, 24),    # Ralentissement: 2-3%
                np.random.uniform(0.5, 1.5, 24),# Récession: 0.5-1.5%
                np.random.uniform(1, 2, 36)     # Reprise: 1-2%
            ]),
            
            # Spread de taux: faible ou négatif avant récession
            'T10Y2Y': np.concatenate([
                np.random.uniform(1, 2, 24),     # Expansion: 1-2%
                np.random.uniform(0, 1, 24),     # Surchauffe: 0-1%
                np.random.uniform(-0.5, 0.5, 24),# Ralentissement: -0.5-0.5%
                np.random.uniform(-1, 0, 24),    # Récession: -1-0%
                np.random.uniform(0.5, 1.5, 36)  # Reprise: 0.5-1.5%
            ]),
            
            # Volatilité du marché: faible en expansion, élevée en récession
            'VIXCLS': np.concatenate([
                np.random.uniform(12, 18, 24),   # Expansion: 12-18
                np.random.uniform(15, 20, 24),   # Surchauffe: 15-20
                np.random.uniform(20, 25, 24),   # Ralentissement: 20-25
                np.random.uniform(25, 35, 24),   # Récession: 25-35
                np.random.uniform(15, 22, 36)    # Reprise: 15-22
            ])
        }
        
        # Création du DataFrame
        self.mock_data = pd.DataFrame(data, index=dates[:n_samples])
        
        # Création de l'instance du classifieur
        self.classifier_unsupervised = EconomicCycleClassifier(supervised=False)
        self.classifier_supervised = EconomicCycleClassifier(supervised=True)
    
    def test_initialization(self):
        """Teste l'initialisation du classifieur."""
        # Vérification des attributs de base
        self.assertFalse(self.classifier_unsupervised.supervised)
        self.assertTrue(self.classifier_supervised.supervised)
        
        # Vérification des modèles initialisés
        self.assertIsNotNone(self.classifier_unsupervised.model)
        self.assertIsNotNone(self.classifier_supervised.model)
        
        # Vérification des labels des phases
        self.assertEqual(len(self.classifier_unsupervised.cycle_labels), 5)
        self.assertIn('Expansion', self.classifier_unsupervised.cycle_labels.values())
        self.assertIn('Récession', self.classifier_unsupervised.cycle_labels.values())
    
    def test_select_features(self):
        """Teste la sélection des features."""
        # Appel de la méthode _select_features
        features = self.classifier_unsupervised._select_features(self.mock_data)
        
        # Vérification du type et de la taille
        self.assertIsInstance(features, pd.DataFrame)
        self.assertLessEqual(features.shape[1], len(self.classifier_unsupervised.key_indicators))
        
        # Vérification que les features sont parmi les indicateurs clés
        for col in features.columns:
            self.assertIn(col, self.classifier_unsupervised.key_indicators)
    
    def test_fit_unsupervised(self):
        """Teste l'entraînement du modèle non supervisé."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Vérification que le modèle a été entraîné
        self.assertTrue(hasattr(self.classifier_unsupervised.model, 'cluster_centers_'))
        self.assertEqual(self.classifier_unsupervised.model.n_clusters, 5)
    
    @patch('sklearn.model_selection.train_test_split')
    @patch('sklearn.model_selection.GridSearchCV')
    def test_fit_supervised(self, mock_gridsearch, mock_split):
        """Teste l'entraînement du modèle supervisé."""
        # Création de labels simulés
        labels = pd.Series(['Expansion'] * 24 + ['Surchauffe'] * 24 + 
                           ['Ralentissement'] * 24 + ['Récession'] * 24 + 
                           ['Reprise'] * 36, index=self.mock_data.index)
        
        # Configuration des mocks
        mock_split.return_value = (None, None, None, None)  # X_train, X_test, y_train, y_test
        mock_gridsearch.return_value.best_estimator_ = self.classifier_supervised.model
        mock_gridsearch.return_value.predict.return_value = labels.values[:20]  # Prédictions pour le test set
        
        # Entraînement du modèle supervisé
        self.classifier_supervised.fit(self.mock_data, labels)
        
        # Vérification que GridSearchCV a été appelé
        mock_gridsearch.assert_called_once()
        
        # Vérification que le modèle a été mis à jour
        self.assertEqual(self.classifier_supervised.model, mock_gridsearch.return_value.best_estimator_)
    
    def test_predict(self):
        """Teste la prédiction des phases économiques."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Prédiction sur les mêmes données
        phases = self.classifier_unsupervised.predict(self.mock_data)
        
        # Vérification du type et de la taille
        self.assertIsInstance(phases, pd.Series)
        self.assertEqual(len(phases), len(self.mock_data))
        
        # Vérification que toutes les phases sont parmi les labels connus
        for phase in phases.unique():
            self.assertIn(phase, self.classifier_unsupervised.cycle_labels.values())
    
    def test_save_load_model(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Sauvegarde dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            temp_file = tmp.name
        
        self.classifier_unsupervised.save_model(temp_file)
        self.assertTrue(os.path.exists(temp_file))
        
        # Chargement du modèle
        loaded_classifier = EconomicCycleClassifier.load_model(temp_file)
        
        # Vérification que le modèle chargé a les mêmes attributs
        self.assertEqual(loaded_classifier.supervised, self.classifier_unsupervised.supervised)
        self.assertEqual(loaded_classifier.cycle_labels, self.classifier_unsupervised.cycle_labels)
        self.assertEqual(loaded_classifier.key_indicators, self.classifier_unsupervised.key_indicators)
        
        # Prédiction avec le modèle chargé
        original_pred = self.classifier_unsupervised.predict(self.mock_data)
        loaded_pred = loaded_classifier.predict(self.mock_data)
        
        # Vérification que les prédictions sont identiques
        pd.testing.assert_series_equal(original_pred, loaded_pred)
        
        # Nettoyage
        os.remove(temp_file)
    
    def test_plot_cycle_distribution(self):
        """Teste la génération du graphique de distribution des cycles."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Génération du graphique
        fig = self.classifier_unsupervised.plot_cycle_distribution(self.mock_data)
        
        # Vérification que la figure a été créée
        self.assertIsInstance(fig, plt.Figure)
        
        # Fermeture de la figure pour éviter les warnings
        plt.close(fig)
    
    def test_plot_cycle_characteristics(self):
        """Teste la génération du graphique des caractéristiques des cycles."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Génération du graphique
        fig = self.classifier_unsupervised.plot_cycle_characteristics(self.mock_data)
        
        # Vérification que la figure a été créée
        self.assertIsInstance(fig, plt.Figure)
        
        # Fermeture de la figure pour éviter les warnings
        plt.close(fig)
    
    def test_analyze_clusters(self):
        """Teste l'analyse des clusters."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.mock_data)
        
        # Sélection des features
        X = self.classifier_unsupervised._select_features(self.mock_data)
        
        # Prédiction des clusters
        clusters = self.classifier_unsupervised.model.predict(
            self.classifier_unsupervised.scaler.transform(X)
        )
        
        # Analyse des clusters
        cluster_stats = self.classifier_unsupervised._analyze_clusters(X, clusters)
        
        # Vérification du type et de la taille
        self.assertIsInstance(cluster_stats, pd.DataFrame)
        self.assertEqual(len(cluster_stats), self.classifier_unsupervised.model.n_clusters)
        
        # Vérification que chaque cluster a des statistiques
        for cluster in range(self.classifier_unsupervised.model.n_clusters):
            self.assertIn(cluster, cluster_stats.index)


if __name__ == '__main__':
    unittest.main()
