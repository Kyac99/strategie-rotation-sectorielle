"""
Tests unitaires pour le module de classification des cycles économiques.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ajout du répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.economic_cycle_classifier import EconomicCycleClassifier


class TestEconomicCycleClassifier(unittest.TestCase):
    """
    Classe de tests pour EconomicCycleClassifier.
    """

    def setUp(self):
        """
        Configuration des tests.
        """
        # Création d'un DataFrame de test pour les données macroéconomiques
        index = pd.date_range(start='2020-01-01', periods=24, freq='M')
        
        # Création de données synthétiques qui simulent différentes phases du cycle
        # Expansion: PIB élevé, chômage bas, inflation modérée
        # Surchauffe: PIB élevé, chômage bas, inflation élevée
        # Ralentissement: PIB en baisse, chômage en hausse, inflation élevée
        # Récession: PIB négatif, chômage élevé, inflation en baisse
        # Reprise: PIB en hausse, chômage en baisse, inflation basse
        
        data = {
            'GDPC1_YOY': np.concatenate([
                np.linspace(3.0, 4.0, 6),      # Expansion
                np.linspace(4.0, 3.5, 4),      # Surchauffe
                np.linspace(3.5, 0.5, 4),      # Ralentissement
                np.linspace(0.5, -2.0, 6),     # Récession
                np.linspace(-2.0, 1.0, 4)      # Reprise
            ])[:24],
            'UNRATE': np.concatenate([
                np.linspace(4.0, 3.5, 6),      # Expansion
                np.linspace(3.5, 3.8, 4),      # Surchauffe
                np.linspace(3.8, 5.0, 4),      # Ralentissement
                np.linspace(5.0, 7.0, 6),      # Récession
                np.linspace(7.0, 6.0, 4)       # Reprise
            ])[:24],
            'CPIAUCSL_YOY': np.concatenate([
                np.linspace(2.0, 2.5, 6),      # Expansion
                np.linspace(2.5, 4.0, 4),      # Surchauffe
                np.linspace(4.0, 3.5, 4),      # Ralentissement
                np.linspace(3.5, 1.5, 6),      # Récession
                np.linspace(1.5, 1.0, 4)       # Reprise
            ])[:24],
            'FEDFUNDS': np.concatenate([
                np.linspace(1.5, 2.0, 6),      # Expansion
                np.linspace(2.0, 3.0, 4),      # Surchauffe
                np.linspace(3.0, 3.5, 4),      # Ralentissement
                np.linspace(3.5, 1.0, 6),      # Récession
                np.linspace(1.0, 0.5, 4)       # Reprise
            ])[:24]
        }
        
        self.test_data = pd.DataFrame(data, index=index)
        
        # Création du classifieur
        self.classifier = EconomicCycleClassifier(supervised=False)

    def test_init(self):
        """
        Test de l'initialisation du classifieur.
        """
        # Test du mode non supervisé
        classifier = EconomicCycleClassifier(supervised=False)
        self.assertFalse(classifier.supervised)
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.cycle_labels)
        self.assertIsNotNone(classifier.key_indicators)
        
        # Test du mode supervisé
        classifier = EconomicCycleClassifier(supervised=True)
        self.assertTrue(classifier.supervised)

    def test_select_features(self):
        """
        Test de la sélection des features.
        """
        features = self.classifier._select_features(self.test_data)
        
        # Vérification que toutes les colonnes disponibles ont été sélectionnées
        for indicator in self.classifier.key_indicators:
            if indicator in self.test_data.columns:
                self.assertTrue(indicator in features.columns)

    def test_fit_unsupervised(self):
        """
        Test de l'entraînement en mode non supervisé.
        """
        # Entraînement du modèle
        self.classifier.fit(self.test_data)
        
        # Vérification que le modèle a été entraîné
        self.assertIsNotNone(self.classifier.model.cluster_centers_)
        self.assertEqual(self.classifier.model.n_clusters, 5)  # 5 phases par défaut

    @patch('src.models.economic_cycle_classifier.KMeans')
    def test_fit_with_mocked_kmeans(self, mock_kmeans):
        """
        Test de l'entraînement avec KMeans mocké.
        """
        # Configuration du mock
        mock_instance = MagicMock()
        mock_kmeans.return_value = mock_instance
        mock_instance.predict.return_value = np.array([0, 1, 2, 3, 4] * 4 + [0, 1, 2, 3])
        
        # Entraînement du modèle avec le mock
        classifier = EconomicCycleClassifier(supervised=False)
        classifier.model = mock_instance
        classifier.fit(self.test_data)
        
        # Vérification que fit a été appelé
        mock_instance.fit.assert_called_once()

    def test_predict(self):
        """
        Test de la prédiction des phases du cycle économique.
        """
        # Entraînement préalable du modèle
        self.classifier.fit(self.test_data)
        
        # Prédiction sur les mêmes données
        phases = self.classifier.predict(self.test_data)
        
        # Vérification du format des résultats
        self.assertIsInstance(phases, pd.Series)
        self.assertEqual(len(phases), len(self.test_data))
        
        # Vérification que toutes les phases prédites sont dans le dictionnaire de labels
        for phase in phases.values:
            self.assertTrue(phase in self.classifier.cycle_labels.values())

    def test_save_load_model(self):
        """
        Test de la sauvegarde et du chargement du modèle.
        """
        # Entraînement préalable du modèle
        self.classifier.fit(self.test_data)
        
        # Sauvegarde du modèle dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            self.classifier.save_model(model_path)
            
            # Vérification que le fichier a été créé
            self.assertTrue(os.path.exists(model_path))
            
            # Chargement du modèle
            loaded_classifier = EconomicCycleClassifier.load_model(model_path)
            
            # Vérification que le modèle chargé a les mêmes attributs
            self.assertEqual(loaded_classifier.supervised, self.classifier.supervised)
            self.assertEqual(loaded_classifier.cycle_labels, self.classifier.cycle_labels)
            self.assertEqual(loaded_classifier.key_indicators, self.classifier.key_indicators)
            
            # Prédiction avec le modèle chargé pour vérifier qu'il fonctionne
            loaded_phases = loaded_classifier.predict(self.test_data)
            original_phases = self.classifier.predict(self.test_data)
            
            # Les prédictions devraient être identiques
            pd.testing.assert_series_equal(loaded_phases, original_phases)
            
        finally:
            # Nettoyage
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_plot_cycle_distribution(self):
        """
        Test de la génération du graphique de distribution des cycles.
        """
        # Entraînement préalable du modèle
        self.classifier.fit(self.test_data)
        
        # Génération du graphique
        fig = self.classifier.plot_cycle_distribution(self.test_data)
        
        # Vérification que le graphique a été créé
        self.assertIsNotNone(fig)

    def test_plot_cycle_characteristics(self):
        """
        Test de la génération du graphique des caractéristiques des cycles.
        """
        # Entraînement préalable du modèle
        self.classifier.fit(self.test_data)
        
        # Génération du graphique
        fig = self.classifier.plot_cycle_characteristics(self.test_data)
        
        # Vérification que le graphique a été créé
        self.assertIsNotNone(fig)

    def test_assign_cluster_labels(self):
        """
        Test de l'assignation des labels aux clusters.
        """
        # Création d'un DataFrame de test pour les statistiques de cluster
        cluster_stats = pd.DataFrame({
            'GDPC1_YOY': [3.5, -1.0, 0.5, 2.0, -0.5],
            'CPIAUCSL_YOY': [2.0, 1.0, 3.5, 4.0, 1.5]
        })
        
        # Test de l'assignation
        self.classifier._assign_cluster_labels(cluster_stats)
        
        # Vérification que tous les clusters ont été assignés
        for i in range(5):
            self.assertTrue(i in self.classifier.cycle_labels)


if __name__ == '__main__':
    unittest.main()
