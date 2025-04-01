"""
Tests unitaires pour le classifieur de cycles économiques.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ajout du répertoire racine au path pour l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.economic_cycle_classifier import EconomicCycleClassifier


class TestEconomicCycleClassifier(unittest.TestCase):
    """Tests pour la classe EconomicCycleClassifier."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.classifier_unsupervised = EconomicCycleClassifier(supervised=False)
        self.classifier_supervised = EconomicCycleClassifier(supervised=True)
        
        # Création de données de test
        dates = pd.date_range('2000-01-01', periods=100, freq='M')
        self.test_data = pd.DataFrame({
            'GDPC1_YOY': np.sin(np.linspace(0, 4*np.pi, 100)) * 3 + 2,  # Simule la croissance du PIB
            'INDPRO_YOY': np.sin(np.linspace(0.5, 4.5*np.pi, 100)) * 5 + 1,  # Simule la production industrielle
            'UNRATE': np.sin(np.linspace(np.pi, 5*np.pi, 100)) * 2 + 6,  # Simule le taux de chômage
            'CPIAUCSL_YOY': np.sin(np.linspace(0.8, 4.8*np.pi, 100)) * 1 + 2.5,  # Simule l'inflation
            'FEDFUNDS': np.sin(np.linspace(0.2, 4.2*np.pi, 100)) * 2 + 3,  # Simule le taux directeur
            'T10Y2Y': np.sin(np.linspace(0.4, 4.4*np.pi, 100)) * 0.5 + 1,  # Simule le spread de taux
            'BAMLH0A0HYM2': np.sin(np.linspace(1.2, 5.2*np.pi, 100)) * 2 + 4,  # Simule le spread de crédit
            'UMCSENT': np.sin(np.linspace(0.6, 4.6*np.pi, 100)) * 10 + 80,  # Simule la confiance des consommateurs
            'VIXCLS': np.sin(np.linspace(1.5, 5.5*np.pi, 100)) * 5 + 15  # Simule la volatilité du marché
        }, index=dates)
        
        # Labels pour les tests supervisés
        self.test_labels = pd.Series(
            np.random.choice(['Expansion', 'Surchauffe', 'Ralentissement', 'Récession', 'Reprise'], size=100),
            index=dates
        )

    def test_init(self):
        """Test de l'initialisation."""
        # Vérification des attributs d'initialisation
        self.assertFalse(self.classifier_unsupervised.supervised)
        self.assertTrue(self.classifier_supervised.supervised)
        
        # Vérification des modèles initialisés
        self.assertIsNotNone(self.classifier_unsupervised.model)
        self.assertIsNotNone(self.classifier_supervised.model)
        
        # Vérification du dictionnaire des phases
        self.assertEqual(len(self.classifier_unsupervised.cycle_labels), 5)
        self.assertEqual(len(self.classifier_supervised.cycle_labels), 5)
        
        # Vérification des indicateurs clés
        self.assertTrue(len(self.classifier_unsupervised.key_indicators) > 0)

    def test_select_features(self):
        """Test de la méthode _select_features."""
        # Appel de la méthode
        X = self.classifier_unsupervised._select_features(self.test_data)
        
        # Vérification
        self.assertIsNotNone(X)
        # On s'attend à ce que toutes les colonnes disponibles dans key_indicators soient sélectionnées
        expected_columns = [col for col in self.classifier_unsupervised.key_indicators if col in self.test_data.columns]
        self.assertEqual(list(X.columns), expected_columns)
        
        # Vérification du traitement des valeurs manquantes
        self.assertEqual(X.isna().sum().sum(), 0)

    def test_fit_unsupervised(self):
        """Test de la méthode fit en mode non supervisé."""
        # Appel de la méthode
        result = self.classifier_unsupervised.fit(self.test_data)
        
        # Vérification
        self.assertIsNotNone(result)
        self.assertEqual(result, self.classifier_unsupervised)  # La méthode retourne self
        
        # Vérification que le modèle a bien été entraîné
        self.assertTrue(hasattr(self.classifier_unsupervised.model, 'cluster_centers_'))
        self.assertEqual(len(self.classifier_unsupervised.model.cluster_centers_), 5)  # 5 clusters

    def test_fit_supervised(self):
        """Test de la méthode fit en mode supervisé."""
        # Appel de la méthode
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
                # Configuration des mocks
                X_scaled = self.classifier_supervised.scaler.fit_transform(
                    self.classifier_supervised._select_features(self.test_data)
                )
                mock_split.return_value = (X_scaled[:80], X_scaled[80:], self.test_labels[:80], self.test_labels[80:])
                mock_grid_instance = MagicMock()
                mock_grid.return_value = mock_grid_instance
                mock_grid_instance.fit.return_value = None
                mock_grid_instance.best_estimator_ = self.classifier_supervised.model
                mock_grid_instance.best_estimator_.predict.return_value = self.test_labels[80:].values
                mock_grid_instance.best_estimator_.feature_importances_ = np.ones(len(available_indicators))
                
                # Appel de la méthode
                result = self.classifier_supervised.fit(self.test_data, self.test_labels)
                
                # Vérification
                self.assertIsNotNone(result)
                self.assertEqual(result, self.classifier_supervised)  # La méthode retourne self
                mock_split.assert_called_once()
                mock_grid.assert_called_once()
                mock_grid_instance.fit.assert_called_once()

    def test_predict(self):
        """Test de la méthode predict."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.test_data)
        
        # Appel de la méthode
        phases = self.classifier_unsupervised.predict(self.test_data)
        
        # Vérification
        self.assertIsNotNone(phases)
        self.assertEqual(len(phases), len(self.test_data))
        self.assertTrue(all(phase in self.classifier_unsupervised.cycle_labels.values() for phase in phases))

    def test_save_and_load_model(self):
        """Test des méthodes save_model et load_model."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.test_data)
        
        # Sauvegarde du modèle dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            # Sauvegarde
            self.classifier_unsupervised.save_model(tmp.name)
            
            # Chargement
            loaded_classifier = EconomicCycleClassifier.load_model(tmp.name)
            
            # Vérification
            self.assertIsNotNone(loaded_classifier)
            self.assertEqual(loaded_classifier.supervised, self.classifier_unsupervised.supervised)
            self.assertEqual(loaded_classifier.cycle_labels, self.classifier_unsupervised.cycle_labels)
            self.assertEqual(loaded_classifier.key_indicators, self.classifier_unsupervised.key_indicators)
            
            # Test de prédiction avec le modèle chargé
            phases_original = self.classifier_unsupervised.predict(self.test_data)
            phases_loaded = loaded_classifier.predict(self.test_data)
            
            # Les prédictions devraient être identiques
            pd.testing.assert_series_equal(phases_original, phases_loaded)

    def test_analyze_clusters(self):
        """Test de la méthode _analyze_clusters."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.test_data)
        
        # Obtention des clusters
        X = self.classifier_unsupervised._select_features(self.test_data)
        X_scaled = self.classifier_unsupervised.scaler.transform(X)
        labels = self.classifier_unsupervised.model.predict(X_scaled)
        
        # Appel de la méthode
        cluster_stats = self.classifier_unsupervised._analyze_clusters(X, labels)
        
        # Vérification
        self.assertIsNotNone(cluster_stats)
        self.assertEqual(len(cluster_stats), 5)  # 5 clusters
        self.assertEqual(list(cluster_stats.index), list(range(5)))  # Clusters 0-4
        self.assertEqual(list(cluster_stats.columns), list(X.columns))  # Mêmes colonnes que X

    def test_assign_cluster_labels(self):
        """Test de la méthode _assign_cluster_labels."""
        # Données de test
        cluster_stats = pd.DataFrame({
            'GDPC1_YOY': [3.0, 4.0, 1.0, -1.0, 2.0],
            'CPIAUCSL_YOY': [2.0, 4.0, 3.0, 1.5, 1.0]
        }, index=range(5))
        
        # Appel de la méthode
        self.classifier_unsupervised._assign_cluster_labels(cluster_stats)
        
        # Vérification
        self.assertIsNotNone(self.classifier_unsupervised.cycle_labels)
        self.assertEqual(len(self.classifier_unsupervised.cycle_labels), 5)
        
        # Les valeurs devraient être les noms des phases
        self.assertTrue(all(phase in ['Expansion', 'Surchauffe', 'Ralentissement', 'Récession', 'Reprise']
                          for phase in self.classifier_unsupervised.cycle_labels.values()))

    def test_plot_cycle_distribution(self):
        """Test de la méthode plot_cycle_distribution."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.test_data)
        
        # Appel de la méthode
        fig = self.classifier_unsupervised.plot_cycle_distribution(self.test_data)
        
        # Vérification
        self.assertIsNotNone(fig)
        
        # Fermeture de la figure pour éviter les fuites de mémoire
        from matplotlib import pyplot as plt
        plt.close(fig)

    def test_plot_cycle_characteristics(self):
        """Test de la méthode plot_cycle_characteristics."""
        # Entraînement du modèle
        self.classifier_unsupervised.fit(self.test_data)
        
        # Appel de la méthode
        fig = self.classifier_unsupervised.plot_cycle_characteristics(self.test_data)
        
        # Vérification
        self.assertIsNotNone(fig)
        
        # Fermeture de la figure pour éviter les fuites de mémoire
        from matplotlib import pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
