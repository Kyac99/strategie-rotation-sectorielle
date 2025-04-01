# Documentation du Projet de Rotation Sectorielle

## Vue d'ensemble

Ce document fournit une documentation complète du projet de rotation sectorielle, expliquant son architecture, ses composants et son utilisation.

## Objectif du Projet

L'objectif principal de ce projet est de développer une stratégie d'investissement basée sur la rotation sectorielle dynamique, qui permet d'optimiser l'allocation d'actifs en fonction des cycles économiques et des indicateurs de marché comme le momentum.

La rotation sectorielle repose sur le principe que différents secteurs du marché surperforment à différentes phases du cycle économique. Par exemple, les secteurs défensifs comme les services publics et la consommation de base ont tendance à surperformer pendant les périodes de récession, tandis que les secteurs cycliques comme la technologie et la consommation discrétionnaire surperforment généralement pendant les périodes d'expansion économique.

## Architecture du Projet

L'architecture du projet suit une structure modulaire organisée comme suit:

```
strategie-rotation-sectorielle/
├── data/               # Données brutes et traitées
├── notebooks/          # Notebooks Jupyter pour l'analyse et la visualisation
├── src/                # Code source Python
│   ├── data/           # Scripts de collecte et de traitement des données
│   ├── features/       # Scripts de création de features pour le modèle
│   ├── models/         # Modèles de prédiction et classification
│   ├── backtest/       # Framework de backtesting
│   ├── utils/          # Fonctions utilitaires
│   └── visualization/  # Scripts de visualisation
├── config/             # Fichiers de configuration
├── tests/              # Tests unitaires et d'intégration
├── docs/               # Documentation
├── models/             # Modèles entraînés sauvegardés
├── results/            # Résultats des backtests et analyses
└── app/                # Interface utilisateur (Streamlit)
```

## Modules Principaux

### 1. Module de Collecte de Données (`src/data/`)

#### 1.1 Collecteur de Données Macroéconomiques (`macro_data_collector.py`)

Ce module est responsable de la collecte des données macroéconomiques via l'API FRED (Federal Reserve Economic Data). Il récupère une variété d'indicateurs économiques, notamment:

- **Indicateurs de croissance**: PIB réel, production industrielle, emploi non-agricole, taux de chômage
- **Indicateurs d'inflation**: IPC, indice des prix PCE, indice des prix à la production
- **Indicateurs de politique monétaire**: taux des fonds fédéraux, spread de taux, anticipations d'inflation
- **Indicateurs de confiance**: confiance des consommateurs, confiance des entreprises
- **Indicateurs de marché**: VIX, spreads de crédit, etc.

Le collecteur prétraite également les données en calculant les variations en glissement annuel et mensuel.

#### 1.2 Collecteur de Données Sectorielles (`sector_data_collector.py`)

Ce module collecte les données historiques des ETFs sectoriels via l'API Yahoo Finance. Les principaux ETFs collectés sont:

- **ETFs sectoriels**: XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU, XLRE, XLC (représentant les 11 secteurs GICS)
- **ETFs de référence**: SPY (S&P 500), QQQ (NASDAQ), etc.
- **ETFs de style**: valeur, croissance, grande/moyenne/petite capitalisation
- **ETFs obligataires et de matières premières**

Le collecteur calcule diverses métriques pour chaque secteur, comme les rendements sur différentes périodes, la force relative par rapport au marché, et la volatilité.

### 2. Module de Modélisation (`src/models/`)

#### 2.1 Classifieur de Cycles Économiques (`economic_cycle_classifier.py`)

Ce module identifie les phases du cycle économique (Expansion, Surchauffe, Ralentissement, Récession, Reprise) à partir des données macroéconomiques. Il implémente deux approches:

- **Approche non supervisée**: utilisation de K-means pour regrouper les données en clusters représentant les différentes phases
- **Approche supervisée**: utilisation d'un RandomForest pour classifier les phases (si des données historiques étiquetées sont disponibles)

#### 2.2 Sélecteur de Secteurs (`sector_selector.py`)

Ce module sélectionne les secteurs à surpondérer en fonction de la phase économique identifiée et du momentum des secteurs. Il combine:

- **Analyse du cycle économique**: utilisation des performances historiques des secteurs dans chaque phase
- **Analyse du momentum**: sélection des secteurs avec le meilleur momentum récent
- **Pondération dynamique**: équilibrage entre les signaux du cycle et du momentum

### 3. Module de Backtesting (`src/backtest/`)

#### 3.1 Moteur de Backtesting (`backtest_engine.py`)

Ce module permet de simuler et d'évaluer les stratégies de rotation sectorielle sur des données historiques. Il offre:

- **Simulation de stratégies**: test de différentes stratégies et paramètres
- **Calcul de métriques de performance**: rendement, volatilité, ratio de Sharpe, drawdown, etc.
- **Optimisation de paramètres**: recherche des paramètres optimaux par grid search
- **Génération de rapports**: création de rapports détaillés sur les performances

### 4. Module de Visualisation (`src/visualization/`)

#### 4.1 Visualiseur de Performances (`performance_visualizer.py`)

Ce module génère des visualisations détaillées des performances de la stratégie:

- **Performance cumulée**: évolution de la valeur du portefeuille
- **Drawdowns**: périodes de baisse et récupération
- **Rendements annuels**: comparaison année par année
- **Allocations sectorielles**: évolution des allocations au fil du temps
- **Métriques glissantes**: ratio de Sharpe, volatilité, etc. sur des fenêtres glissantes

### 5. Module Utilitaire (`src/utils/`)

#### 5.1 Utilitaires Communs (`common_utils.py`)

Ce module fournit des fonctions utilitaires réutilisables à travers le projet:

- **Gestion de configuration**: chargement des paramètres depuis les fichiers YAML
- **Calcul de métriques**: fonctions pour calculer diverses métriques de performance
- **Manipulation de données**: fonctions pour traiter et transformer les données
- **Visualisation**: fonctions auxiliaires pour la création de graphiques

### 6. Application Web (`app/`)

#### 6.1 Application Streamlit (`main.py`)

Ce module implémente une interface utilisateur interactive avec Streamlit pour:

- **Visualisation des données macroéconomiques et sectorielles**
- **Affichage des phases du cycle économique identifiées**
- **Visualisation des performances des secteurs**
- **Recommandations sectorielles basées sur la phase actuelle**
- **Backtesting interactif de la stratégie**
- **Mise à jour des données et du modèle**

## Notebooks d'Analyse

Le projet inclut plusieurs notebooks Jupyter pour l'analyse et la visualisation:

### 1. Analyse Exploratoire des Données Macroéconomiques (`notebooks/01_analyse_exploratoire_donnees_macro.ipynb`)

Ce notebook explore les données macroéconomiques et identifie les cycles économiques:
- Analyse des indicateurs économiques clés
- Identification des phases du cycle par clustering
- Visualisation des cycles économiques
- Validation avec les récessions officielles

### 2. Analyse des Performances Sectorielles (`notebooks/02_analyse_exploratoire_performances_sectorielles.ipynb`)

Ce notebook analyse les performances des différents secteurs:
- Performance historique des secteurs
- Corrélation entre les secteurs
- Performance des secteurs dans chaque phase du cycle
- Stratégies simples de rotation sectorielle

### 3. Backtesting de la Stratégie (`notebooks/03_backtest_strategie_rotation.ipynb`)

Ce notebook teste et optimise la stratégie de rotation sectorielle:
- Backtesting de différentes stratégies
- Optimisation des paramètres
- Analyse de l'impact des coûts de transaction
- Analyse de différentes fréquences de rééquilibrage
- Génération de rapports de performance

## Installation et Utilisation

### Prérequis

- Python 3.8 ou supérieur
- Pip (gestionnaire de paquets Python)

### Installation

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/Kyac99/strategie-rotation-sectorielle.git
   cd strategie-rotation-sectorielle
   ```

2. Installer les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

3. Configurer les clés API:
   - Copier le fichier `.env.example` en `.env`
   - Ajouter votre clé API FRED dans le fichier `.env`

### Utilisation

#### Collecte de données

Pour collecter et traiter les données:

```python
from src.data.macro_data_collector import MacroDataCollector
from src.data.sector_data_collector import SectorDataCollector

# Collecte des données macroéconomiques
macro_collector = MacroDataCollector()
macro_data = macro_collector.get_all_series(start_date="2000-01-01", frequency='m')
processed_macro = macro_collector.preprocess_data(macro_data)

# Collecte des données sectorielles
sector_collector = SectorDataCollector()
etf_data = sector_collector.get_all_etf_data(start_date="2000-01-01")
processed_sectors = sector_collector.preprocess_data(etf_data)
```

#### Identification des cycles économiques

Pour identifier les phases du cycle économique:

```python
from src.models.economic_cycle_classifier import EconomicCycleClassifier

# Création et entraînement du modèle
classifier = EconomicCycleClassifier(supervised=False)
classifier.fit(processed_macro)

# Prédiction des phases
phases = classifier.predict(processed_macro)
```

#### Sélection des secteurs

Pour sélectionner les secteurs à surpondérer:

```python
from src.models.sector_selector import SectorSelector

# Création du sélecteur
selector = SectorSelector(cycle_classifier=classifier)

# Sélection des secteurs
weights = selector.select_sectors(
    macro_data=processed_macro,
    sector_data=processed_sectors,
    num_sectors=3,
    momentum_weight=0.5
)
```

#### Backtesting

Pour tester la stratégie sur des données historiques:

```python
from src.backtest.backtest_engine import BacktestEngine

# Création du moteur de backtesting
backtest = BacktestEngine(
    sector_data=processed_sectors,
    macro_data=processed_macro
)

# Exécution du backtest
results, allocations = backtest.run_simple_strategy(
    strategy_func=cycle_based_strategy,
    strategy_params={
        'cycle_classifier': classifier,
        'top_n': 3,
        'momentum_weight': 0.5
    },
    start_date="2010-01-01",
    end_date="2023-12-31",
    frequency="M"
)

# Calcul des métriques de performance
metrics = backtest.calculate_performance_metrics(results)
```

#### Visualisation

Pour visualiser les performances:

```python
from src.visualization.performance_visualizer import PerformanceVisualizer

# Création du visualiseur
visualizer = PerformanceVisualizer(interactive=True)

# Génération des visualisations
dashboard = visualizer.generate_performance_dashboard(
    results,
    allocations,
    output_dir="results/visualizations",
    format="html"
)
```

#### Lancement de l'application

Pour lancer l'application Streamlit:

```bash
cd app
streamlit run main.py
```

## Configuration

Le fichier de configuration principal (`config/config.yaml`) permet de personnaliser divers aspects du projet:

- **Sources de données**: API keys, séries à collecter
- **Paramètres des modèles**: configuration du classifieur de cycles, sélecteur de secteurs
- **Paramètres de backtesting**: période, capital initial, coûts de transaction
- **Visualisation**: style, format, interactivité
- **Application**: configuration de l'interface Streamlit

## Exemples d'Utilisation

### Exemple 1: Analyse macroéconomique

Consultez le notebook `notebooks/01_analyse_exploratoire_donnees_macro.ipynb` pour une analyse détaillée des indicateurs macroéconomiques et l'identification des cycles économiques.

### Exemple 2: Analyse sectorielle

Consultez le notebook `notebooks/02_analyse_exploratoire_performances_sectorielles.ipynb` pour analyser les performances des différents secteurs et leur comportement dans les différentes phases du cycle économique.

### Exemple 3: Backtesting d'une stratégie de rotation

Consultez le notebook `notebooks/03_backtest_strategie_rotation.ipynb` pour voir comment tester et optimiser une stratégie de rotation sectorielle basée sur les cycles économiques et le momentum.

## Limitations et Améliorations Futures

### Limitations Actuelles

- **Dépendance aux données historiques**: la stratégie suppose que les relations historiques entre cycles économiques et performances sectorielles se maintiendront
- **Délai des indicateurs macroéconomiques**: certains indicateurs sont publiés avec retard, ce qui peut affecter l'identification en temps réel
- **Simplifications**: le modèle simplifie la complexité réelle des marchés et des cycles économiques

### Améliorations Futures

- **Modèles plus avancés**: exploration de modèles d'apprentissage profond pour l'identification des phases
- **Indicateurs avancés**: inclusion d'indicateurs avancés pour anticiper les changements de phase
- **Facteurs supplémentaires**: intégration de facteurs comme la valorisation, les flux d'actifs
- **Optimisation de portefeuille**: techniques d'optimisation pour la pondération des secteurs
- **Trading automatisé**: intégration avec des plateformes de trading algorithmique

## Contributions

Les contributions à ce projet sont les bienvenues. Pour contribuer:

1. Forker le dépôt
2. Créer une branche pour votre fonctionnalité
3. Ajouter vos modifications
4. Soumettre une pull request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contact

Pour toute question ou suggestion, veuillez contacter le mainteneur du projet.
