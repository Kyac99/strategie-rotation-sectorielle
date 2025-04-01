# Stratégie de Rotation Sectorielle Dynamique

## Présentation du Projet
Ce projet développe une stratégie d'investissement basée sur la rotation sectorielle dynamique, permettant d'optimiser l'allocation d'actifs en fonction des cycles économiques et des indicateurs de marché. Cette approche repose sur l'analyse des performances sectorielles et l'identification des opportunités de switch entre secteurs pour maximiser le rendement ajusté au risque.

## Structure du Projet
```
strategie-rotation-sectorielle/
├── data/               # Données brutes et traitées
├── notebooks/          # Notebooks Jupyter pour l'analyse et la visualisation
├── src/                # Code source Python
│   ├── data/           # Scripts de collecte et de traitement des données
│   ├── features/       # Scripts de création de features pour le modèle
│   ├── models/         # Modèles de prédiction et classification
│   ├── backtest/       # Framework de backtesting
│   └── visualization/  # Scripts de visualisation
├── config/             # Fichiers de configuration
├── tests/              # Tests unitaires et d'intégration
├── docs/               # Documentation
└── app/                # Interface utilisateur (Streamlit/Dash)
```

## Objectifs
- Étudier la performance des secteurs dans différents cycles économiques (croissance, récession, inflation, stagflation, etc.)
- Développer un algorithme de switch sectoriel basé sur des indicateurs macroéconomiques et de momentum
- Mettre en place un backtest robuste de la stratégie pour évaluer sa performance et sa robustesse
- Automatiser la mise à jour et le suivi des indicateurs économiques et sectoriels

## Sources de Données
- **Données macroéconomiques**: PIB, taux d'intérêt, inflation, chômage, indices de confiance
- **Données sectorielles**: performances historiques, valorisation, rotation des flux, momentum
- **Indicateurs de marché**: volatilité, spreads de crédit, courbes de taux

## Méthodologie
1. **Identification des cycles économiques** basée sur les données macroéconomiques
2. **Classification des secteurs** adaptés à chaque phase du cycle
3. **Intégration d'un module de momentum** pour affiner les décisions d'allocation
4. **Pondération dynamique** pour ajuster l'exposition aux secteurs sélectionnés

## Fonctionnalités Principales
- Collecte et traitement automatisé des données macroéconomiques et sectorielles
- Identification dynamique des phases du cycle économique
- Modèle de sélection sectorielle basé sur la phase économique et le momentum
- Backtesting complet avec métriques de performance et d'attribution
- Interface utilisateur pour le suivi et l'analyse de la stratégie

## Technologies Utilisées
- **Langages**: Python (Pandas, NumPy, SciPy, Scikit-learn)
- **Base de données**: SQL, MongoDB
- **Infrastructure**: Cloud computing (AWS, Google Cloud)
- **Visualisation**: Matplotlib, Plotly, Dash/Streamlit
- **Déploiement**: Docker, GitHub Actions

## Installation et Utilisation
```bash
# Cloner le repository
git clone https://github.com/Kyac99/strategie-rotation-sectorielle.git
cd strategie-rotation-sectorielle

# Installation des dépendances
pip install -r requirements.txt

# Exécution de l'application
cd app
python main.py
```

## Calendrier de Développement
- **Phase 1** (1 mois): Collecte et traitement des données
- **Phase 2** (2 mois): Développement de l'algorithme et tests préliminaires
- **Phase 3** (2 mois): Backtesting et optimisation
- **Phase 4** (1 mois): Déploiement et automatisation
- **Phase 5** (1 mois): Documentation et finalisation

## Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
