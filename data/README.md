# Données

Ce répertoire contient les données brutes et traitées utilisées par le projet.

## Structure

- `raw/` : Données brutes téléchargées depuis les sources externes
- `processed/` : Données traitées et prêtes à être utilisées par les modèles
- `external/` : Données externes provenant de tiers

## Sources de données

### Données macroéconomiques
- FRED (Federal Reserve Economic Data)
- Banque Mondiale
- BCE (Banque Centrale Européenne)
- INSEE

### Données sectorielles
- ETF sectoriels (SPDR, iShares, etc.)
- Indices sectoriels (GICS, MSCI, etc.)
- Facteurs de style (momentum, value, growth, etc.)

### Indicateurs de marché
- VIX (volatilité)
- Spreads de crédit
- Courbes de taux

## Format des données

Les données sont principalement stockées aux formats suivants :
- CSV pour les séries temporelles
- Parquet pour les données volumineuses
- JSON pour les métadonnées et configurations

## Mise à jour des données

Les scripts de mise à jour automatique se trouvent dans `src/data/`.
