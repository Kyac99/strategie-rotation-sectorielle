# Configuration

Ce répertoire contient les fichiers de configuration pour le projet.

## Fichiers de configuration

- `config.yaml` : Configuration générale du projet
- `data_sources.yaml` : Configuration des sources de données
- `backtest_params.yaml` : Paramètres pour le backtesting
- `model_params.yaml` : Paramètres pour les modèles

## Variables d'environnement

Les informations sensibles comme les clés d'API sont stockées dans un fichier `.env` qui n'est pas versionné. Un exemple de fichier `.env.example` est fourni pour référence.

## Comment utiliser

Les fichiers de configuration sont chargés au démarrage de l'application à l'aide du module `configparser` ou `pyyaml`.

```python
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
```
