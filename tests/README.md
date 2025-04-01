# Tests

Ce répertoire contient les tests unitaires et d'intégration pour le projet.

## Structure

- `unit/` : Tests unitaires
- `integration/` : Tests d'intégration
- `data/` : Données de test
- `conftest.py` : Configuration pytest

## Tests unitaires

Les tests unitaires vérifient le bon fonctionnement des composants individuels :
- Tests des fonctions de traitement de données
- Tests des modèles de classification
- Tests des utilitaires

## Tests d'intégration

Les tests d'intégration vérifient l'interaction entre les différents composants :
- Tests du pipeline complet
- Tests de l'API
- Tests de l'interface utilisateur

## Comment exécuter les tests

```bash
# Exécuter tous les tests
pytest

# Exécuter les tests unitaires uniquement
pytest tests/unit

# Exécuter les tests avec coverage
pytest --cov=src tests/
```
