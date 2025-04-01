# Application

Ce répertoire contient l'interface utilisateur de la stratégie de rotation sectorielle dynamique.

## Structure

- `app.py` : Point d'entrée principal de l'application
- `pages/` : Pages individuelles de l'application
- `components/` : Composants réutilisables
- `assets/` : Ressources statiques (CSS, images, etc.)

## Fonctionnalités

- Dashboard de suivi des performances de la stratégie
- Visualisation des indicateurs macroéconomiques et sectoriels
- Interface de configuration et de paramétrage de la stratégie
- Rapports de performance et d'attribution

## Comment exécuter

```bash
# Depuis la racine du projet
cd app
python app.py
```

L'application sera accessible à l'adresse `http://localhost:8050` ou `http://localhost:8501` en fonction du framework utilisé (Dash ou Streamlit).
