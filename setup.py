"""
Configuration du package strategie-rotation-sectorielle
"""

from setuptools import setup, find_packages
import os

# Lecture du fichier README.md pour la description longue
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Lecture des dépendances depuis requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Filtrage des commentaires et lignes vides
    requirements = [req for req in requirements if not req.startswith("#") and req.strip()]

setup(
    name="strategie-rotation-sectorielle",
    version="0.1.0",
    author="Pêgdwendé Yacouba KONSEIGA",
    author_email="kyac.konseiga@example.com",
    description="Stratégie de rotation sectorielle dynamique basée sur les cycles économiques et indicateurs de marché",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kyac99/strategie-rotation-sectorielle",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rotation-app=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "data/processed/*.csv", "models/*.joblib"],
    },
)
