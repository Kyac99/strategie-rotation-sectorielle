"""
Application Streamlit pour la stratégie de rotation sectorielle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ajout du répertoire parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation des modules personnalisés
try:
    from src.data.macro_data_collector import MacroDataCollector
    from src.data.sector_data_collector import SectorDataCollector
    from src.models.economic_cycle_classifier import EconomicCycleClassifier
    from src.models.sector_selector import SectorSelector
except ImportError as e:
    st.error(f"Erreur d'importation des modules: {e}")
    st.info("Veuillez vérifier que le projet est correctement installé.")


def load_data():
    """
    Charge les données macroéconomiques et sectorielles.
    """
    # Chemins des données
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    macro_path = os.path.join(data_dir, "macro_data.csv")
    sector_path = os.path.join(data_dir, "sector_data.csv")
    phases_path = os.path.join(data_dir, "economic_phases.csv")
    recommendations_path = os.path.join(data_dir, "sector_recommendations.csv")
    
    # Chargement du modèle
    cycle_model_path = os.path.join(models_dir, "economic_cycle_classifier.joblib")
    
    data = {}
    
    # Chargement des données macroéconomiques
    if os.path.exists(macro_path):
        data['macro'] = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    else:
        st.warning("Données macroéconomiques non trouvées. Certaines fonctionnalités seront limitées.")
        data['macro'] = None
    
    # Chargement des données sectorielles
    if os.path.exists(sector_path):
        data['sector'] = pd.read_csv(sector_path, index_col=0, parse_dates=True)
    else:
        st.warning("Données sectorielles non trouvées. Certaines fonctionnalités seront limitées.")
        data['sector'] = None
    
    # Chargement des phases économiques
    if os.path.exists(phases_path):
        phases_df = pd.read_csv(phases_path)
        phases_df['date'] = pd.to_datetime(phases_df['date'])
        phases_df.set_index('date', inplace=True)
        data['phases'] = phases_df
    else:
        data['phases'] = None
    
    # Chargement des recommandations sectorielles
    if os.path.exists(recommendations_path):
        data['recommendations'] = pd.read_csv(recommendations_path, index_col=0)
    else:
        data['recommendations'] = None
    
    # Chargement du modèle
    if os.path.exists(cycle_model_path):
        try:
            data['cycle_model'] = EconomicCycleClassifier.load_model(cycle_model_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {e}")
            data['cycle_model'] = None
    else:
        data['cycle_model'] = None
    
    return data


def plot_economic_indicators(data):
    """
    Affiche les principaux indicateurs économiques.
    """
    st.subheader("Indicateurs économiques")
    
    # Liste des indicateurs économiques importants à afficher
    indicators = {
        'GDPC1_YOY': 'Croissance du PIB (% annuel)',
        'UNRATE': 'Taux de chômage (%)',
        'CPIAUCSL_YOY': 'Inflation (% annuel)',
        'FEDFUNDS': 'Taux directeur (%)',
        'T10Y2Y': 'Spread de taux 10 ans - 2 ans',
        'VIXCLS': 'Indice de volatilité VIX'
    }
    
    # Filtrage des indicateurs disponibles
    available_indicators = {k: v for k, v in indicators.items() if k in data['macro'].columns}
    
    if available_indicators:
        # Sélection de la période
        start_date = st.date_input(
            "Date de début",
            value=pd.Timestamp(data['macro'].index[-60]),
            min_value=pd.Timestamp(data['macro'].index[0]),
            max_value=pd.Timestamp(data['macro'].index[-1])
        )
        
        end_date = st.date_input(
            "Date de fin",
            value=pd.Timestamp(data['macro'].index[-1]),
            min_value=pd.Timestamp(start_date),
            max_value=pd.Timestamp(data['macro'].index[-1])
        )
        
        # Filtrage des données selon la période sélectionnée
        filtered_data = data['macro'].loc[start_date:end_date].copy()
        
        # Création du graphique avec plotly
        fig = make_subplots(
            rows=len(available_indicators), 
            cols=1,
            subplot_titles=list(available_indicators.values()),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, (indicator, label) in enumerate(available_indicators.items(), 1):
            if indicator in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index, 
                        y=filtered_data[indicator],
                        mode='lines',
                        name=label
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            height=300 * len(available_indicators),
            width=800,
            showlegend=False,
            title_text="Évolution des indicateurs économiques"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucun indicateur économique disponible.")


def plot_economic_cycles(data):
    """
    Affiche les phases du cycle économique identifiées.
    """
    st.subheader("Cycles économiques")
    
    if data['phases'] is not None:
        # Création d'une variable numérique pour les phases
        phase_mapping = {phase: i for i, phase in enumerate(data['phases']['phase'].unique())}
        data['phases']['phase_numeric'] = data['phases']['phase'].map(phase_mapping)
        
        # Création du graphique avec plotly
        fig = go.Figure()
        
        # Ajout des données
        for phase, numeric in phase_mapping.items():
            phase_data = data['phases'][data['phases']['phase'] == phase]
            fig.add_trace(
                go.Scatter(
                    x=phase_data.index,
                    y=[numeric] * len(phase_data),
                    mode='markers',
                    name=phase,
                    marker=dict(
                        size=10,
                        opacity=0.8
                    )
                )
            )
        
        # Configuration de l'axe y pour afficher les noms des phases
        fig.update_layout(
            title="Phases du cycle économique",
            xaxis_title="Date",
            yaxis=dict(
                tickmode='array',
                tickvals=list(phase_mapping.values()),
                ticktext=list(phase_mapping.keys())
            ),
            height=500,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage de la phase actuelle
        current_phase = data['phases']['phase'].iloc[-1]
        st.info(f"Phase économique actuelle: **{current_phase}**")
        
        # Affichage du nombre de mois passés dans cette phase
        current_phase_start = data['phases'][data['phases']['phase'] != current_phase].index[-1]
        months_in_phase = (data['phases'].index[-1] - current_phase_start).days // 30
        st.info(f"Durée de la phase actuelle: **{months_in_phase} mois**")
    else:
        st.warning("Données des cycles économiques non disponibles.")


def plot_sector_performance(data):
    """
    Affiche les performances des secteurs.
    """
    st.subheader("Performance des secteurs")
    
    if data['sector'] is not None:
        # Liste des ETFs sectoriels
        sector_etfs = {
            'XLY': 'Consommation discrétionnaire',
            'XLP': 'Consommation de base',
            'XLE': 'Énergie',
            'XLF': 'Finance',
            'XLV': 'Santé',
            'XLI': 'Industrie',
            'XLB': 'Matériaux',
            'XLK': 'Technologie',
            'XLU': 'Services publics',
            'XLRE': 'Immobilier',
            'XLC': 'Services de communication'
        }
        
        # Filtrage des secteurs disponibles
        available_sectors = [s for s in sector_etfs.keys() if s in data['sector'].columns]
        
        if available_sectors:
            # Sélection de la période
            periods = ['1 mois', '3 mois', '6 mois', '1 an', '3 ans', '5 ans', 'Tout']
            selected_period = st.selectbox("Période", periods, index=3)
            
            # Définition de la date de début selon la période sélectionnée
            end_date = data['sector'].index[-1]
            
            if selected_period == '1 mois':
                start_date = end_date - pd.DateOffset(months=1)
            elif selected_period == '3 mois':
                start_date = end_date - pd.DateOffset(months=3)
            elif selected_period == '6 mois':
                start_date = end_date - pd.DateOffset(months=6)
            elif selected_period == '1 an':
                start_date = end_date - pd.DateOffset(years=1)
            elif selected_period == '3 ans':
                start_date = end_date - pd.DateOffset(years=3)
            elif selected_period == '5 ans':
                start_date = end_date - pd.DateOffset(years=5)
            else:  # Tout
                start_date = data['sector'].index[0]
            
            # Filtrage des données selon la période sélectionnée
            filtered_data = data['sector'].loc[start_date:end_date].copy()
            
            # Calcul des rendements cumulés
            returns = {}
            for sector in available_sectors:
                if sector in filtered_data.columns:
                    sector_data = filtered_data[sector]
                    total_return = (sector_data.iloc[-1] / sector_data.iloc[0] - 1) * 100
                    returns[sector] = total_return
            
            # Création du DataFrame pour plotly
            performance_df = pd.DataFrame({
                'Secteur': [f"{s} ({sector_etfs[s]})" for s in returns.keys()],
                'Rendement (%)': list(returns.values())
            })
            
            # Tri par rendement
            performance_df = performance_df.sort_values('Rendement (%)', ascending=False)
            
            # Création du graphique avec plotly
            fig = px.bar(
                performance_df,
                x='Secteur',
                y='Rendement (%)',
                title=f"Performance des secteurs sur la période: {selected_period}",
                color='Rendement (%)',
                color_continuous_scale='RdYlGn',
                height=500
            )
            
            # Ajout de la ligne horizontale à 0
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(performance_df) - 0.5,
                y0=0,
                y1=0,
                line=dict(color='black', width=1, dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Affichage de l'évolution des prix
            st.subheader("Évolution des prix des secteurs")
            
            # Normalisation des prix (base 100 au début de la période)
            normalized_prices = filtered_data[available_sectors].copy()
            for sector in available_sectors:
                normalized_prices[sector] = normalized_prices[sector] / normalized_prices[sector].iloc[0] * 100
            
            # Création du graphique avec plotly
            fig = go.Figure()
            
            for sector in available_sectors:
                fig.add_trace(
                    go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices[sector],
                        mode='lines',
                        name=f"{sector} ({sector_etfs[sector]})"
                    )
                )
            
            fig.update_layout(
                title=f"Évolution des prix (base 100) - {selected_period}",
                xaxis_title="Date",
                yaxis_title="Prix (base 100)",
                height=500,
                width=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun secteur disponible.")
    else:
        st.warning("Données sectorielles non disponibles.")


def plot_sector_recommendations(data):
    """
    Affiche les recommandations sectorielles selon la phase économique.
    """
    st.subheader("Recommandations sectorielles")
    
    if data['recommendations'] is not None and data['phases'] is not None:
        # Obtention de la phase actuelle
        current_phase = data['phases']['phase'].iloc[-1]
        
        # Obtention des recommandations pour la phase actuelle
        if current_phase in data['recommendations'].index:
            recommendations = data['recommendations'].loc[current_phase]
            
            # Affichage des recommandations
            st.info(f"Phase économique actuelle: **{current_phase}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Secteurs recommandés:**\n\n{recommendations['Top 3 secteurs']}")
            with col2:
                st.error(f"**Secteurs à éviter:**\n\n{recommendations['À éviter']}")
            
            # Affichage du tableau des recommandations pour toutes les phases
            st.subheader("Recommandations par phase du cycle")
            st.dataframe(data['recommendations'])
        else:
            st.warning(f"Pas de recommandations disponibles pour la phase {current_phase}.")
    else:
        st.warning("Données de recommandations non disponibles.")


def run_backtest(data):
    """
    Exécute un backtest de la stratégie de rotation sectorielle.
    """
    st.subheader("Backtesting de la stratégie")
    
    if data['macro'] is not None and data['sector'] is not None and data['cycle_model'] is not None:
        # Paramètres du backtest
        st.write("Paramètres du backtest")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Date de début",
                value=pd.Timestamp('2010-01-01'),
                min_value=pd.Timestamp(max(data['macro'].index[0], data['sector'].index[0])),
                max_value=pd.Timestamp(min(data['macro'].index[-1], data['sector'].index[-1]))
            )
        
        with col2:
            end_date = st.date_input(
                "Date de fin",
                value=pd.Timestamp(min(data['macro'].index[-1], data['sector'].index[-1])),
                min_value=pd.Timestamp(start_date),
                max_value=pd.Timestamp(min(data['macro'].index[-1], data['sector'].index[-1]))
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rebalance_freq = st.selectbox(
                "Fréquence de rééquilibrage",
                options=['Mensuelle', 'Trimestrielle'],
                index=0
            )
            freq = 'M' if rebalance_freq == 'Mensuelle' else 'Q'
        
        with col2:
            num_sectors = st.slider(
                "Nombre de secteurs",
                min_value=1,
                max_value=5,
                value=3,
                step=1
            )
        
        with col3:
            momentum_weight = st.slider(
                "Poids du momentum",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # Bouton pour lancer le backtest
        if st.button("Lancer le backtest"):
            with st.spinner("Backtesting en cours..."):
                try:
                    # Filtrage des données selon la période sélectionnée
                    macro_bt = data['macro'].loc[start_date:end_date].copy()
                    sector_bt = data['sector'].loc[start_date:end_date].copy()
                    
                    # Création du sélecteur de secteurs
                    selector = SectorSelector(cycle_classifier=data['cycle_model'])
                    
                    # Exécution du backtest
                    results, allocations = selector.backtest_strategy(
                        macro_bt, sector_bt,
                        start_date=start_date, end_date=end_date,
                        rebalance_freq=freq, num_sectors=num_sectors,
                        momentum_weight=momentum_weight
                    )
                    
                    # Calcul des métriques de performance
                    metrics = selector.calculate_performance_metrics(results)
                    
                    # Affichage des résultats
                    st.subheader("Résultats du backtest")
                    
                    # Métriques de performance
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            label="Rendement annualisé",
                            value=f"{metrics['Rendement annualisé (%)']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Volatilité annualisée",
                            value=f"{metrics['Volatilité annualisée (%)']:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            label="Ratio de Sharpe",
                            value=f"{metrics['Ratio de Sharpe']:.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            label="Drawdown maximum",
                            value=f"{metrics['Maximum drawdown (%)']:.2f}%"
                        )
                    
                    # Graphique de performance
                    fig = selector.plot_performance(results)
                    st.pyplot(fig)
                    
                    # Graphique des allocations
                    fig = selector.plot_allocations(allocations)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erreur lors du backtest: {e}")
    else:
        st.warning("Données ou modèle manquants pour exécuter le backtest.")


def collect_new_data():
    """
    Collecte de nouvelles données macroéconomiques et sectorielles.
    """
    st.subheader("Collecte de nouvelles données")
    
    st.warning("Cette fonctionnalité nécessite des clés API pour FRED et potentiellement d'autres sources.")
    
    if st.button("Collecter de nouvelles données"):
        with st.spinner("Collecte des données en cours..."):
            try:
                # Création des répertoires si nécessaires
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
                os.makedirs(data_dir, exist_ok=True)
                
                # Collecte des données macroéconomiques
                st.info("Collecte des données macroéconomiques...")
                macro_collector = MacroDataCollector()
                macro_data = macro_collector.get_all_series(start_date="2000-01-01", frequency='m')
                processed_macro = macro_collector.preprocess_data(macro_data)
                
                # Sauvegarde des données macroéconomiques
                macro_path = os.path.join(data_dir, "macro_data.csv")
                processed_macro.to_csv(macro_path)
                st.success(f"Données macroéconomiques sauvegardées dans {macro_path}")
                
                # Collecte des données sectorielles
                st.info("Collecte des données sectorielles...")
                sector_collector = SectorDataCollector()
                etf_data = sector_collector.get_all_etf_data(start_date="2000-01-01")
                processed_sectors = sector_collector.preprocess_data(etf_data)
                
                # Sauvegarde des données sectorielles
                sector_path = os.path.join(data_dir, "sector_data.csv")
                processed_sectors.to_csv(sector_path)
                st.success(f"Données sectorielles sauvegardées dans {sector_path}")
                
                # Entraînement du modèle de classification des cycles
                st.info("Entraînement du modèle de classification des cycles économiques...")
                models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
                os.makedirs(models_dir, exist_ok=True)
                
                classifier = EconomicCycleClassifier(supervised=False)
                classifier.fit(processed_macro)
                
                # Sauvegarde du modèle
                model_path = os.path.join(models_dir, "economic_cycle_classifier.joblib")
                classifier.save_model(model_path)
                st.success(f"Modèle de classification sauvegardé dans {model_path}")
                
                # Identification des phases du cycle
                st.info("Identification des phases du cycle économique...")
                phases = classifier.predict(processed_macro)
                
                # Sauvegarde des phases
                phases_df = pd.DataFrame({'phase': phases}, index=phases.index)
                phases_path = os.path.join(data_dir, "economic_phases.csv")
                phases_df.to_csv(phases_path)
                st.success(f"Phases économiques sauvegardées dans {phases_path}")
                
                # Notification de succès
                st.success("Collecte et traitement des données terminés avec succès.")
                st.info("Veuillez rafraîchir l'application pour voir les nouvelles données.")
                
            except Exception as e:
                st.error(f"Erreur lors de la collecte des données: {e}")


def main():
    """
    Fonction principale de l'application Streamlit.
    """
    st.set_page_config(
        page_title="Stratégie de Rotation Sectorielle",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Stratégie de Rotation Sectorielle Dynamique")
    
    # Chargement des données
    data = load_data()
    
    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sélectionner une page",
        [
            "Accueil",
            "Indicateurs économiques",
            "Cycles économiques",
            "Performance des secteurs",
            "Recommandations sectorielles",
            "Backtest de la stratégie",
            "Collecte de données"
        ]
    )
    
    # Affichage de la page sélectionnée
    if page == "Accueil":
        st.write("""
        ## Bienvenue dans l'application de Stratégie de Rotation Sectorielle Dynamique
        
        Cette application permet d'analyser les cycles économiques et d'identifier les opportunités 
        de rotation sectorielle pour maximiser le rendement ajusté au risque de votre portefeuille.
        
        ### Fonctionnalités
        
        - **Indicateurs économiques**: Visualisation des principaux indicateurs macroéconomiques
        - **Cycles économiques**: Identification des phases du cycle économique
        - **Performance des secteurs**: Analyse des performances sectorielles
        - **Recommandations sectorielles**: Recommandations basées sur les phases du cycle
        - **Backtest de la stratégie**: Simulation de la stratégie de rotation sectorielle
        - **Collecte de données**: Mise à jour des données macroéconomiques et sectorielles
        
        Utilisez la barre latérale pour naviguer entre les différentes fonctionnalités.
        """)
        
        # Affichage des infos sur les données
        st.subheader("Informations sur les données")
        
        # Données macroéconomiques
        if data['macro'] is not None:
            st.info(f"Données macroéconomiques disponibles: {data['macro'].shape[0]} observations du {data['macro'].index[0].strftime('%d/%m/%Y')} au {data['macro'].index[-1].strftime('%d/%m/%Y')}")
        else:
            st.warning("Données macroéconomiques non disponibles.")
        
        # Données sectorielles
        if data['sector'] is not None:
            st.info(f"Données sectorielles disponibles: {data['sector'].shape[0]} observations du {data['sector'].index[0].strftime('%d/%m/%Y')} au {data['sector'].index[-1].strftime('%d/%m/%Y')}")
        else:
            st.warning("Données sectorielles non disponibles.")
        
        # Phase économique actuelle
        if data['phases'] is not None:
            current_phase = data['phases']['phase'].iloc[-1]
            st.success(f"Phase économique actuelle: **{current_phase}**")
        else:
            st.warning("Données des cycles économiques non disponibles.")
    
    elif page == "Indicateurs économiques":
        plot_economic_indicators(data)
    
    elif page == "Cycles économiques":
        plot_economic_cycles(data)
    
    elif page == "Performance des secteurs":
        plot_sector_performance(data)
    
    elif page == "Recommandations sectorielles":
        plot_sector_recommendations(data)
    
    elif page == "Backtest de la stratégie":
        run_backtest(data)
    
    elif page == "Collecte de données":
        collect_new_data()


if __name__ == "__main__":
    main()
