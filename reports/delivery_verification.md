# Vérification du parcours utilisateur avant remise

**Date : 17 novembre 2026.** Les étapes ci-dessous ont été réellement exécutées sur macOS ARM64, depuis un projet sans environnement Python local. Python 3.15.2 a été fourni temporairement pour le test ; Poetry 2.1.3 et Make étaient déjà disponibles sur la machine.

## Résultat du parcours

| Étape | Vérification effectuée | Résultat |
| --- | --- | --- |
| Installation | `make install`, installation des dépendances et du package, contrôle du Python utilisé et de la commande installée. | Réussie avec Python 3.15.2. |
| Noyau des notebooks | `make kernel`, contrôle du noyau ContextIQ et de son lien avec le Python du projet. | Réussie. |
| Exécution des notebooks | Tous les blocs de code exécutés dans huit noyaux Jupyter indépendants, depuis `notebooks/`. | 8 notebooks, 51 cellules, aucune erreur. |
| Première utilisation des embeddings | Initialisation du retriever avec un cache temporaire vide et les téléchargements autorisés, puis rechargement avec `local_files_only: true`. | Réussie ; D2 retrouvé en tête pour la question de résiliation annuelle. |
| Application | `make app`, réponse HTTP et parcours dans un navigateur isolé. | Réussie sur `127.0.0.1:8501`. |
| Discussion et journal | Envoi d’une question, réponse citant D2 et vérification de son enregistrement dans le journal local. | Réussie. |
| Revue humaine | Enregistrement d’un avis depuis la plateforme, puis vérification de sa persistance. | Réussie. |
| Évaluations de la plateforme | Lancement des trois méthodes, affichage des deux tableaux et contrôle des métriques enregistrées dans MLflow. | Réussie. |
| Suivi MLflow | `make mlflow-ui`, vérification HTTP de l’interface et enregistrement d’un bilan de conversations. | Réussie sur `127.0.0.1:5000`. |
| Navigation et thème | Consultation de la base, du guide et basculement vers le thème sombre. | Réussis dans le navigateur. |
| Qualité du code | `make ci` : validation Poetry, lint, formatage et tests automatisés. | 142 tests réussis ; contrôles conformes. |

L’installation crée désormais explicitement `.venv` dans le projet grâce au réglage du Makefile, afin de rendre son emplacement et son nettoyage prévisibles.

## Notebooks exécutés

| Notebook | Cellules de code exécutées | Résultat |
| --- | --- | --- |
| 01 — Exploration des données | 6 | Réussi |
| 02 — Méthodes de recherche | 9 | Réussi |
| 03 — Évaluation du retrieval | 5 | Réussi |
| 04 — Abstention | 7 | Réussi |
| 05 — Prompt et sécurité | 6 | Réussi |
| 06 — Réponses extractives | 6 | Réussi |
| 07 — Évaluation des réponses | 9 | Réussi |
| 08 — Suivi MLflow | 3 | Réussi |

L’exécution a utilisé de vrais noyaux IPython et les vrais poids d’embeddings. Les fichiers `.ipynb`, les données fournies et les YAML ont conservé leurs empreintes initiales. La sélection du noyau dans l’interface de VS Code n’a pas été automatisée ; son enregistrement et son exécution ont été vérifiés via Jupyter.

## Contrôle des résultats de recherche

Le parcours de la plateforme à K = 1 retrouve les précisions moyennes suivantes sur les cinq questions avec source :

| Méthode | Precision@1 |
| --- | --- |
| TF-IDF | 0,600 |
| BM25 | 0,400 |
| Embeddings | 0,800 |

Ces valeurs correspondent aux résultats déjà documentés. Cette vérification confirme le fonctionnement du parcours et la cohérence des mesures ; elle ne transforme pas les six questions connues en test indépendant et ne constitue pas une validation de production.

## État pour la remise

Le nettoyage a été effectué après arrêt des serveurs : `.venv`, le Python temporaire, le cache modèle de test, le noyau de test et les caches/journaux générés ont été retirés. Les sources, données, notebooks, configurations et résultats documentés sont conservés. Les fichiers des notebooks, des données et des configurations ont conservé leurs empreintes initiales.

Le détail des vérifications et l’état final du nettoyage figurent dans [le relevé JSON](metrics/delivery_verification_2026-11-17.json).
