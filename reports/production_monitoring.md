# Suivi en production — ContextIQ

## Objectif et état actuel

Le suivi doit détecter une baisse de qualité, une panne ou une réponse dangereuse, puis permettre une correction vérifiable. **MLflow est utilisé en local pour suivre les évaluations** ; **Loguru** conserve les traces techniques des opérations.

La version actuelle enregistre une exécution MLflow par méthode et valeur de K via `MLflowTracker.evaluate()`. Elle ne déploie pas de service de production, de collecte des retours conseillers ni de système automatique d’alertes. Le plan ci-dessous décrit le fonctionnement à appliquer pour un pilote. Aucun accès à des dossiers clients n’est ajouté.

## Utiliser le suivi local

1. Installer les dépendances avec `poetry install`.
2. Exécuter le notebook `08_mlflow_monitoring.ipynb` pour enregistrer les évaluations.
3. Depuis la racine du projet, lancer `make mlflow-ui`.
4. Ouvrir **http://127.0.0.1:5000** ; sélectionner l’expérience `contextiq-rag-assistant` et comparer les runs. Arrêter le serveur avec Ctrl+C.

Le nom de l’expérience, le dossier de stockage et le port se règlent dans `configs/monitoring.yaml`. Adapter l’adresse si le port change. Le dossier relatif est résolu depuis le parent de `configs/`, ce qui conserve le même stockage depuis un notebook.

MLflow utilise `mlflow/tracking.db` pour les métadonnées et `mlflow/artifacts/` pour les artefacts. Le client communique directement avec SQLite : l’interface n’a pas besoin de tourner pour enregistrer un run. Le serveur reste lié à `127.0.0.1`. La télémétrie est désactivée par le tracker et la commande Make. Aucun autolog n’est activé.

Chaque exécution conserve :

- Méthode, K, seuil et paramètres BM25 ou nom/révision du modèle d’embeddings.
- Empreintes des documents, annotations, configuration, code Python et lockfile disponible.
- Versions des bibliothèques et version Python.
- Métriques de retrieval, qualité des réponses, effectifs, fausses acceptations et durée de la boucle d’évaluation.
- Artefacts JSON : configuration de recherche sans chemin de cache, métriques agrégées et versions.

Les questions, documents, réponses et identifiants de sources ne sont pas enregistrés. Les empreintes servent à identifier les changements, pas à reconstituer une version : conserver séparément les versions autorisées du corpus. Le dossier MLflow est ignoré par Git. Un run terminé avec succès indique que l’évaluation a fonctionné, pas que le système est approuvé pour la production.

## Indicateurs, contrôles et responsabilités proposés

| Domaine | Indicateurs et source | Fréquence proposée | Responsable et réaction |
| --- | --- | --- | --- |
| Qualité documentaire | Recall@K, MRR@K et classements sur un jeu fixe ; résultats MLflow | Avant chaque changement et revue hebdomadaire en pilote | AI engineer : comparer à la référence sur les mêmes cas et investiguer les régressions. |
| Qualité de réponse | Succès conjoint, fidélité, citations, faux refus et effectifs ; MLflow + revue humaine | Chaque version et échantillon hebdomadaire | Référent métier : vérifier que la source permet réellement de répondre ; bloquer les erreurs commerciales critiques. |
| Hors périmètre | Fausses acceptations parmi les cas sans réponse ; MLflow | Chaque version | Métier + AI engineer : revoir les erreurs et la règle de décision, sans régler le seuil sur le test final. |
| Disponibilité | Exceptions, opérations échouées et durée ; traces Loguru | Contrôle quotidien, puis collecte continue si service déployé | Exploitant : analyser l’incident, rétablir la version précédente ou suspendre le service. |
| Latence | Médiane et p95 par requête sur un échantillon représentatif | Quotidienne en pilote | Exploitant : vérifier la charge, les ressources et l’index. La mesure `evaluation_seconds` n’est pas une latence de service. |
| Sécurité | Attaques manquées, faux blocages et incidents confirmés ; jeu dédié et signalements | Chaque version et après incident | Référent sécurité : suspendre une version ayant divulgué une donnée ou restitué une instruction dangereuse, analyser et revalider. |
| Utilité conseiller | Réponses utiles, incorrectes, incomplètes et refus injustifiés ; retours qualifiés | Revue hebdomadaire | Référent métier : arbitrer, corriger la documentation ou créer des cas d’évaluation. |

Les métriques sur le trafic réel exigent une collecte supplémentaire, à concevoir avec les règles de confidentialité. Les seules évaluations périodiques ne détectent pas toutes les dérives d’usage. Les taux actuels sur six questions ne doivent pas devenir des objectifs de production.

## Alertes et décision de mise en service

Avant un pilote, le métier et l’exploitant fixent les seuils acceptables de qualité, de faux refus, de fausses acceptations, d’erreurs techniques et de latence sur un jeu suffisamment représentatif. Ces limites ne sont pas encore approuvées.

Pour chaque revue, comparer les runs MLflow à une version de référence validée, en vérifiant que les empreintes des questions et les dénominateurs sont comparables. Une variation de corpus doit être explicitée. En cas d’écart, ouvrir un incident local avec l’identifiant des runs, l’indicateur, la gravité, un responsable et la décision : investiguer, corriger, suspendre ou revenir en arrière.

Toute erreur commerciale critique ou divulgation confirmée exige une analyse avant remise en service. En l’absence de seuils approuvés et de validation indépendante, la réussite technique des tests ne vaut pas autorisation de mise en production.

**Dans cette première version, les alertes sont une procédure de revue, pas des notifications automatiques MLflow.** Leur automatisation pourra être ajoutée après validation des indicateurs et des responsables.

## Retours des conseillers

Prévoir un formulaire minimal : catégorie du problème, identifiant de version, date, appréciation utile/inutile et commentaire expurgé des données personnelles. Les éléments nécessaires à une analyse détaillée doivent rester dans un espace autorisé, distinct des métriques agrégées MLflow.

Le référent métier relit les signalements chaque semaine, élimine les doublons, qualifie le défaut et définit la réponse attendue. Un cas confirmé devient un test de non-régression. Distinguer un document manquant, une source mal classée, un mauvais refus et une règle commerciale ambiguë. Les cas nouvellement utilisés pour corriger le système appartiennent au développement ; ils ne constituent plus un test indépendant.

Le formulaire et ce circuit ne sont pas encore implémentés.

## Après modification de documents ou de paramètres

1. Conserver la version précédente autorisée des documents et la configuration ; vérifier la cohérence et faire relire les nouvelles règles métier.
2. Charger les documents avec Pydantic, reconstruire les index et contrôler les limites de longueur du modèle d’embeddings.
3. Exécuter `make ci`, puis les évaluations métier et de sécurité avec les vrais poids. Les tests unitaires seuls ne mesurent pas la qualité réelle du retrieval.
4. Enregistrer de nouveaux runs MLflow avec les empreintes et comparer les métriques à la référence ; inspecter les cas critiques : plafonds, éligibilité, hors périmètre, citations et injection.
5. Obtenir la validation métier/sécurité pour une version candidate, puis surveiller un pilote limité. Ne pas promouvoir automatiquement le meilleur score sur six questions.
6. En cas de régression critique, restaurer documents, paramètres et index précédents, puis vérifier à nouveau. MLflow conserve la trace des mesures mais ne réalise pas le retour arrière.

## Confidentialité et conservation

Le dossier MLflow reste local. Pour un pilote, désigner les personnes autorisées, définir une durée de conservation des évaluations et sauvegarder ensemble la base SQLite et les artefacts lorsque les écritures sont arrêtées. La rotation Loguru ne purge pas MLflow ; sa conservation nécessite une procédure distincte. Aucun dépôt distant ni service hébergé n’est nécessaire à ce suivi.

## Limites et étape suivante

Le suivi local est opérationnel pour les évaluations. La supervision continue, les alertes automatiques, les retours conseillers et les critères chiffrés de validation restent à mettre en place si un pilote est décidé. La prochaine priorité de qualité demeure l’enrichissement et la séparation des jeux de développement, calibration et test.

Référence technique : [MLflow — stockage local SQLite](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/).
