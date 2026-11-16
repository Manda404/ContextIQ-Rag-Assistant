# Revue de conformité aux exigences

Date : **15 novembre 2026**. Référence : `note/README-0.md`.
État examiné : branche `feat/monitoring-mlflow`, commit `2a47cab`.

## Verdict

**Toutes les demandes explicites de la partie 2 ont une réalisation ou une proposition documentée dans ce projet.** Le suivi en production, manquant lors du premier audit, dispose maintenant d’un plan et d’une implémentation locale MLflow pour les évaluations.

Cette conclusion porte sur les livrables demandés : elle ne certifie ni une qualité de réponse suffisante en exploitation ni une résistance générale aux injections. L’énoncé demande de proposer un protocole et un suivi ; il ne demande pas de déployer une supervision continue ni d’appeler un LLM.

**L’ensemble du test technique n’est pas couvert par ce seul dépôt** : la partie 1, les scénarios de la partie 3 et plusieurs livrables finaux ne sont pas présents ici. Aucun autre projet n’a été audité pour les déclarer réalisés.

## Vérification de chaque demande de la partie 2

« Couvert » signifie que la demande a une preuve identifiable. Pour une proposition, la preuve attendue est un document ; pour une fonction, du code et des vérifications.

| N° | Exigence détaillée | Statut | Preuve | Réserve |
| --- | --- | --- | --- | --- |
| 1 | Implémenter un retriever TF-IDF ou embeddings | Couvert | `src/contextiq/retrieval/tfidf.py`, `embeddings.py` ; BM25 en complément | Trois méthodes, alors qu’une seule suffisait. |
| 2 | Évaluer sur le jeu fourni | Couvert | Notebook 03 ; `evaluation/evaluator.py`, `retrieval_metrics.py` ; `evaluation_report.md` | Six questions seulement ; résultats exploratoires. |
| 3 | Expliquer le traitement du hors périmètre | Couvert | Notebook 04 ; `safety/abstention.py` ; `evaluation/abstention_metrics.py` | Seuils illustratifs, faux refus et fausses acceptations encore observés. |
| 4 | Rédiger un prompt système | Couvert | `generation/prompt_builder.py`, `PromptBuilder.SYSTEM_PROMPT` ; notebook 05 | Prompt préparé, aucun appel LLM. |
| 5 | Construire le prompt à partir du contexte récupéré | Couvert | `PromptBuilder.build()` : question et documents en JSON dans un message utilisateur distinct | La structure sépare les rôles, sans constituer une protection absolue. |
| 6 | Exiger une réponse en français | Couvert | Consigne explicite dans le prompt système | Le générateur actuel copie des documents français. |
| 7 | Exiger la citation des sources | Couvert | Consigne de citation par affirmation ; identifiants du contexte ; citations extractives et tests | Une citation valide ne prouve pas la pertinence de la source. |
| 8 | Prévoir le refus si l’information est absente | Couvert au niveau demandé | Consigne de refus dans le prompt ; politique d’abstention ; refus sans contexte et tests | Le seuil ne détecte pas parfaitement toutes les absences d’information. Ce comportement n’est pas garanti sur toute question. |
| 9 | Expliquer et traiter l’injection documentaire donnée | Couvert pour le cas demandé | `safety/prompt_injection.py`, `PromptBuilder.build()` ; notebooks 05/06 ; tests de blocage | Le filtre bloque le contexte suspect ; variantes inconnues et faux blocages possibles. |
| 10 | Juger fidélité et pertinence de la réponse sur les frais annuels | Couvert | Notebook 07, cas 1 ; section 6 du rapport | Verdict : sur le sujet mais non fidèle ; le plafond de 150 euros manque. |
| 11 | Juger fidélité et pertinence de la remise à six mois d’ancienneté | Couvert | Notebook 07, cas 2 ; section 6 du rapport | Verdict : sur le sujet mais recommandation fausse ; condition de plus de 12 mois et absence de retard omise. |
| 12 | Juger fidélité et pertinence du préavis mensuel | Couvert | Notebook 07, cas 3 ; section 6 du rapport | Verdict : fidèle et pertinent, citation D1 correcte. |
| 13 | Proposer un protocole d’évaluation complet | Couvert comme proposition | [Protocole d’évaluation](evaluation_protocol.md) | Décrit annotations, séparation des jeux, métriques, revue métier, sécurité, incertitude, critères et non-régression. Le jeu indépendant reste à constituer. |
| 14 | Proposer un suivi en production | Couvert comme proposition, avec réalisation partielle | [Plan de suivi](production_monitoring.md) ; `monitoring/mlflow_tracking.py` ; notebook 08 ; `configs/monitoring.yaml` | Évaluations locales suivies ; journal automatique des échanges et revue humaine ajoutés dans la plateforme. Alertes automatiques non déployées. |

L’énoncé ne fixe pas de seuil minimal de performance, de nombre de méthodes ou de technologie de monitoring. Il ne demande ni interface Streamlit, ni base vectorielle, ni reranker, ni conteneur. Ces éléments peuvent être utiles sans être nécessaires à la couverture formelle de la partie 2.

## Vérifications techniques de cette revue

`make ci` a été relancé sur cet état :

- Validation Poetry et lockfile : réussie.
- Ruff : aucune erreur.
- Formatage : 54 fichiers conformes.
- Tests unitaires : **126 réussis**.

Les tests de suivi utilisent un vrai stockage SQLite temporaire ; les tests unitaires d’embeddings utilisent des vecteurs simulés. Le recalcul avec les vrais poids, l’exécution des cellules du notebook 08 et le test HTTP de l’interface MLflow ont été réalisés lors des étapes précédentes. Ils n’ont pas été tous répétés pendant cette revue. Aucun succès du workflow distant n’est déduit de ces vérifications locales.

## Ce qui reste à améliorer sans constituer une exigence oubliée

- Enrichir le jeu avec D6, des variantes, plusieurs questions hors périmètre et des cas métier critiques ; séparer développement, calibration et test.
- Corriger le classement de la question sur la remise et la décision de refus pour la ligne coupée ; un seuil unique ne suffit pas à séparer tous les cas actuels.
- Préserver les signes et les décimales dans la représentation lexicale : `-30 %` perd son signe et `0,9` est découpé.
- Évaluer les injections et faux blocages sur des cas indépendants.
- Valider les critères d’acceptation métier et sécurité avant un pilote.
- Actualiser certaines formulations historiques : le notebook 07 annonce encore le suivi en production comme restant à réaliser ; un plan existe désormais et les évaluations sont suivies, tandis que la supervision continue reste future. Le rapport historique mentionne 103 tests à sa date de mesure, contre 126 dans l’état courant.

Les résultats précédemment recalculés restent de 4 succès conjoints sur 6 pour les embeddings, contre 2 sur 6 pour TF-IDF et BM25. Ils ne permettent pas d’annoncer une supériorité générale ni une validation de production.

## Autres parties de l’énoncé : vérification exhaustive du périmètre

| Exigence | État dans ce dépôt |
| --- | --- |
| Partie 1.1 — Exploration churn et choix de traitement | Non trouvée. L’exploration documentaire du RAG ne remplace pas l’exploration churn. |
| Partie 1.2 — Modèle churn et évaluation | Non trouvé. |
| Partie 1.3 — Ciblage mensuel et gain annuel avec incertitude | Non trouvé : capacité de 300 contacts et économie de campagne non traitées ici. |
| Partie 1.4 — Explicabilité et interdiction d’usage direct de l’âge | Analyse liée au ciblage churn non trouvée. |
| Partie 1.5 — Efficacité de campagne et pilote 2023 | Non trouvée : causalité, sélection des contactés et validation de l’acceptation non traitées ici. |
| Partie 1.6 — Exploitation durable du churn | Non trouvée : labels à 30 jours, réentraînement mensuel, séparation temporelle et scoring de 1,2 million de clients non couverts ici. |
| Partie 3 — Au moins deux scénarios, une page maximum chacun | Aucun document répondant aux scénarios concurrent, valeur future/offres, biais de contact, discrimination ou budget réduit trouvé. Seuls deux scénarios sont obligatoires. |
| Partie 4 — Partie 1 livrée comme projet Python réutilisable | Non vérifiable dans ce dépôt RAG. La réutilisabilité du RAG ne satisfait pas à elle seule cette demande portant sur le churn. |
| Bonus — Suivi d’expériences | Présent pour le RAG avec MLflow. |
| Bonus — Validation de schéma | Présente avec Pydantic. |
| Bonus — Intégration continue | Workflow configuré ; contrôles locaux réussis ; résultat distant non vérifié. |
| Bonus — Conteneur | Non trouvé ; facultatif. |
| Bonus — API | Pas de service HTTP. Les classes Python sont réutilisables ; le serveur de suivi MLflow ne constitue pas une API métier RAG. Facultatif. |
| Bonus — Plan de monitoring | Présent pour le RAG. |

Les bonus sont cités dans la partie 4 portant sur la partie 1 : leur réalisation sur le RAG constitue un complément utile, sans démontrer l’industrialisation du churn.

## Conditions et livrables de remise

| Demande | État constaté |
| --- | --- |
| Projet | Projet RAG présent ; test global incomplet dans ce dépôt. |
| Synthèse de 1 à 2 pages | Non trouvée. Les rapports détaillés ne remplacent pas ce livrable. |
| Présentation PDF ou PowerPoint pour 10 minutes | Non trouvée. |
| Dépôt Git local et historique | Présents : deux commits visibles. |
| Archive ZIP incluant `.git` | Non trouvée. |
| Absence de publication en ligne | Conformité non établie : remote GitHub et référence `origin/main` présents. Le dernier push a été bloqué par la revue automatique ; cela ne prouve pas l’absence de publication antérieure. |
| Envoi par le canal autorisé et suppression des données à la fin | Étapes de clôture non vérifiées ; aucune suppression ni transmission réalisée pendant la revue. |

## Bonus Streamlit : réalisation ajoutée après la revue

La plateforme locale est désormais présente dans `src/contextiq/ui/`, avec une dépendance Streamlit et une commande `make app`. Elle appelle les classes existantes pour :

1. Poser une question, choisir TF-IDF, BM25 ou embeddings, K et le seuil ; afficher la réponse ou le refus et les sources.
2. Comparer les classements et les métriques, en conservant les effectifs et les limites visibles.
3. Lancer une évaluation et consulter ses références MLflow.
4. Explorer les documents et comprendre les contrôles de sécurité.

L’interface préserve la validation Pydantic, la configuration portable, l’abstention et les contrôles de sécurité. Une action explicite lance une évaluation ; un simple rafraîchissement ne crée pas de run MLflow. Les textes des sources et des questions sont affichés sans interprétation HTML. La simulation est une suite de questions indépendantes, sans appel LLM.

Référence visuelle retenue après clarification : **BRVM-Stock-Price-Prediction**, dont le fichier `brvm_dashboard.py` a été inspecté. Reprendre sa présentation large, sa palette verte, ses cartes d’indicateurs, sa barre latérale et son choix clair/sombre, en les adaptant au parcours RAG. L’interface reprend cette référence dans un espace adapté au RAG. Les vérifications de la plateforme sont ajoutées après la revue initiale ; le total de tests indiqué plus haut décrit l’état avant cet ajout.
