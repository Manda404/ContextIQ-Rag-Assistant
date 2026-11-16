# Audit du projet ContextIQ-Rag-Assistant

Date : **15 novembre 2026**. Référence : `note/README-0.md`.

> Mise à jour après audit : le suivi local MLflow et le [plan de suivi en production](production_monitoring.md) ont été ajoutés. Les constats ci-dessous décrivent l’état avant cet ajout ; le principal livrable documentaire manquant est désormais présent. La collecte continue et les alertes automatiques restent proposées, sans déploiement.

Voir aussi la [revue exhaustive actualisée après MLflow](requirements_review.md), qui distingue les exigences de la partie 2, les bonus et les livrables globaux.

## Conclusion

Le projet répond largement au périmètre fonctionnel de la **partie 2 — assistant RAG et évaluation** : recherche, comparaison de méthodes, prompt sourcé, refus, protection contre l’injection fournie et analyse des trois réponses imposées. Le manque explicite principal est le **plan de suivi en production**, encore annoncé comme restant à rédiger.

La réalisation technique est plus avancée que le minimum demandé, mais sa validation métier reste exploratoire. Trois retrievers et 120 tests ne compensent pas les limites des six questions disponibles. Le système n’est pas validé pour la production.

Si ce dépôt est censé constituer la remise de **tout le test**, il est incomplet : la partie 1, présentée comme le cœur du test, et les scénarios de la partie 3 ne sont pas présents ici. Leur éventuelle réalisation dans un autre projet n’a pas été auditée.

## Méthode et vérifications exécutées

- Lecture intégrale de l’énoncé et des deux rapports d’évaluation existants.
- Inspection des modules de chargement, recherche, génération, sécurité, évaluation, configuration et journalisation, ainsi que des sept notebooks.
- Exécution locale de `make ci` : validation Poetry, Ruff, formatage et **120 tests réussis** ; 50 fichiers reconnus correctement formatés.
- Recalcul sur les données actuelles avec les trois méthodes, dont le vrai modèle d’embeddings en cache, sans téléchargement : métriques pour K = 1, 2, 3 et évaluation des réponses après abstention.
- Inspection de l’état Git et des fichiers de livrables. Aucune publication ni modification du pipeline réalisée pendant l’audit.

Les notebooks contiennent des sorties sans cellule d’erreur enregistrée, mais ils n’ont pas tous été réexécutés intégralement pendant cet audit. La vérification locale du workflow ne prouve pas son exécution réussie sur les runners GitHub.

## Correspondance avec la partie 2

| Demande de l’énoncé | Statut | Réalisation et réserve |
| --- | --- | --- |
| Implémenter un retriever TF-IDF ou embeddings | Réalisé, au-delà du minimum | TF-IDF, BM25 et embeddings multilingues exposent une interface de recherche commune dans `src/contextiq/retrieval/`. |
| Évaluer sur le jeu fourni | Réalisé | Precision, Recall, Hit Rate et MRR pour plusieurs K ; détails et erreurs dans le notebook 03. Les résultats sont descriptifs sur cinq questions avec source. |
| Expliquer le traitement du hors périmètre | Réalisé, efficacité partielle | Politique d’abstention et notebook 04 ; faux refus et fausses acceptations mesurés. Seuils illustratifs et un seul exemple hors périmètre. |
| Prompt système français, citations, refus si information absente | Réalisé | `PromptBuilder` définit les consignes et construit les messages système/utilisateur à partir des sources. Le texte du prompt n’est pas une garantie que le contexte est suffisant. |
| Traiter le document injecté de l’énoncé | Réalisé pour le cas connu | Détection heuristique et blocage de tout contexte suspect avant production ; tests et notebooks 05/06. Pas de validation générale contre les attaques. |
| Juger fidélité et pertinence des trois réponses fournies | Réalisé | Notebook 07 et rapport : plafond annuel omis, remise inéligible à six mois, préavis mensuel correct. |
| Proposer un protocole d’évaluation complet | Réalisé comme proposition | `evaluation_protocol.md` décrit annotations, séparation des jeux, métriques, sécurité, revue métier et critères de décision. Les nouveaux jeux indépendants ne sont pas encore constitués. |
| Proposer un suivi en production | Partiel, livrable manquant | Logs locaux présents, mais absence d’un plan opérationnel précisant indicateurs, alertes, responsables, retours conseillers et procédure après modification documentaire. |

**Un appel à un LLM n’est pas exigé par l’énoncé.** Son absence n’est donc pas un défaut de conformité. La réponse extractive est une réalisation supplémentaire : elle copie un document et le cite, sans produire une synthèse ni raisonner sur l’éligibilité d’un client.

## Ce qui constitue déjà une base solide

- Chargement portable, modèles Pydantic, validation des références et configurations YAML séparées.
- Logique réutilisable dans `src/`, appelée depuis les notebooks ; pas de dépendance aux chemins personnels dans les API.
- Comparaison des méthodes sur les mêmes documents et questions ; annotations utilisées pour l’évaluation, pas pour produire les réponses.
- Contrôle des textes trop longs pour les embeddings, sans troncature silencieuse.
- Mesures de recherche, d’abstention et de réponse séparées, avec dénominateurs explicites.
- Journalisation Loguru configurable, console désactivée par défaut, fichier local avec rotation et rétention ; instrumentation automatique sans contenu brut des questions et documents.
- Tests unitaires et contrôles qualité reproductibles. Les tests embeddings utilisent un modèle simulé ; le recalcul métier de cet audit utilise les vrais poids.

## Résultats vérifiés et défauts observés

Résultats recalculés avec les seuils actuels : TF-IDF 0,3 ; BM25 2,0 ; embeddings 0,4.

| Méthode | Source attendue au rang 1, avant refus | Succès final conjoint | Faux refus | Hors périmètre accepté |
| --- | --- | --- | --- | --- |
| TF-IDF | 3/5 | 2/6 | 2/5 | 1/1 |
| BM25 | 2/5 | 2/6 | 2/5 | 1/1 |
| Embeddings | 4/5 | 4/6 | 1/5 | 0/1 |

Le succès conjoint exige la source attendue, la fidélité de l’extrait et une citation valide, ou un refus correct pour le cas hors périmètre. Chaque méthode produit quatre extraits, tous fidèles à leur document et correctement cités. **Une copie fidèle d’un document hors sujet reste une mauvaise réponse.**

### Défauts prioritaires

1. **Classement de la question sur la remise.** La meilleure source est incorrecte avec les trois méthodes. La bonne source peut être dans le top 2 sans être restituée : le générateur ne retient que le premier document.
2. **Décision de refus sur la ligne coupée.** Les embeddings retrouvent la bonne source, mais son score d’environ 0,314 est inférieur au seuil 0,4. Une question répondable est refusée.
3. **Limite structurelle du seuil.** Le cas hors périmètre obtient environ 0,360, au-dessus de la question pertinente précédente. Aucun seuil unique croissant sur ce score ne peut accepter l’une et refuser l’autre simultanément. Baisser le seuil seul ne résout pas ce défaut.
4. **Couverture très faible.** Six questions, une seule hors périmètre et aucune question dédiée à D6. Pas de validation indépendante, de cas multi-sources, de contradictions ni de mesure de robustesse aux paraphrases.
5. **Prétraitement numérique imparfait.** La tokenisation lexicale supprime le signe dans `-30 %` et découpe `0,9`. Les sources restituées restent intactes, mais la représentation de recherche perd une information métier.
6. **Sécurité évaluée sur des signatures connues.** Le filtre peut manquer des reformulations et bloquer une citation pédagogique bénigne. Les tests de blocage ne mesurent pas un taux de résistance sur un jeu indépendant.

Ces défauts sont déjà en partie documentés dans le rapport existant : l’enjeu est désormais de les traiter et de mesurer les effets, plutôt que d’ajouter uniquement de nouvelles méthodes.

## Suivi en production : ce qui reste à décrire

Les logs répondent au besoin de diagnostic technique, pas à l’ensemble du suivi demandé. Un plan court doit préciser :

- **Indicateurs** : erreurs techniques, latence, taux de réponse/refus, sources utilisées, pertinence et fidélité sur un échantillon revu, incidents de sécurité.
- **Alertes et actions** : référence de comparaison, seuils à faire valider, fréquence de contrôle, destinataire responsable, investigation et suspension ou retour arrière si nécessaire.
- **Retours des conseillers** : signalement d’une réponse incorrecte ou inutile, contexte minimal autorisé, qualification et revue métier, transformation en cas de non-régression.
- **Changement de documents** : validation du schéma et du contenu, reconstruction de l’index, réévaluation des cas critiques et de sécurité, comparaison avant/après et possibilité de revenir à la version précédente.
- **Traçabilité et confidentialité** : versions du corpus et des paramètres, accès et rétention des traces, sans journaliser les données clients par défaut.

Pour satisfaire l’énoncé, ce plan peut être un document. MLflow, LangSmith, une API et un tableau de bord ne sont pas des prérequis.

## Périmètre global et livrables

| Élément | État dans ce dépôt |
| --- | --- |
| Partie 1 : churn, campagne, gain annuel et incertitude, contrainte juridique, pilote 2023 | Non trouvée ; hors du périmètre RAG effectivement développé. Bloquant pour une remise globale si elle n’existe pas ailleurs. |
| Partie 3 : au moins deux scénarios écrits | Non trouvée. À intégrer à la synthèse de la remise globale. |
| Partie 4 : industrialisation réutilisable de la partie 1 | Non vérifiable ici. La structure Python du RAG ne remplace pas cette exigence portant explicitement sur le churn. |
| Projet Python RAG, schémas et CI | Présents. |
| Note de synthèse de 1 à 2 pages | Non trouvée. Le rapport technique détaillé ne la remplace pas. |
| Présentation de 10 minutes, PDF ou PowerPoint | Non trouvée. |
| Historique Git | Un commit visible ; dépôt distant `origin` et référence `origin/main` présents. L’historique progressif du développement n’est pas représenté. |
| Archive finale incluant `.git` | Non trouvée dans le périmètre inspecté. |

### Consigne de remise à clarifier

L’énoncé interdit explicitement tout dépôt en ligne, public ou privé. Le dépôt local dispose pourtant d’un remote GitHub et d’une référence de suivi `origin/main`. Cela ne permet pas d’établir l’accessibilité actuelle du dépôt distant ni une éventuelle dérogation, mais exige de vérifier l’autorisation avant toute nouvelle publication. L’exclusion de `data/` ne suffit pas à régler ce point : notebooks, sorties et rapports reprennent aussi des éléments du travail et du contexte métier.

Aucune action sur le dépôt distant n’a été effectuée pendant cet audit.

## Cohérence documentaire et éléments secondaires

- Le rapport du 9 novembre annonce 103 tests lors de son exécution ; l’état courant en possède 120. Conserver la distinction entre mesure historique et vérification actuelle.
- L’instantané des métriques fige une ancienne version du code et de la configuration. Les métriques principales ont été retrouvées pendant cet audit, mais il ne constitue pas automatiquement un instantané du code actuel.
- `chunking.py` et `ranking.py` sont des emplacements sans implémentation. Le découpage et un reranker ne sont pas requis pour satisfaire le sujet avec six documents courts.
- La CLI expose uniquement aide et version. C’est une limite d’ergonomie, pas un manque explicite de l’énoncé : les classes et notebooks permettent déjà l’utilisation.
- Aucun benchmark de charge ou SLA n’est établi. Ce n’est pas nécessaire pour le prototype, mais il faut éviter de présenter les temps sur six questions comme une validation de capacité.

## Ordre proposé pour avancer progressivement

| Bloc | Action | Critère de fin |
| --- | --- | --- |
| 1 | Rédiger le plan de suivi en production. | Document opérationnel couvrant indicateurs, alertes, retours et changements documentaires. Ferme le principal manque explicite de la partie 2. |
| 2 | Enrichir et faire relire les cas d’évaluation, sans modifier les fichiers fournis. | Couverture de D6, conditions commerciales, hors périmètre proche, variantes et sécurité ; séparation développement/calibration/test documentée. |
| 3 | Corriger la représentation des nombres et signes. | Tests ciblés et comparaison avant/après ; sources originales inchangées. |
| 4 | Travailler le classement et la décision de réponse. | Comparaison sur le développement d’une amélioration simple, puis validation sur des cas indépendants ; ne pas se limiter à baisser le seuil. |
| 5 | Consolider la remise. | Rapport daté, synthèse courte, slides et archive Git locale ; vérification du périmètre global et de l’autorisation de partage. |

Ne pas ajouter un LLM, un service de suivi d’expériences ou une quatrième méthode avant de fermer ces lacunes : ces ajouts ne sont pas nécessaires pour répondre aux exigences restantes.
