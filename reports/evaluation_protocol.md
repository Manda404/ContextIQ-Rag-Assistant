# Protocole d’évaluation de ContextIQ

## 1. Objectif et statut

L’assistant doit fournir au conseiller une information pertinente, fidèle aux documents,
correctement sourcée, ou refuser lorsque le contexte ne permet pas de répondre. La qualité
du classement, celle de la réponse et la sécurité sont évaluées séparément.

Le projet dispose actuellement de six documents et de six questions, dont cinq avec source
et une hors périmètre. Ce jeu a déjà été exploré pendant le développement : il constitue
un jeu de diagnostic, pas un test indépendant. Les seuils YAML sont illustratifs.
Le présent protocole définit la démarche à appliquer ; les nouveaux jeux et leur validation
indépendante ne sont pas encore réalisés. Le rapport distingue les mesures exécutées des
étapes proposées.

## 2. Constituer et annoter les cas

Conserver un identifiant de cas, la question, le scénario métier, le périmètre attendu,
les sources et passages justificatifs, une réponse de référence et les faits obligatoires.
Annoter les montants, plafonds, délais, exceptions et conditions d’éligibilité. Une source
thématiquement proche ne suffit pas si elle ne permet pas de répondre à la question.

Distinguer les étiquettes :

- Réponse disponible dans la base, avec une ou plusieurs sources suffisantes.
- Information absente, même si la question concerne l’opérateur : refus attendu.
- Question extérieure au domaine : refus attendu.
- Question ambiguë ou sources contradictoires : refus ou clarification attendue selon la règle métier retenue.
- Demande interdite ou injection : blocage attendu, indépendamment du score de retrieval.

Le schéma actuel `q` / `gold_docs` couvre uniquement les cas simples avec source et hors
périmètre. Les autres catégories nécessitent des annotations supplémentaires ; elles ne
sont pas déduites automatiquement d’une liste de sources vide.

Couvrir tous les documents, notamment D6, et les situations suivantes :

| Famille | Cas à couvrir |
|---|---|
| Résiliation | Contrat mensuel, annuel, résiliation anticipée, exception de déménagement et justificatif. |
| Conditions commerciales | Plafond de 150 euros ; ancienneté de 6, 12 et 13 mois ; retard de paiement ; durée et signe de la remise. |
| Formulations | Paraphrases, synonymes, accents, fautes, questions courtes et différences entre « coupée » et « suspendue ». |
| Information absente | Offre non documentée, date manquante, information client non disponible, question proche du domaine sans source suffisante. |
| Contexte complexe | Sources complémentaires, source hors sujet mieux classée, documents contradictoires, passage trop long. |
| Sécurité | Injection dans un document, dans la question ou l’identifiant ; variantes de langue, de ponctuation et d’espacement ; citations pédagogiques bénignes. |

Les cas sont relus par une personne connaissant les règles métier. Une seconde lecture
est particulièrement utile pour les cas critiques ; les désaccords sont arbitrés et la
décision conservée. Les trois réponses imposées par l’énoncé restent des cas de revue
manuelle documentés, sans servir à calibrer les seuils.

## 3. Séparer développement, calibration et test

Enrichir le jeu avant de prétendre constituer un test représentatif. Répartir les cas par
groupes de scénarios et de variantes proches : une question et ses paraphrases ne doivent
pas se retrouver des deux côtés de la séparation. La base documentaire indexée peut rester
la même : le test porte sur des questions non utilisées pour le réglage.

Le jeu de développement sert aux choix de prétraitement, de méthode, de K et de paramètres.
Une partie de calibration ou une validation par groupes sert à fixer le seuil propre à
chaque méthode. Le test final est conservé à l’écart, puis exécuté une fois la configuration
figée. Un jeu de sécurité dédié mesure les attaques et les faux blocages ; ses cas finaux
ne servent pas à compléter les signatures du filtre avant leur évaluation.

Ne pas transformer les six questions déjà étudiées en test indépendant par un simple
redécoupage. Les conserver pour le diagnostic et la non-régression. Après toute révision
fondée sur une erreur du test, constituer de nouveaux cas indépendants avant de renouveler
une conclusion finale.

## 4. Fixer la configuration et l’environnement

Avant chaque comparaison, figer les documents, annotations, YAML, code, dépendances et
révision du modèle. Conserver leurs empreintes, la date, la version Python et les versions
principales des bibliothèques. Tous les retrievers reçoivent la même base et les mêmes
questions ; les annotations ne participent ni à l’indexation ni à la production de réponse.

TF-IDF et BM25 utilisent le prétraitement lexical commun. Les embeddings utilisent les
textes originaux et leur tokenizer. Rapporter cette différence de représentation plutôt
que prétendre comparer les méthodes avec un traitement identique. Les documents courts
sont indexés entiers. Un dépassement de longueur du modèle doit être signalé, pas tronqué
silencieusement. Les poids sont préchargés, puis l’inférence reste locale.

## 5. Évaluer chaque niveau

### Recherche documentaire

Calculer Precision@K, Recall@K, Hit Rate@K et MRR@K sur les questions avec source, pour les
K prévus dans `evaluation.yaml`. Precision@K garde K au dénominateur même si moins de K
résultats sont retournés. MRR@K vaut zéro si la première source pertinente est au-delà de K.
Conserver les classements et les scores par question pour l’analyse des erreurs.

Les questions sans source sont analysées séparément. Avec une seule source annotée par
question, Recall@K et Hit Rate@K coïncident ; ces deux colonnes ne constituent pas deux
preuves indépendantes. Des annotations non exhaustives peuvent sous-estimer la pertinence.

### Abstention et calibration

Rapporter le nombre et le taux de faux refus parmi les questions avec source, ainsi que
les fausses acceptations parmi les questions hors périmètre. Afficher le taux d’acceptation
global et les effectifs. Étudier les seuils sur le développement uniquement, séparément
pour chaque méthode. Les scores des méthodes ne partagent pas la même échelle.

Le choix du seuil dépend d’un compromis métier explicite : limiter les acceptations sans
réponse tout en maintenant une couverture utile. Faire valider les coûts et les limites
acceptables par le responsable métier avant la sélection finale. Ne pas choisir un seuil
uniquement pour améliorer un indicateur sur les six exemples disponibles.

Vérifier si un seuil simple peut séparer les cas. Si une question pertinente a un score
inférieur à une question hors périmètre, un seuil croissant sur ce score ne pourra pas
classer correctement les deux à la fois. Documenter cette limite et évaluer des améliorations
sur le développement avant de les retenir.

### Réponses extractives

Utiliser `ExtractiveAnswerEvaluator` pour vérifier la source attendue, la reproduction du
passage original, la citation et le refus. La référence est la base originale, indépendante
du texte déclaré par la réponse. Un refus correct utilise le message standard sans source
exposée pour une question annotée hors périmètre.

| Indicateur | Dénominateur |
|---|---|
| Fidélité verbatim et citation valide | Extraits produits. |
| Source attendue citée | Extraits produits pour les questions avec source. |
| Faux refus | Questions avec source. |
| Refus correct | Questions hors périmètre. |
| Taux de réponse et succès conjoint | Toutes les questions. |

Le succès conjoint exige les critères réunis pour un extrait, ou un refus correct pour une
question hors périmètre. Les groupes absents donnent `None`. Toujours publier les effectifs :
un bon score sur les seules réponses acceptées peut masquer beaucoup de refus.

Compléter les contrôles par une revue humaine : le passage répond-il effectivement à la
question, contient-il les conditions nécessaires et peut-il guider une bonne décision ?
Une citation existante ne garantit ni sa pertinence ni la justesse métier. Un extrait copié
correctement peut être hors sujet. Le contrôle verbatim n’est pas applicable tel quel aux
réponses libres ou reformulées d’un futur LLM ; celles-ci nécessitent une grille sémantique
et une revue des affirmations contre leurs sources.

### Sécurité

Mesurer les tentatives bloquées et les attaques manquées parmi les cas malveillants, puis
les faux blocages parmi les cas bénins. Vérifier qu’un contexte suspect ne produit ni prompt
prêt à envoyer ni extrait malveillant. Inclure une source injectée de score inférieur à la
meilleure source pour contrôler le blocage de tout le contexte.

Rapporter les résultats du filtre de signatures comme tels. Ils ne prouvent pas la résistance
d’un LLM, puisqu’aucun LLM n’est appelé ici. Pour une future intégration, évaluer également les
réponses réellement obtenues, les instructions obfusquées et les demandes de divulgation.
Les autorisations et l’accès aux données clients restent imposés hors du modèle.

### Temps de traitement

Séparer téléchargement, chargement du modèle, indexation et recherche. Après échauffement,
répéter les mesures, conserver le nombre de requêtes, la machine et les versions, puis
rapporter médiane et percentile 95 lorsque l’échantillon le permet. Ne pas déduire de SLA
ni de capacité de production de six questions sur six documents.

## 6. Analyser les erreurs et l’incertitude

Conserver un tableau par question : périmètre attendu, sources attendues, classement,
score, décision, réponse, citation, verdict et catégorie d’erreur. Distinguer défaut de
retrieval, seuil trop exigeant, fausse acceptation, annotation insuffisante, contradiction
métier et erreur de sécurité. Comparer les méthodes sur les mêmes cas.

Sur un jeu élargi, rapporter les effectifs et des intervalles d’incertitude adaptés aux
proportions. Pour comparer des méthodes, préserver l’appariement par scénario et utiliser,
si nécessaire, un rééchantillonnage par groupes. Aucun intervalle sur un seul cas hors
périmètre ne transforme ce cas en preuve de robustesse ; les mesures actuelles restent
descriptives et aucune supériorité générale n’est établie.

## 7. Critères de décision avant un pilote

Définir et figer les critères avec les responsables métier et sécurité avant le test final :
qualité minimale de réponse, couverture utile, limites de faux refus et de fausses
acceptations, temps de réponse et traitement des erreurs critiques.

Le jeu critique doit notamment vérifier les plafonds, l’éligibilité, les citations connues
et l’absence de contenu malveillant restitué. Une contradiction commerciale critique ou une
divulgation observée entraîne l’analyse du défaut et une nouvelle validation avant un pilote.
L’absence de défaut sur un jeu fini ne garantit pas une absence de risque en exploitation.
Les valeurs chiffrées d’acceptation ne sont pas encore approuvées et le système actuel ne
peut pas être déclaré validé pour la production.

## 8. Non-régression, traçabilité et restitution

Relancer les contrôles de code et les tests unitaires avec `make ci`. Relancer séparément
les évaluations avec le vrai modèle après modification du corpus, du prétraitement, du
retriever, des seuils, du prompt ou du générateur. La CI ne télécharge pas les poids et ne
mesure pas la performance réelle du modèle : ses tests utilisent des données fictives.

Archiver localement la configuration figée, les empreintes, les résultats détaillés et le
rapport. Les conclusions doivent nommer le jeu utilisé, les dénominateurs, les défauts et
les travaux restant à effectuer. Les données, l’énoncé et les travaux restent locaux,
conformément aux conditions du test technique. Le [plan de suivi en production](production_monitoring.md) décrit le suivi local MLflow,
les indicateurs et les procédures à prévoir pour un pilote. La collecte continue et les alertes
automatiques ne sont pas encore déployées.
