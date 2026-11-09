# Rapport d’évaluation — ContextIQ RAG Assistant

Mesures recalculées le **2026-11-09T00:34:02+01:00**. Statut : **étude exploratoire,
sans validation indépendante et sans accord de mise en production**.

Le système retrouve des documents, applique une règle d’abstention et un filtre
d’injection, puis restitue le document le mieux classé avec sa citation. Aucun LLM
n’est appelé. Les chiffres sont issus de [l’instantané JSON](metrics/evaluation_snapshot.json).
Le [protocole d’évaluation](evaluation_protocol.md) précise les étapes de validation
restant à appliquer et les critères à faire approuver.

## 1. Conclusion et décision actuelle

Sur les six questions fournies, les embeddings obtiennent **4 succès conjoints sur 6**,
contre **2 sur 6** pour TF-IDF et BM25. Cette différence est observée avec les seuils
illustratifs actuels ; elle ne prouve pas une supériorité générale.

Les embeddings constituent une piste prioritaire pour poursuivre les essais, sans
être désignés comme solution validée. Leur politique refuse encore une question pertinente
et leur meilleure source reste incorrecte pour une question de remise. La recommandation
est de poursuivre la validation locale, d’enrichir les cas et de confirmer la pertinence
métier avant un pilote. Le système actuel n’est pas déclaré prêt pour la production.

## 2. Données, configuration et reproductibilité

La base contient six documents et le jeu six questions : cinq avec une source attendue,
une hors périmètre. D6 n’a pas de question dédiée. Chaque question pertinente possède
une seule source annotée. Les questions ont déjà été examinées pendant le développement :
il ne s’agit pas d’un jeu de test indépendant. Aucun seuil n’a été réoptimisé pour ce rapport.

| Paramètre | Valeur |
| --- | --- |
| Top K utilisé pour les réponses | 3 |
| K comparés pour le retrieval | [1, 2, 3] |
| BM25 k1 / b | 1.5 / 0.75 |
| Seuil TF-IDF / BM25 / Embeddings | 0.3 / 2.0 / 0.4 |
| Modèle | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Révision | e8f8c211226b894fcb81acc59f3b34ba3efd5f42 |
| Exécution | Python 3.15.2, Darwin arm64, CPU, poids déjà en cache |

Les documents sont indexés entiers. TF-IDF et BM25 partagent la tokenisation lexicale ;
les embeddings utilisent les textes originaux et leur tokenizer. L’inférence de cette
exécution était hors ligne. Les empreintes du corpus, des annotations, de la configuration,
du code source et du lockfile sont conservées dans le JSON avec les versions des bibliothèques.

Pour reproduire la démarche, exécuter les notebooks 03, 04 et 07 avec les mêmes fichiers,
YAML et révision du modèle. Le notebook 06 montre les réponses produites. Comparer les
métriques aux champs `methods` du JSON. Une nouvelle exécution ne constitue pas une nouvelle
validation indépendante. Le rapport et son instantané décrivent la configuration mesurée,
pas automatiquement toute modification ultérieure du projet.

## 3. Résultats du retrieval

Moyennes sur les **cinq questions avec source**, avant abstention. Precision@K conserve K
au dénominateur ; MRR@K vaut zéro si la source pertinente est absente du top K. Les questions
hors périmètre sont exclues de ces moyennes.

| Méthode | K | Precision@K | Recall@K | Hit Rate@K | MRR@K |
| --- | --- | --- | --- | --- | --- |
| TF-IDF | 1 | 0.600 | 0.600 | 0.600 | 0.600 |
| TF-IDF | 2 | 0.400 | 0.800 | 0.800 | 0.700 |
| TF-IDF | 3 | 0.267 | 0.800 | 0.800 | 0.700 |
| BM25 | 1 | 0.400 | 0.400 | 0.400 | 0.400 |
| BM25 | 2 | 0.500 | 1.000 | 1.000 | 0.700 |
| BM25 | 3 | 0.333 | 1.000 | 1.000 | 0.700 |
| Embeddings | 1 | 0.800 | 0.800 | 0.800 | 0.800 |
| Embeddings | 2 | 0.500 | 1.000 | 1.000 | 0.900 |
| Embeddings | 3 | 0.333 | 1.000 | 1.000 | 0.900 |

Les embeddings placent la source attendue en tête pour quatre questions sur cinq,
TF-IDF pour trois, BM25 pour deux. BM25 et les embeddings retrouvent les cinq sources
attendues dans le top 2. L’extracteur ne restitue cependant que la meilleure source :
retrouver la bonne source en deuxième position ne suffit pas à produire le bon passage.
Recall@K et Hit Rate@K sont identiques ici, car chaque question n’a qu’une source attendue.

## 4. Réponses finales et refus

Le succès conjoint exige une source attendue, un passage fidèle et une citation valide
pour une question avec source, ou un refus correct pour une question hors périmètre.
Les effectifs ci-dessous portent sur les réponses finales, après les contrôles.

| Méthode | Extraits / questions | Succès conjoint / questions | Source attendue / extraits avec source attendue | Faux refus / questions avec source | Fausses acceptations / hors périmètre |
| --- | --- | --- | --- | --- | --- |
| TF-IDF | 4/6 | 2/6 | 2/3 | 2/5 | 1/1 |
| BM25 | 4/6 | 2/6 | 2/3 | 2/5 | 1/1 |
| Embeddings | 4/6 | 4/6 | 3/4 | 1/5 | 0/1 |

| Méthode | Fidélité verbatim / extraits | Citation valide / extraits |
| --- | --- | --- |
| TF-IDF | 4/4 | 4/4 |
| BM25 | 4/4 | 4/4 |
| Embeddings | 4/4 | 4/4 |

Les quatre extraits de chaque méthode reproduisent leur source et possèdent une citation
valide. C’est attendu pour notre générateur qui copie les passages : **100 % de fidélité
verbatim ne signifie pas 100 % de réponses pertinentes**. Une citation connue peut pointer
vers un document hors sujet. Les contrôles ne jugent pas le sens de réponses libres de LLM.

## 5. Analyse par question

« Succès » désigne le critère conjoint, pas une certification de justesse sémantique.
« Refus » sur une question avec source signale ici un faux refus selon les annotations.

| Cas | Question | Attendu | TF-IDF | BM25 | Embeddings |
| --- | --- | --- | --- | --- | --- |
| Q1 | Combien de temps de préavis pour un contrat mensuel ? | D1 | D1 — succès | D1 — succès | D1 — succès |
| Q2 | Quels frais si je quitte mon abonnement annuel avant la fin ? | D2 | D2 — succès | D2 — succès | D2 — succès |
| Q3 | Je déménage, puis-je partir sans payer ? | D3 | Refus — échec | Refus — échec | D3 — succès |
| Q4 | Ma ligne est coupée, que faire ? | D4 | Refus — échec | Refus — échec | Refus — échec |
| Q5 | Puis-je proposer une remise à un client qui veut partir ? | D5 | D1 — échec | D1 — échec | D3 — échec |
| Q6 | Quel est le cours de l'action de l'entreprise ? | Hors périmètre | D4 — échec | D1 — échec | Refus — succès |

Trois constats expliquent les écarts :

1. **Ligne coupée (Q4).** Les embeddings classent D4 en tête, mais le score 0.314
   reste sous le seuil 0,4 : une recherche correcte devient un refus. TF-IDF et BM25 refusent
   également cette question avec leurs seuils actuels.
2. **Remise commerciale (Q5).** Les embeddings placent D3 devant D5 ; TF-IDF et BM25
   restituent D1. Le générateur copie correctement un passage qui ne répond pas à la question.
3. **Hors périmètre (Q6).** TF-IDF et BM25 acceptent et citent une source hors sujet.
   Les embeddings refusent cet exemple, mais un seul cas ne démontre pas une bonne robustesse.

Le score de la question pertinente Q4 (0.314) est inférieur à celui de Q6
(0.360) avec les embeddings. Avec une acceptation fondée sur `score >= seuil`,
aucun seuil unique ne peut accepter Q4 et refuser Q6 simultanément. La difficulté ne se
résout donc pas uniquement par une meilleure valeur du seuil ; elle justifie des essais
sur une règle de décision plus informative et davantage de cas, sans modifier le test final.

## 6. Revue des trois réponses imposées par l’énoncé

Cette revue est **manuelle** et fondée sur les documents, distincte des mesures automatiques
sur les extraits. La pertinence thématique et la justesse métier sont distinguées.

| Réponse fournie | Fidélité | Pertinence et justesse métier | Citation | Correction |
| --- | --- | --- | --- | --- |
| 50 % des mensualités restantes, sans plafond | Non : contradiction avec D2 | Sur le sujet, mais montant potentiellement excessif | Absente | 50 % des mensualités restantes, plafonnés à 150 euros. [D2] |
| Remise de -30 % pendant 3 mois possible à six mois | Non : condition de D5 omise | Sur le sujet, mais mauvaise décision d’éligibilité | Absente | Non à six mois : plus de 12 mois et absence de retard de paiement sont requis. [D5] |
| Préavis mensuel de 10 jours sans pénalité, source D1 | Oui | Pertinente et correcte pour le contrat mensuel | Correcte | Réponse conservée ; préciser le contrat mensuel pour éviter une généralisation. [D1] |

## 7. Sécurité et validation du code

Le constructeur de prompt sépare les instructions des données. Une source contenant
l’injection de l’énoncé est bloquée avant restitution, même si elle n’est pas la source
la mieux classée. Les tests et le notebook 05 vérifient ces cas connus, ainsi que les refus
sans message prêt à envoyer. Le notebook 06 vérifie aussi l’absence d’extrait malveillant.

Le filtre demeure une heuristique à signatures : les tests de variantes connues ne sont
pas une mesure de résistance générale. Aucun LLM, outil client ou accès aux dossiers clients
n’est exposé par ce projet. Il reste à mesurer les attaques manquées et les faux blocages
sur un jeu de sécurité indépendant. Aucun taux global de sécurité n’est revendiqué ici.

La validation locale `make ci` vérifie Poetry, Ruff, le formatage et **103 tests unitaires**.
Ces tests utilisent des données fictives et des modèles simulés pour les embeddings.
Ils vérifient les mécanismes du code et ne remplacent pas l’évaluation du vrai modèle.
La CI GitHub est configurée mais n’a pas été déclenchée. Le vrai modèle a été exécuté
localement pour les mesures du présent rapport.

## 8. Limites et travaux prioritaires

- Les six questions sont déjà connues du développement et les seuils ne sont pas calibrés indépendamment.
- Une seule question hors périmètre et l’absence de cas dédiés à D6 limitent fortement la couverture.
- Les annotations ne fournissent pas encore de faits obligatoires, passages suffisants ou scénarios ambigus ; un hit de source est un indicateur imparfait de pertinence.
- Une copie fidèle ne synthétise pas plusieurs sources et ne déduit pas une décision personnalisée. La source classée en tête peut être erronée.
- La tokenisation lexicale actuelle perd le signe de `-30 %` et sépare `0,9`. Les textes cités restent intacts ; le traitement lexical mérite une correction et une non-régression ciblée.
- La sécurité repose sur des signatures partielles et aucune validation indépendante d’attaque n’a été réalisée.
- Les essais de temps du notebook 02 ne permettent pas de conclure sur la capacité ou un SLA en production.

Appliquer ensuite le protocole : enrichir et faire relire les cas, séparer développement
et test par groupes de scénarios, calibrer les règles sur le développement, figer la
configuration et lancer le test indépendant. Les critères métier et sécurité doivent
être approuvés avant un pilote. Le plan de suivi en production reste un livrable distinct.

## 9. Traçabilité

L’instantané associé contient les métriques, les classements et scores par question,
les décisions, les textes effectivement restitués et les contrôles des réponses.
Les empreintes sont calculées sur les documents, annotations et settings sérialisés
canoniquement, ainsi que sur les fichiers Python du code source et le lockfile.

| Élément | SHA-256 |
| --- | --- |
| documents_sha256 | f5063795238a8330216c29473f1570b18bcc4312e1b5872e5e6a34c621189ced |
| questions_sha256 | 9d08e14a2b57d00fb4ebb8bbd7dafb0e73729dcf8249a7852d59177b0d7f3d0c |
| settings_sha256 | 8c9a9eb2ac97078f79421c05dc60185b46f6c175fec3235de9daf57fc5815857 |
| source_code_sha256 | a5762874080fb881c28c15220404f45c93d514f4faed17e1f571eb9e25d1f08f |
| poetry_lock_sha256 | 7386cd030ff7acd351e32602735bda1cde5f88000dc64864e3fef40be00fabd1 |

Les versions de Python et des bibliothèques figurent dans `runtime` et les paramètres
mesurés dans `settings` de l’instantané. Ces artefacts restent locaux conformément aux
conditions du test technique ; aucune publication n’a été effectuée pour cette exécution.
