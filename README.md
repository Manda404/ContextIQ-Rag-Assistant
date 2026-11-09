# ContextIQ-Rag-Assistant

Un assistant documentaire en Python pour **retrouver des informations, produire des réponses sourcées et évaluer leur fiabilité**. Le projet compare trois méthodes de recherche sur une même base de connaissances : **TF-IDF, BM25 et embeddings multilingues**.

Les notebooks présentent les expériences et leurs résultats. La logique métier est regroupée dans des classes Python réutilisables, avec validation des données et des configurations par **Pydantic**.

## Fonctionnement

```text
Documents JSON → Validation → Index de recherche
                                      ↓
Question → Recherche → Décision d’abstention → Contrôles de sécurité
                                                      ↓
                                           Extrait cité ou refus
```

Le système restitue actuellement le meilleur document retenu, accompagné de son identifiant. Il prépare aussi un prompt pour une future génération par LLM, mais **aucun LLM n’est appelé**.

L’évaluation couvre :

- **Recherche** : Precision@K, Recall@K, Hit Rate@K et MRR@K.
- **Abstention** : faux refus et acceptations de questions hors périmètre.
- **Réponses** : pertinence de la source, fidélité de l’extrait, validité des citations et refus corrects.
- **Sécurité** : détection heuristique d’injections de prompt dans les questions et les documents.

## Installation

Prérequis : **Python 3.15.2** et **Poetry 2.1.3**. Depuis la racine du projet :

```sh
poetry env use python3.15
poetry install
make ci
```

Dans votre éditeur, ouvrez les notebooks avec l’environnement Python créé par Poetry. L’utilisation de TF-IDF, de BM25 et des réponses extractives ne nécessite aucune clé d’API.

### Données

Placez les deux fichiers UTF-8 suivants dans `data/raw/`, ou fournissez votre propre dossier au chargeur. Les exemples ci-dessous illustrent le format attendu et ne reproduisent pas les données du projet.

**`knowledge_base.json`** — liste de documents aux identifiants uniques :

```json
[
  {"id": "DOC-1", "text": "Le support est ouvert du lundi au vendredi."}
]
```

**`eval_set.json`** — questions et identifiants des documents attendus :

```json
[
  {"q": "Quand le support est-il ouvert ?", "gold_docs": ["DOC-1"]},
  {"q": "Quel temps fera-t-il demain ?", "gold_docs": []}
]
```

Une liste `gold_docs` vide désigne une question hors périmètre. Les références sont vérifiées au chargement. Le dossier `data/` est exclu de Git.

### Modèle d’embeddings

Le modèle utilisé est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Son nom et sa révision sont définis dans [configs/retrieval.yaml](configs/retrieval.yaml).

Par défaut, `local_files_only: true` impose l’utilisation de poids déjà présents dans le cache local. Pour une première utilisation sans cache, passez temporairement cette valeur à `false`, puis initialisez `EmbeddingRetriever` dans le notebook 02. Cette étape télécharge les poids et nécessite une connexion Internet. Remettez ensuite la valeur à `true` pour les exécutions hors ligne.

Les poids sont conservés dans le cache utilisateur Hugging Face, ou dans le dossier indiqué par `cache_dir`. Les vecteurs des documents sont calculés à l’initialisation du retriever et conservés en mémoire.

## Utilisation rapide

Depuis la racine du projet ou le dossier `notebooks/`, cet exemple utilise BM25 sans télécharger de modèle :

```python
from contextiq.generation.extractive_answer import ExtractiveAnswerGenerator
from contextiq.ingestion.loader import DataLoader
from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.settings import Settings

settings = Settings.load()
dataset = DataLoader().load()

retriever = BM25Retriever(
    dataset.documents,
    **settings.retrieval.bm25.model_dump(),
)
policy = AbstentionPolicy(
    threshold=settings.retrieval.abstention_thresholds["BM25"],
)

question = "Quand le support est-il ouvert ?"  # À adapter à votre base.
results = retriever.retrieve(question, top_k=settings.retrieval.top_k)
decision = policy.decide(results)
answer = ExtractiveAnswerGenerator().generate(question, decision)

print(answer.text)
```

Pour utiliser des données situées ailleurs :

```python
dataset = DataLoader(data_dir="chemin/vers/mes-donnees").load()
```

Les chemins du projet sont détectés automatiquement depuis sa racine ou ses notebooks. Aucun chemin propre à une machine n’est nécessaire. La commande `contextiq` fournit actuellement uniquement l’aide et la version ; les expériences s’exécutent via les notebooks ou les classes Python.

## Parcours des notebooks

À consulter dans cet ordre ; chaque notebook initialise les objets dont il a besoin.

| Notebook | Objectif |
| --- | --- |
| [01 — Comprendre les données](notebooks/01_data_understanding.ipynb) | Examiner les documents, les questions et la couverture des annotations. |
| [02 — Expérimenter la recherche](notebooks/02_retrieval_experiments.ipynb) | Comprendre le prétraitement et comparer les résultats des trois méthodes. |
| [03 — Évaluer la recherche](notebooks/03_retrieval_evaluation.ipynb) | Comparer les métriques pour plusieurs valeurs de K et inspecter les erreurs. |
| [04 — Analyser l’abstention](notebooks/04_abstention_analysis.ipynb) | Explorer les seuils, les faux refus et les acceptations hors périmètre. |
| [05 — Prompt et sécurité](notebooks/05_prompt_and_safety.ipynb) | Construire un contexte sourcé et examiner les contrôles contre les injections. |
| [06 — Produire des réponses](notebooks/06_extractive_answers.ipynb) | Restituer un extrait cité ou refuser de répondre. |
| [07 — Évaluer les réponses](notebooks/07_answer_evaluation.ipynb) | Analyser des réponses métier et mesurer la qualité du pipeline extractif. |

## Configuration et logs

Les paramètres sont chargés par `Settings.load()` et validés avec Pydantic.

| Fichier | Paramètres |
| --- | --- |
| [retrieval.yaml](configs/retrieval.yaml) | Top K, BM25, modèle d’embeddings et seuils d’abstention par méthode. |
| [evaluation.yaml](configs/evaluation.yaml) | Valeurs de K et grilles de seuils à comparer. |
| [logging.yaml](configs/logging.yaml) | Affichage des logs, dossier, niveau, rotation et rétention. |

La journalisation **Loguru** écrit dans `logs/contextiq-rag-assistant.log`. L’affichage dans les notebooks et le terminal est **désactivé par défaut** :

```yaml
console: false
log_dir: logs
level: INFO
rotation: "10 MB"
retention: "7 days"
```

Passez `console` à `true` pour afficher aussi les logs, puis rechargez `Settings.load()`. Les traces automatiques enregistrent les opérations, durées et métadonnées techniques sans inclure le texte des questions, documents ou réponses. Les fichiers de logs sont exclus de Git.

Pour déplacer les données ou les configurations, utilisez `CONTEXTIQ_DATA_DIR` ou `CONTEXTIQ_CONFIG_DIR`, ou fournissez `data_dir` et `config_dir` aux classes concernées. Les variables doivent être exportées ou définies dans l’environnement Python : **le fichier `.env` n’est pas chargé automatiquement**. Les options disponibles figurent dans [.env.example](.env.example).

## Organisation du projet

```text
ContextIQ-Rag-Assistant/
├── configs/           # Paramètres YAML
├── data/raw/          # Données locales, exclues de Git
├── src/contextiq/
│   ├── ingestion/     # Chargement et prétraitement
│   ├── retrieval/     # TF-IDF, BM25 et embeddings
│   ├── generation/    # Prompt et réponses extractives
│   ├── evaluation/    # Métriques et évaluateurs
│   ├── safety/        # Abstention et détection d’injections
│   ├── schemas/       # Modèles Pydantic
│   ├── settings.py    # Chargement des configurations
│   └── logging.py     # Journalisation Loguru
├── notebooks/         # Parcours d’expérimentation et d’analyse
├── tests/             # Tests unitaires
├── reports/           # Protocole, résultats et analyse des erreurs
├── logs/              # Journaux locaux, exclus de Git
└── .github/workflows/ # Pipeline CI
```

## Qualité du code

```sh
make ci            # Validation Poetry, lint, formatage et tests unitaires
make test          # Tests seuls, sans téléchargement de modèle
make lint          # Analyse statique avec Ruff
make format-check  # Vérification du formatage
make format        # Application du formatage
```

Le workflow CI reprend les contrôles de qualité et les tests unitaires. Ces tests vérifient le comportement du code ; les notebooks évaluent la qualité des résultats sur les données métier.

## Résultats et limites

Le [rapport d’évaluation](reports/evaluation_report.md) présente les mesures obtenues et les erreurs observées. Le [protocole d’évaluation](reports/evaluation_protocol.md) décrit les critères, les jeux à constituer et les validations à réaliser. Les résultats détaillés sont disponibles dans le [snapshot des métriques](reports/metrics/evaluation_snapshot.json).

Les résultats actuels sont **exploratoires**, sur six documents et six questions. Les seuils d’abstention sont illustratifs, sans calibration sur un jeu indépendant. Une source citée peut rester non pertinente : la fidélité d’un extrait ne suffit pas à garantir une réponse correcte. La détection d’injections est heuristique et ne constitue pas une protection complète.
