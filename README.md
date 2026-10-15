# Social Media Content Influence Study

A research notebook pipeline that describes the **emotional, toxic, and moral content of social-media captions and video transcripts**, then summarizes the scores by participant.

It compares pretrained NLP models, routes disagreements for LLM scoring, and exports per-post and per-participant CSVs. These are measures of content exposure. The code does not establish that exposure caused a change in a participant's emotions or mental health.

## How it works

```text
Caption + transcript → merged text → pretrained model scores
                                             ↓
                              agreement / disagreement routing
                                             ↓
                              selected posts → LLM scoring
                                             ↓
                            final scores and source labels
                                             ↓
                               participant-level summaries
```

The main workflow uses filenames ending in `_905`. The earlier `_709` notebooks are caption-only experiments using a different workbook. These suffixes identify experiment versions, not a guaranteed row count for new data.

## Data availability

**Raw study data, generated CSVs, and credentials are not included.** Running the complete pipeline requires authorized access to the study data or a compatible replacement dataset.

Create the following local layout:

```text
social-media-influence-study/
├── README.md
├── .gitignore
├── .env                      # local API configuration
├── Data/
│   └── data2_done_april1(data2_done_april1).csv
├── Outputs/                  # create before saving results
└── notebooks/                # committed analysis notebooks
```

Run kernels with **`notebooks/` as the working directory** because input/output paths use `../Data` and `../Outputs`. The notebook inside `notebooks/exploration/` also uses `../Outputs`; change it to `../../Outputs` if running from that subdirectory.

## Setup

Saved notebooks report Python 3.12.1. The repository has no pinned requirements file or environment lock, so dependency compatibility must be checked in your environment.

```bash
git clone https://github.com/gautambalgi/social-media-influence-study.git
cd social-media-influence-study
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install the packages used by the notebooks:

```bash
python -m pip install jupyterlab pandas numpy torch transformers detoxify tqdm openpyxl anthropic python-dotenv emoji==0.6.0
```

Create `Data/` and `Outputs/`, provide the input file, then launch:

```bash
cd notebooks
python -m jupyterlab
```

Model steps download external checkpoints. Some notebooks explicitly select CUDA when available; others use pipeline defaults. Reduce batch sizes if memory is limited.

### Input schema

`00_merge_text.ipynb` reads the CSV with `encoding="latin-1"` and expects:

| Column | Use |
|---|---|
| `Your Name` | Renamed to `student_id`; should contain participant pseudonyms |
| `anon_name` | Dropped as redundant; currently expected to exist |
| A column starting with `Q5` | Renamed to `Q5_caption` |
| `Full Transcription` | Video transcript text |

Other survey columns are preserved. The MMADS notebook also compares results against the exact Q9 survey column referenced in its code; adapt that cell for replacement data.

Captions and transcripts are joined with a blank line. Empty/`NA` values and transcript markers such as `NOT A VIDEO` are treated as unavailable. The output adds `text_for_analysis`, `text_source`, and `has_text`.

## Notebook execution order

These notebooks record an evolving experiment rather than one automated runner. Comparison cells can require earlier outputs even when the model inference itself is independent.

| Stage | Notebook | Purpose |
|---|---|---|
| 1 | [`00_merge_text`](notebooks/00_merge_text.ipynb) | Prepare text; export `merged_text_905.csv` |
| 2 | [`03`](notebooks/03_emotion_distilroberta_905.ipynb), [`04`](notebooks/04_emotion_goemotions_905.ipynb), [`05`](notebooks/05_emotion_cardiff_twitter_905.ipynb) | Run original emotion models |
| 3 | [`06`](notebooks/06_toxicity_detoxify_905.ipynb), [`07`](notebooks/07_toxicity_sNLP_905.ipynb), [`08`](notebooks/08_toxicity_cardiff_offensive_905.ipynb) | Run toxicity models in this order; later notebooks compare earlier CSVs |
| 4 | [`09_morality_MMADS`](notebooks/09_morality_MMADS_905.ipynb) | Score moral-foundation virtue/vice dimensions |
| 5 | [`13_routing_pipeline`](notebooks/13_routing_pipeline_905.ipynb) | Combine initial results; export `routed_master_905.csv` |
| 6 | [`14_llm_scoring_yellow`](notebooks/14_llm_scoring_yellow_905.ipynb) | Test, pilot, and checkpointed LLM scoring |
| 7 | [`17_emotion_bertweet`](notebooks/17_emotion_bertweet_905.ipynb) | Generate BERTweet scores; compare with v1 routing/LLM results |
| 8 | [`18_test_new_emotion_models`](notebooks/18_test_new_emotion_models.ipynb) | Probe models, export MichelleJieli/Bhadresh results, and compare scores |
| 9 | [`19_routing_pipeline_v2`](notebooks/19_routing_pipeline_v2_905.ipynb) | Route with Cardiff/BERTweet/Bhadresh; append LLM emotion results for newly disputed posts |
| 10 | [`20_final_consolidation`](notebooks/20_final_consolidation_905.ipynb) | Consolidate scores, aggregate by participant, and append morality summaries |

The `_709` emotion and MFormer notebooks use `Data/Social_Media_Posting_Report_ANONYMIZED.xlsx`. They are historical alternatives, not prerequisites for this sequence.

### Models

| Task | Checkpoints / implementation |
|---|---|
| Initial emotions | `j-hartmann/emotion-english-distilroberta-base`, `SamLowe/roberta-base-go_emotions`, `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest` |
| V2 emotions | Cardiff above, `finiteautomata/bertweet-base-emotion-analysis`, `bhadresh-savani/distilbert-base-uncased-emotion` |
| Toxicity | `Detoxify('original')`, `s-nlp/roberta_toxicity_classifier`, `cardiffnlp/twitter-roberta-base-offensive` |
| Morality | `MMADS/MoralFoundationsClassifier`; earlier experiments use `joshnguyen/mformer-*` |
| LLM scoring | Anthropic SDK; the code sets `MODEL = "claude-sonnet-4-5"` |

V2 aligns emotions to anger, disgust, fear, joy, sadness, and surprise. Bhadresh lacks disgust, so that score is filled with zero. Notebook 18 combines joy/love using their maximum before exporting; notebook 19 consumes the mapped output. These choices affect disagreement and final averages.

## Routing and final scores

| Bucket | Rule in notebooks 13/19 |
|---|---|
| Gray | No usable text, or fewer than 20 stripped characters |
| Green | No disagreement or toxicity escalation rule triggers |
| Yellow | Maximum cross-model emotion range or broad toxicity range exceeds `0.25`; toxicity also routes here when Detoxify insult exceeds `0.5` and s-nlp toxic is below `0.3` |
| Red | Toxicity only: Detoxify threat, identity attack, or severe toxicity exceeds `0.5` |

Notebook 20 uses the following actual behavior:

- Green emotion: mean of the three v2 transformer scores.
- Yellow/red emotion: objective LLM scores at confidence `>= 0.8`; scores at `>= 0.2` are also retained with a medium-confidence review tag. Lower/missing confidence leaves scores missing.
- Green toxicity: Detoxify subtype scores.
- Yellow/red toxicity: mapped objective LLM scores at those confidence thresholds, otherwise a Detoxify fallback.
- Gray: final emotion/toxicity scores remain missing.

Medium-confidence scores remain in numeric outputs and aggregates; a tag does not implement a separate review queue. Youth-perspective prompts are model-generated perspectives, not measured responses from young participants, and do not determine consolidated scores.

## API configuration

For notebooks 14/19, create a local `.env` in the repository root:

```dotenv
ANTHROPIC_API_KEY=your_key_here
```

Verify account access to the configured model. These cells send post text to an external API and incur charges. Cost calculations in the notebooks are historical estimates, not current price guarantees.

Notebook 14 prints a prefix of the API key; remove that diagnostic when adapting it or sharing outputs. Keep credentials and participant text out of commits, and review saved notebook outputs before redistributing them.

## Generated outputs

| File in `Outputs/` | Contents |
|---|---|
| `merged_text_905.csv` | Input rows with merged text and source flags |
| `emotion_*.csv`, `toxicity_*.csv`, `morality_MMADS_905.csv` | Individual model scores |
| `routed_master_905.csv`, `routed_master_v2_905.csv` | Mapped scores and routing decisions |
| `llm_emotion_*.csv`, `llm_toxicity_*.csv` | Objective/youth scores, confidence, reasoning, and identifiers |
| `final_per_post_905.csv` | Final scores, source tags, model detail, and morality scores after the final merge |
| `final_per_student_905.csv` | Participant exposure summaries plus morality summaries |

Aggregates include means, standard deviations, proportions with scores `>= 0.5`, and conditional means over high-scoring posts. Emotion summaries also include minima/maxima. Morality uses its own `has_text` filter, so included posts can differ from the emotion/toxicity subset. Cross-model standard deviation measures disagreement; it is not a statistical confidence interval.

## Reproducibility notes

- **Missing files:** provide data, create output folders, and follow the dependency sequence.
- **Row alignment:** several joins depend on unchanged row indices. Preserve ordering and check `_post_index` uniqueness before consolidation.
- **Repeated v2 runs:** notebook 19 appends LLM results without checkpoint deduplication. Re-running append cells can duplicate IDs and multiply rows in later merges. Check duplicates before consolidating.
- **Failed API calls:** notebook 14 checkpoints failed rows and resumes by ID. Inspect `_failed` entries; a resume does not ensure retrying them.
- **Model differences:** scores from different domains and label sets are not automatically calibrated. Zero-filled labels, character clipping, and token truncation affect comparisons.
- **Interpretation:** irony, slang, quotation, and text describing emotions can confuse classifiers. Visual/audio context absent from transcripts is not analysed. Human annotation and independent validation are needed for accuracy or personal-impact claims.

The repository provides research code and saved observations, with no packaged app, automated tests, or reproducible data bundle. Regenerate counts and agreement statistics for the actual data used.
