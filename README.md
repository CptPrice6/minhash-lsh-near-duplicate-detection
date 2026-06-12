# Near-Duplicate Detection with MinHash and LSH

This project implements a near-duplicate detection pipeline for text documents:

```text
text preprocessing -> word shingles -> MinHash signatures -> LSH banding
```

The system is evaluated on:

- 20 Newsgroups
- Reuters-21578
- a streamed Wikipedia sample for scalability tests

The experiments measure precision, recall, F1-score, candidate reduction, runtime, and the effect of the main parameters:

- shingle size `k`
- number of MinHash values
- hash family
- LSH bands `b`
- rows per band `r`

The repository also includes validation against `datasketch`, theoretical and empirical S-curves, and an optional SimHash experiment.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

The datasets are downloaded automatically on the first run. An internet connection is required.

## Run the main experiments

```powershell
python scripts\run_experiments.py --dataset 20_newsgroups
python scripts\run_experiments.py --dataset reuters
python scripts\generate_plots.py
```

## Run the additional experiments

MinHash validation:

```powershell
python scripts\validate_datasketch.py
```

LSH S-curves:

```powershell
python scripts\generate_s_curve.py --dataset 20_newsgroups
python scripts\generate_s_curve.py --dataset reuters
```

SimHash extension:

```powershell
python scripts\run_simhash_experiment.py
```

Wikipedia scalability benchmark:

```powershell
python scripts\run_wikipedia_scalability.py --repetitions 3
```

The Wikipedia experiment is the slowest because it processes up to 10,000 documents.

## Results

Generated tables are saved in:

```text
results/tables/
```

Generated figures are saved in:

```text
results/figures/
```

The scripts use fixed random seeds so the document samples and candidate sets are reproducible. Runtime measurements may vary slightly between machines.

## Tests

```powershell
ruff format --check .
ruff check .
pytest -q
```

## Project structure

```text
src/near_dup/     core implementation
scripts/          experiment and plotting scripts
tests/            automated tests
results/          generated tables and figures
notebooks/        exploratory analysis and main results
```

## Main limitations

The implementation runs on one machine and keeps shingles and signatures in memory. LSH greatly reduces the number of candidate pairs, but the number of candidates still depends on the dataset and the selected banding parameters.
