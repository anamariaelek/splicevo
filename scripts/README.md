# SplicEvo

This README provides instructions for using the SplicEvo scripts for data loading, splitting, model training, prediction, and evaluation.

Make sure to have the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo
```

## Memory-Efficient Data Loading and Splitting Approach

The refactored approach processes one genome at a time to minimize memory usage and enables parallel/distributed processing of genomes

### Step 1: data_load.py - Process Individual Genomes

Processes **one genome at a time** and saves its data in a dedicated directory.

**Usage:**
```bash
python ${SPLICEVO_DIR}/scripts/data_load.py \
  --config configs/genomes.json \
  --genome_id human_GRCh37 \
  --output_dir data/processed \
  --window_size 1000 \
  --context_size 450 \
  --alpha_threshold 5 \
  --n_cpus 8 &
```

**What it does:**
1. Loads only the specified genome
2. Extracts sequences, labels, and usage arrays
3. Maps all genes/splice sites to their sequence
4. Saves to `output_dir/genome_id/`:
   - `sequences.mmap` - sequences
   - `labels.mmap` - labels for each sequence
   - `usage_alpha.mmap`, `usage_beta.mmap`, `usage_sse.mmap` - usage arrays
   - `metadata.csv` - links sequences to genes/sites
   - `summary.json` - metadata about the genome data

**Run for each genome:**
```bash
for genome in human_GRCh37 mouse_GRCm38 rat_Rnor_5.0; do
  python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config configs/genomes.json \
    --genome_id $genome \
    --output_dir data/processed \
    --window_size 1000 \
    --context_size 450 \
    --quiet \
    --n_cpus 8 &
done
```

**Benchmarking:**
Data loading for `--window_size 5000 --context_size 450 --n_cpus 4`:

| Genome       | Subset | Chromosomes        | Tissues                  | Timepoints | Time (hh:mm:ss) | Max memory | Sequences |
|--------------|--------|--------------------|--------------------------|------------|-----------------|------------|-----------|
| human_GRCh37 | small  | 11, 19, 20, 21, 22 | Brain, Cerebellum, Heart |1, 5, 10    | 00:24:26        | 11.4 GB    | 18,327    |
| mouse_GRCm38 | small  | 15, 16, 17, 18, 19 | Brain, Cerebellum, Heart |1, 5, 10    | 00:17:02        | 14.8 GB    | 17,963    |
| rat_Rnor_5.0 | small  | 16, 17, 18, 19, 20 | Brain, Cerebellum, Heart |1, 5, 10    | 00:13:47        | 11.4 GB    | 15,153    |
| human_GRCh37 | full   | all                | all                      |all         | 21:32:20        | 557 GB     | 127,990   |
| mouse_GRCm38 | full   | all                | all                      |all         | 12:06:09        | 671 GB     | 104,151   |
| rat_Rnor_5.0 | full   | all                | all                      |all         | 18:06:28        | 586 GB     | 101,460   |


### Step 2: data_split.py - Combine into Train/Test Sets

Reads individual genome directories and combines them into train/test splits based on orthology.

**Usage:**
```bash
SPLICEVO_DIR=/home/elek/projects/splicevo
ORTHOLOGY_FILE=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/ortholog_groups.tsv
python ${SPLICEVO_DIR}/scripts/data_split.py \
  --input_dir data/processed \
  --output_dir data/splits/run1 \
  --orthology_file ${ORTHOLOGY_FILE} \
  --pov_genome human_GRCh37 \
  --test_chromosomes 1 3 5 \
  --quiet &
```

**What it does:**
1. Discovers all genome directories in `input_dir`
2. Loads orthology file
3. Determines train/test split based on POV genome chromosomes
4. Expands split to ortholog groups
5. For each genome:
   - Loads only sequences assigned to train or test
   - Avoids loading entire genome data
6. Combines across genomes and saves to:
   - `output_dir/train/` - training data
   - `output_dir/test/` - test data
   - Each with sequences.mmap, labels.mmap, usage_*.mmap, metadata.csv

**Key features:**

1. **Memory Efficiency**: Only one genome loaded at a time during data_load
2. **Deduplication**: Same sequence saved once per genome, linked to all overlapping genes
3. **Parallel Processing**: Can run data_load.py for different genomes in parallel
4. **Incremental**: Can process additional genomes later without reprocessing existing ones
5. **Flexible Splitting**: Can create multiple train/test splits from the same loaded data

**Benchmarking:**

Data splitting for data loaded with `--window_size 5000 --context_size 450`, and using `--pov_genome mouse_GRCm38` and `--test_chromosomes` as specified below:

| Genomes                                | Subset | Test chr | Time (hh:mm:ss) | Max memory | Train sequences | Test sequences |
|----------------------------------------|--------|----------|-----------------|------------|-----------------|----------------|
| mouse_GRCm38 rat_Rnor_5.0 human_GRCh37 | small  | 15       | 00:04:10        | 390 MB     | 40,027          | 3,619          |
| mouse_GRCm38 rat_Rnor_5.0 human_GRCh37 | full   | 2, 4     | 04:02:03        | 3.81 GB    | 257,403         | 26,868         |
## Directory Structure

```
data/
  processed/
    human_GRCh37/
      sequences.mmap
      labels.mmap
      usage_alpha.mmap
      usage_beta.mmap
      usage_sse.mmap
      metadata.csv
      sequence_mapping.csv
      summary.json
    mouse_GRCm38/
      ...
    rat_Rnor_6.0/
      ...
  splits/
    run1/
      train/
        sequences.mmap
        labels.mmap
        usage_*.mmap
        metadata.csv
      test/
        ...
      summary.json
```
## Training SplicEvo Model

```bash
TRAINING_CONFIG=configs/training_transformer.yaml
DATA_TRAIN_DIR=data/splits/run1/train
MODEL_DIR=models/splicevo_run1

python ${SPLICEVO_DIR}/scripts/splicevo_train.py \
  --config ${TRAINING_CONFIG} \
  --data ${DATA_TRAIN_DIR} \
  --checkpoint-dir ${MODEL_DIR} \
  --quiet

```

To monitor training progress, use:

```bash
tensorboard --logdir=models/splicevo_run1/tensorboard/
``` 

**Note on HPC Submission:**
Run `splicevo_train_submit.sh` to submit the training script to HPC cluster. The resources and training config there are optimized for training the full model (mouse, rat and human genomes) on Helix HPC. Alternativelly, `splicevo_train_submit_small.sh` is submission script with resources and configs 
optimized for training a small SplicEvo model on subset of data.

## Predictions Using SplicEvo model

### Generate Predictions on Test Set

```bash
DATA_TEST_DIR=data/splits/run1/test
PREDICTIONS_DIR=predictions/splicevo_run1

python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
  --checkpoint ${MODEL_DIR}/best_model.pt \
  --test-data ${DATA_TEST_DIR} \
  --output ${PREDICTIONS_DIR} \
  --use-memmap --save-memmap \
  --batch-size 128
```

### Evaluation of Model Performance on Test Set

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_evaluate.py \
  --test-data ${DATA_TEST_DIR} \
  --predictions ${PREDICTIONS_DIR} \
  --output ${PREDICTIONS_DIR}/evaluation/
```

## Attributions

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_attributions.py \
  --model ${MODEL_PATH} \
  --data ${DATA_TEST_DIR} \
  --predictions ${PREDICTIONS_DIR} \
  --window ${WINDOW} \
  --output ${OUTPUT_DIR}
```

By default, this will calculate both splice site attributions and usage attributions across all conditioins, for all sequences in the test set. This may just be too computationally intensive and take a long time.

To specify a subset of sequences to process, set the `--sequences` parameter to either a range (e.g. `0:1000`) or a comma-separated list  (e.g. `0,5,10,100`) of sequence indices in the test set.

To calculate usage attributions shared across all conditions, set the `--share-attributions-across-conditions` flag.

To skip either splice site attributions or usage attributions calculation, set the `--skip-splice-attributions` or `--skip-usage-attributions` flags respectively.

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_attributions.py \
  --model ${MODEL_PATH} \
  --data ${DATA_TEST_DIR} \
  --predictions ${PREDICTIONS_DIR} \
  --sequences "1:1000" \
  --window ${WINDOW} \
  --output ${OUTPUT_DIR} \
  --share-attributions-across-conditions
```