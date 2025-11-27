# SplicEvo Scripts

Make sure to have the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo

# Where to save the results
export OUT_DIR=/home/elek/projects/splicing/results/
```

## 1. Load all genome data

Load splice site data from all genomes and save as intermediate files.  

This step:
- Loads genomes and transcript annotations for all species
- Loads all SpliSER usage values for tissues and timepoints
- Saves intermediate data that can be reused for different train/test splits

You need to have a config file (e.g. `${SPLICEVO_DIR}/configs/data_human_mouse.json`) specifying the genomes and usage files to load. Note also that config specifies `orthology_file` which will be used in the next step. Then run:

```bash
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_human_mouse_rat.json \
    --output_dir ${OUT_DIR}/data/load/hsap_mmus_rnor \
    --n_cpus 16 \
    --quiet &
```

To add additional genomes or usage files, run the script again with new or updated config file and `--append` flag. For example, to add rat genome and usage data:

```bash
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_rat.json \
    --output_dir ${OUT_DIR}/data/load/hsap_mmus_rnor \
    --n_cpus 16 \
    --append \
    --quiet &
```

To **overwrite** existing genomes:

```bash
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_human_mouse.json \
    --output_dir ${OUT_DIR}/data/load/hsap_mmus \
    --n_cpus 16 \
    --overwrite human_GRCh37 \
    --quiet &
```

The `--overwrite` flag accepts one or more genome IDs. These genomes will be:
- Removed from existing loaded data
- Re-added with current config settings
- Re-loaded from GTF and usage files

**Note:** The `--overwrite` flag automatically loads the existing state, so you don't need to specify `--append` when overwriting.

**Output files:**

```
data/load/
├── loader_state.pkl          # Serialized loader with all splice sites
├── splice_sites.csv.gz       # DataFrame of all splice sites
├── species_info.json         # Species mapping information
├── conditions_info.json      # Usage conditions metadata
└── usage_summary.csv         # Usage statistics summary
```

### Config files details

```json
{
  "orthology_file": "/path/to/ortholog_groups.tsv",
  "genomes": [
    {
      "genome_id": "human_GRCh37",
      "genome_path": "/path/to/Homo_sapiens.fa.gz",
      "gtf_path": "/path/to/Homo_sapiens.gtf.gz",
      "chromosomes": ["1", "2", "3", ...],
      "metadata": {
        "species": "homo_sapiens",
        "assembly": "GRCh37"
      },
      "common_name": "human"
    }
  ],
  "usage_files": {
    "human_GRCh37": {
      "pattern": "/path/to/Human.{tissue}.{timepoint}.combined.tsv",
      "tissues": ["Brain", "Cerebellum", ...],
      "timepoints": [1, 2, 3, ...]
    }
  }
}
```

**Config fields:**
- `genome_id`: Unique identifier for the genome
- `genome_path`: Path to FASTA file
- `gtf_path`: Path to GTF annotation file
- `chromosomes`: List of chromosomes to load (optional, loads all if not specified)
- `metadata.species`: Scientific species name (e.g., "homo_sapiens")
- `metadata.assembly`: Genome assembly version
- `common_name`: Common species name used for species embeddings (e.g., "human", "mouse")

**Important notes:**

1. **Always use the same orthology file** across runs when using `--append` to ensure consistent ortholog group naming
2. **Genome IDs must match exactly** between config files and `--overwrite` arguments
3. **Usage data is preserved** when appending new genomes
4. **Order matters**: Load your primary genome(s) first, then append additional genomes

**Troubleshooting:**

If you notice missing genomes after running with `--append`:
- Check that genome IDs match exactly in your config files
- Verify the log file shows "Loaded existing data with X splice sites from Y genomes"
- Confirm usage files were found (warnings appear for missing files)

### Small test example

Load only a few chrs and usage files each:

```bash
# Initial load of human and mouse
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_human_mouse_small.json \
    --output_dir ${OUT_DIR}/data/load/small/ \
    --n_cpus 8 \
    --quiet &

# Append rat to existing data
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_rat_small.json \
    --output_dir ${OUT_DIR}/data/load/small/ \
    --n_cpus 8 \
    --append \
    --quiet &

# Overwrite genome
python ${SPLICEVO_DIR}/scripts/data_load.py \
    --config ${SPLICEVO_DIR}/configs/data_human_mouse_small.json \
    --output_dir ${OUT_DIR}/data/load/small/ \
    --n_cpus 8 \
    --overwrite human_GRCh37 \
    --quiet &
```

## 2. Split data into train and test sets

Split the loaded data into training and test sets. This step:
- Identifies genes on test chromosomes in the selected POV genome
- Finds orthologous genes across all species (using `orthology_file` specified in config)
- Creates train/test splits respecting orthology
    - Genes on test chromosomes in the specified POV genome are assigned to test set, and all other genes are used for training.
    - All other genomes are split so that all orhologs of test genes are included in the test set, and the remaining genes are used for training.
- Converts to windowed sequences and saves as memmap
- Uses memory-efficient genome-by-genome chunked processing

To generate the train/test split, run:

```bash
python ${SPLICEVO_DIR}/scripts/data_split.py \
    --input_dir ${OUT_DIR}/data/load/hsap_mmus \
    --output_dir ${OUT_DIR}/data/split/hsap_mmus \
    --n_cpus 2 \
    --pov_genome human_GRCh37 \
    --test_chromosomes 1 3 5 \
    --window_size 1000 \
    --context_size 450 \
    --alpha_threshold 5 \
    --chunk-size 1000 \
    --quiet &
```

**Note:** The script uses memory-efficient chunked processing by default:
- Processes one genome at a time
- Splits genes into chunks (default: 2000 genes per chunk)
- Uses conservative worker count (default: 2) to avoid OOM
- If still experiencing OOM, try: `--n_cpus 1` and/or `--chunk-size 1000`

Monitoring:
- Check memory usage: `watch -n 1 free -h`
- Monitor the process: `ps aux | grep data_split` and `top -p <PID>`
- Check exit code: `echo $?` (137 typically means killed by OOM)

Small test example:

```bash
python ${SPLICEVO_DIR}/scripts/data_split.py \
    --input_dir ${OUT_DIR}/data/load/small \
    --output_dir ${OUT_DIR}/data/split/small \
    --n_cpus 2 \
    --pov_genome human_GRCh37 \
    --test_chromosomes 21 \
    --window_size 1000 \
    --context_size 450 \
    --alpha_threshold 5 \
    --chunk-size 1000 \
    --quiet &
```

Arguments:

`--pov_genome`: Point-of-view genome for defining test chromosomes (default: human_GRCh37)  
`--test_chromosomes`: Chromosomes to use for test set in POV genome (default: 1 3 5)  
`--window_size`: Size of central window containing splice sites (default: 1000)  
`--context_size`: Size of context on each side of window (default: 450)  
`--alpha_threshold`: Minimum alpha value; lower values set to 0 (default: 5)  
`--n_cpus`: Number of parallel workers (default: 2, clamped to 1-2 in chunked mode)  
`--chunk-size`: Genes per chunk (default: 2000; reduce if OOM occurs)  

**Output files:**

```
data/split/
├── memmap_train/
│   ├── sequences.mmap
│   ├── labels.mmap
│   ├── species_ids.mmap
│   ├── usage_sse.mmap
│   └── metadata.json
├── memmap_test/
│   ├── sequences.mmap
│   ├── labels.mmap
│   ├── species_ids.mmap
│   ├── usage_sse.mmap
│   └── metadata.json
├── metadata_train.csv.gz
├── metadata_test.csv.gz
├── species_info_train.json
├── species_info_test.json
├── usage_info_train.json
├── usage_info_test.json
├── usage_summary_train.csv
├── usage_summary_test.csv
├── train_genes.txt
├── test_genes.txt
└── split_info.json
```

## 3. Train the model

Modify the config file to specify parameters of the model to train. Then train the model:

```bash
# The data to train on
DATA_TRAIN_DIR="${OUT_DIR}/data/split/memmap_train"

# Where to save the model
MODEL_DIR="${OUT_DIR}/models/resnet_hybridloss"

python ${SPLICEVO_DIR}/scripts/splicevo_train.py \
    --config ${SPLICEVO_DIR}/configs/training_resnet.yaml \
    --data ${DATA_TRAIN_DIR} \
    --checkpoint-dir ${MODEL_DIR} \
    --quiet &
```

Resume training from latest checkpoint automatically:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \ 
    --config ${SPLICEVO_DIR}/configs/training_resnet.yaml \ 
    --resume auto
```

Resume training from specific checkpoint:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \ 
    --config ${SPLICEVO_DIR}/configs/training_resnet.yaml \ 
    --resume ${MODEL_DIR}/checkpoint_epoch_20.pt
```

Tensorboard visualization:

```bash
tensorboard --logdir ${MODEL_DIR}/tensorboard/
```

For tensorboard visualizations, I use the following colors:

Loss train #12b5cb
Loss val #e52592
Loss train ones #425066
Loss train middle #12b5cb
Loss train zeros #96eaf5
Loss val ones #970056
Loss val middle #e52592
Loss val zeros #ffb6df

Plot diagnostic plots

```bash
python ${SPLICEVO_DIR}/scripts/plot_diagnostics.py \
    --checkpoint-dir ${MODEL_DIR} \
    --output-dir ${MODEL_DIR}/diagnostics/ \
    --quiet &
```

## 4. Predict

Predict using a trained model (no normalization-stats needed for SSE):

```bash
# The data to predict for
DATA_TEST_DIR="${OUT_DIR}/data/split/memmap_test"

# Where the model checkpoint is located
MODEL_DIR="${OUT_DIR}/models/resnet_hybridloss"

# Where to save the predictions
PREDICTIONS_DIR="${OUT_DIR}/predictions/resnet_hybridloss"

# Predict from memmap directory and save as memmap
python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
    --checkpoint ${MODEL_DIR}/best_model.pt \
    --test-data ${DATA_TEST_DIR} \
    --output ${PREDICTIONS_DIR} \
    --use-memmap \
    --save-memmap \
    --batch-size 64 \
    --quiet &
```
