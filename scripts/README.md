With the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo

# Where to save the results
export OUT_DIR=/home/elek/projects/splicing/results/
```

## 1. Process data

Modify the script to specify  
- paths to reference genome and annotation files  
- train and test chromosomes  
- paths to splice site usage files.  

This will load and process the data: 1) genomes and transcript annotations for selected species, with train/test chromosome split, and 2) all SpliSER usage values for tissues and timepoints, as inidicated in the config file.

```bash
python ${SPLICEVO_DIR}/scripts/process_data.py \
    --group test \
    --output_dir ${OUT_DIR}/data_processing_brain \
    --n_cpus 8 \
    --quiet &

python ${SPLICEVO_DIR}/scripts/process_data.py \
    --group train \
    --output_dir ${OUT_DIR}/data_processing_brain \
    --n_cpus 16 \
    --quiet &
```

The results will include the following files saved in the specified output directory:

```
memmap_test
memmap_train
metadata_test.csv.gz
metadata_train.csv.gz
usage_info_test.json
usage_info_train.json
usage_summary_test.csv
usage_summary_train.csv
```

## 2. Train Model

Modify the config file to specify parameters of the model to train. Then train the model:

```bash
# The data to train on
DATA_DIR="${OUT_DIR}/data_processing_brain/memmap_train"

# Where to save the model
CHECKPOINT_DIR="${OUT_DIR}/models/brain_sse_"

python ${SPLICEVO_DIR}/scripts/splicevo_train.py \
    --config ${SPLICEVO_DIR}/configs/training_full.yaml \
    --data ${DATA_DIR} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --quiet &
```

Resume training from latest checkpoint automatically:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \ 
    --config ${SPLICEVO_DIR}/configs/training_default.yaml \ 
    --resume auto
```

Resume training from specific checkpoint:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \ 
    --config ${SPLICEVO_DIR}/configs/training_default.yaml \ 
    --resume ${CHECKPOINT_DIR}/checkpoint_epoch_20.pt
```

Tensorboard visualization:

```bash
tensorboard --logdir ${CHECKPOINT_DIR}/tensorboard/
```

## 3. Predict

Predict using a trained model:

```bash
# The data to predict for
DATA_DIR="${OUT_DIR}/data_processing_brain/memmap_test"

# Where the model checkpoint is located
CHECKPOINT_DIR="${OUT_DIR}/models/brain_sse_"

# Where to save the predictions
PREDICTIONS_DIR="${OUT_DIR}/predictions_brain_sse_"

# Predict from memmap directory and save as memmap
python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
    --checkpoint ${CHECKPOINT_DIR}/best_model.pt \
    --normalization-stats ${CHECKPOINT_DIR}/normalization_stats.json \
    --test-data ${DATA_DIR} \
    --output ${PREDICTIONS_DIR} \
    --use-memmap \
    --save-memmap \
    --batch-size 64 \
    --quiet &
```
