With the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo

# Where to save the results
export OUT_DIR=/home/elek/projects/splicing/results/
export CHECKPOINT_DIR="brain"
```

Modify the script:  
- paths to reference genome and annotation files  
- train and test chromosomes  
- paths to splice site usage files.  

Prepare the data:
```bash
python ${SPLICEVO_DIR}/scripts/process_data.py \
    --group test \
    --output_dir ${OUT_DIR}/data_processing \
    --n_cpus 8 \
    > ${SPLICEVO_DIR}/logs/splicevo_process_data_test.log 2>&1 &

python ${SPLICEVO_DIR}/scripts/process_data.py \
    --group train \
    --output_dir ${OUT_DIR}/data_processing \
    --n_cpus 16 \
    > ${SPLICEVO_DIR}/logs/splicevo_process_data_train.log 2>&1 &
```

Modify the config file:
- path to processed training  
- output directory to save the model to.  

Train the model:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \
    --config ${SPLICEVO_DIR}/configs/training_full.yaml \
    --checkpoint-dir ${OUT_DIR}/models/${CHECKPOINT_DIR}
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
    --resume ${OUT_DIR}/models/${CHECKPOINT_DIR}/checkpoint_epoch_20.pt
```

Tensorboard visualization:

```bash
tensorboard --logdir ${OUT_DIR}/models/${CHECKPOINT_DIR}/tensorboard/
```

Predict using a trained model:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_predict.py \
    --checkpoint ${OUT_DIR}/models/${CHECKPOINT_DIR}/best_model.pt \
    --test-data ${OUT_DIR}/data_processing/processed_data_test.npz \
    --normalization-stats ${OUT_DIR}/models/${CHECKPOINT_DIR}/normalization_stats.json \
    --output ${OUT_DIR}/predictions/test_predictions.npz
```
