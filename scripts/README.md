With the following variables set:

```bash
# Where SplicEvo is located
export SPLICEVO_DIR=/home/elek/projects/splicevo
# Where to save the results
export OUT_DIR=/home/elek/projects/splicing/results/
export CHECKPOINT_DIR="resnet"
```

Train the model:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_train.py \ 
    --config ${SPLICEVO_DIR}/configs/training_default.yaml
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
    --resume ${OUT_DIR}/${CHECKPOINT_DIR}/checkpoint_epoch_20.pt
```

Predict using a trained model:

```bash
python ${SPLICEVO_DIR}/scripts/splicevo_predict.py --checkpoint ${OUT_DIR}/models/${CHECKPOINT_DIR}/best_model.pt --test-data ${OUT_DIR}/data_processing/processed_data_test.npz --normalization-stats ${OUT_DIR}/models/${CHECKPOINT_DIR}normalization_stats.json --output ${OUT_DIR}/predictions/test_predictions_resnet.npz
```