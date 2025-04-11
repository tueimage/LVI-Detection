# Task setting
TASKCFG=finetune_lvi/task_configs/lvi.yaml
DATASETCSV=dataset_csv/LVI/lvi.csv
PRESPLITDIR=dataset_csv/LVI/ # Use the predefined split
ROOTPATH=/home/20215294/Data/LVI/patches_20x/feat/h5_files/
MAX_WSI_SIZE=198656  # Maximum WSI size in pixels for the longer side (width or height).
TILE_SIZE=2048        # Tile size in pixels.
# Model settings
HFMODEL=/home/20215294/.cache/slide_encoder.pth
MODELARCH=gigapath_slide_enc12l768d
TILEEMBEDSIZE=1536
LATENTDIM=768
# Training settings
EPOCH=40
GC=1
LR=0.001
WD=0.01 
LD=0.95
FEATLAYER=12
DROPOUT=0.01
# Output settings
WORKSPACE=outputs/LVI/lvi_finetune
SAVEDIR=$WORKSPACE
EXPNAME=run_blr-${BLR}_wd-${WD}_ld-${LD}_feat-${FEATLAYER}_dropout-${DROPOUT}_not_freeze_aug_fixed

echo "Data directory set to $ROOTPATH"


python finetune_lvi/main.py --task_cfg_path ${TASKCFG} \
    --dataset_csv $DATASETCSV \
    --root_path $ROOTPATH \
    --model_arch $MODELARCH \
    --lr $LR \
    --layer_decay $LD \
    --optim_wd $WD \
    --dropout $DROPOUT \
    --drop_path_rate 0 \
    --val_r 0.1 \
    --epochs $EPOCH \
    --input_dim $TILEEMBEDSIZE \
    --latent_dim $LATENTDIM \
    --feat_layer $FEATLAYER \
    --warmup_epochs 1 \
    --gc $GC \
    --model_select val \
    --lr_scheduler cosine \
    --folds 1 \
    --dataset_csv $DATASETCSV \
    --pre_split_dir $PRESPLITDIR \
    --save_dir $SAVEDIR \
    --pretrained $HFMODEL \
    --report_to tensorboard \
    --exp_name $EXPNAME \
    --max_wsi_size $MAX_WSI_SIZE\
    --tile_size $TILE_SIZE \
