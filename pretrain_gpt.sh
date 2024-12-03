export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16
# CUDA_LAUNCH_BLOCKING=1
GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=experiments/codeparrot-small
TENSORBOARD_LOGS_PATH="experiments/tensorboard"
WANDB_LOGS_PATH="experiments/wandb"
PROJECT_NAME="codeparrot-small"

VOCAB_FILE=/gpfs/public/vl/gjs/model/codeparrot-small/vocab.json
MERGE_FILE=/gpfs/public/vl/gjs/model/codeparrot-small/merges.txt
DATA_PATH=/gpfs/public/vl/gjs/Megatron-LM/data/codeparrot_content_document
LEARNING_RATE=0.0005

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12 
    --seq-length 1024
    --max-position-embeddings 1024
)

TRAINING_ARGS=(
    --micro-batch-size 16 
    --global-batch-size 32 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr $LEARNING_RATE
    --lr-decay-style cosine 
    --min-lr 0.0001
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)
EXP_NAME="$LEARNING_RATE-$NUM_NODES-$GPUS_PER_NODE"
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 1000
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --use-pytorch-profiler
#     --wandb-project $PROJECT_NAME
#     --wandb-exp-name $EXP_NAME
#     --wandb-save-dir $WANDB_LOGS_PATH
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}


# python tools/preprocess_data.py \
#        --input codeparrot_data.json \
#        --output-prefix codeparrot \
#        --vocab-file /gpfs/public/vl/gjs/model/codeparrot-small/vocab.json \
#        --merge-file /gpfs/public/vl/gjs/model/codeparrot-small/merges.txt \
#        --tokenizer-type GPT2BPETokenizer \
#        --json-keys content \
#        --workers 32 \
#        --append-eod