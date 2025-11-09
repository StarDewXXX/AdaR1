#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ATTENTION_BACKEND=XFORMERS

# 默认参数设置
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# 可选的数据集类型：aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"       # 默认输出目录
N_SAMPLES=1              # 默认 n_samples 值
RESPONSE_LENGTH=12288    # 默认 response_length 值

# 解析传入的命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            shift
            DATATYPES=()
            # 当遇到下一个以 "--" 开头的参数时退出循环
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --response_length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory> [--n_samples <n_samples>] [--response_length <response_length>]"
            exit 1
            ;;
    esac
done

# 输出解析的参数，便于检查
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "n_samples: ${N_SAMPLES}"
echo "response_length: ${RESPONSE_LENGTH}"

# 对每个 dataset 类型循环执行生成任务
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=./processed_data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.json \
        data.n_samples=${N_SAMPLES} \
        data.batch_size=12288 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=${RESPONSE_LENGTH} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.85 \
        rollout.tensor_model_parallel_size=1 \
        +data.skip_format_reward=True
done
