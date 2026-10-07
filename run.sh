#!/bin/bash

# =========================================================
# 通用 nnUNetv2 多模型批量训练脚本
# 适用于 SCI 医学影像分割实验
# =========================================================

set -e

# =========================================================
# GPU 设置
# =========================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3

# =========================================================
# nnUNet 路径
# =========================================================
# export nnUNet_raw="/data/sdb/medical/projects/nnUNet-master/datasets/nnUNet_raw"
# export nnUNet_preprocessed="/data/sdb/medical/projects/nnUNet-master/datasets/nnUNet_preprocessed"
# export nnUNet_results="/data/sdb/medical/projects/nnUNet-master/datasets/nnUNet_results"

# =========================================================
# 线程控制
# =========================================================
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# =========================================================
# 科研可重复性
# =========================================================
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# =========================================================
# 日志目录
# =========================================================
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# =========================================================
# 数据集
# =========================================================
DATASET_ID=305
CONFIG=3d_fullres
FOLD=0
NUM_GPUS=4

# =========================================================
# 所有模型
# =========================================================
TRAINERS=(
    "nnUNetTrainerAttentionUnet"
    "nnUNetTrainerDynUNet"
    # "nnUNetTrainerLMambaMFDSNet"
    "nnUNetTrainerSegFormer"
    "nnUNetTrainerSegMamba"
    "nnUNetTrainerSegResNet"
    "nnUNetTrainerSwinUNETR"
    "nnUNetTrainerUMamba"
    "nnUNetTrainerUNETR"
    "nnUNetTrainerVeloxSeg"
    "nnUNetTrainerVNet"
)

# =========================================================
# 开始训练
# =========================================================
for TRAINER in "${TRAINERS[@]}"
do

    echo "=================================================="
    echo "开始训练: ${TRAINER}"
    echo "=================================================="

    START_TIME=$(date +%s)

    # =====================================================
    # Mamba 系列禁用 compile
    # =====================================================
    if [[ "${TRAINER}" == *"Mamba"* ]]; then
        export nnUNet_compile=false
        echo "[INFO] ${TRAINER} 使用 compile=False"
    else
        unset nnUNet_compile
    fi

    # =====================================================
    # 启动训练
    # =====================================================
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -num_gpus ${NUM_GPUS} \
        2>&1 | tee ${LOG_DIR}/${TRAINER}.log

    END_TIME=$(date +%s)

    DURATION=$((END_TIME - START_TIME))

    echo "=================================================="
    echo "${TRAINER} 训练完成"
    echo "耗时: ${DURATION} 秒"
    echo "=================================================="

done

echo "=================================================="
echo "所有模型训练完成"
echo "=================================================="