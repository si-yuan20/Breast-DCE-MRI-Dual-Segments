#!/usr/bin/env bash
# ==============================================================================
# nnunet_train.sh — 按模型名切换的 nnU-Net v2 训练入口
#
# 用法:
#   ./nnunet_train.sh <TrainerName> <dataset> <config> <fold> [额外参数...]
#   ./nnunet_train.sh --list     列出所有可用模型
#   ./nnunet_train.sh --doctor   检查环境变量、可选依赖、各模型可运行性
#
# 示例:
#   ./nnunet_train.sh nnUNetTrainerSwinUNETR 801 3d_fullres 0 -num_gpus 4
#   CUDA_VISIBLE_DEVICES=0,1 ./nnunet_train.sh nnUNetTrainerVNet 305 3d_fullres 1 -num_gpus 2 --npz
#
# 说明:
#   -tr 会自动补上，无需手写；额外参数原样透传给 nnUNetv2_train。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="${REPO_ROOT}/nnunetv2/training/nnUNetTrainer"
LOG_DIR="${REPO_ROOT}/logs"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'

die()  { echo "${RED}[错误]${NC} $*" >&2; exit 1; }
warn() { echo "${YELLOW}[警告]${NC} $*" >&2; }
info() { echo "${BLUE}[信息]${NC} $*"; }
ok()   { echo "${GREEN}[通过]${NC} $*"; }

usage() {
    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# 扫描 trainer 目录得到类名。约定：类名与文件名一致；不一致时取文件里第一个
# nnUNetTrainer* 子类声明（recursive_find_python_class 也是按类名匹配的）。
list_trainers() {
    local f base
    for f in "${TRAINER_DIR}"/nnUNetTrainer*.py; do
        [ -f "$f" ] || continue
        base="$(basename "$f" .py)"
        if grep -qE "^class ${base}\b" "$f"; then
            echo "$base"
        else
            grep -oE '^class nnUNetTrainer[A-Za-z0-9_]*' "$f" | sed 's/^class //' | head -1
        fi
    done
}

# 纯文本匹配，不 import —— 缺依赖时也能正确判定模型是否存在
trainer_exists() {
    grep -rqE "^class $1\b" "${TRAINER_DIR}" --include='*.py' 2>/dev/null
}

# 各模型真正需要的可选依赖（惰性导入后只有跑该模型时才需要它们）
deps_for() {
    case "$1" in
        nnUNetTrainerSegMamba|nnUNetTrainerUMamba)
            echo "monai mamba_ssm" ;;
        nnUNetTrainer|nnUNetTrainerSegFormer|nnUNetTrainerLMambaMFDSNet|nnUNetTrainerBaseChainLMambaMFDS)
            echo "" ;;
        *)
            echo "monai" ;;
    esac
}

pkg_version() {
    python -c "import $1, sys; print(getattr($1, '__version__', 'ok'))" 2>/dev/null
}

# ── --list ────────────────────────────────────────────────────────────────────
cmd_list() {
    echo "可用模型（配合 nnunet_train.sh 作为第一个参数）:"
    echo
    local t deps
    while read -r t; do
        [ -n "$t" ] || continue
        deps="$(deps_for "$t")"
        printf '  %-38s %s\n' "$t" "${deps:-(无额外依赖)}"
    done < <(list_trainers)
    echo
    echo "共 $(list_trainers | wc -l | tr -d ' ') 个。"
}

# ── --doctor ──────────────────────────────────────────────────────────────────
cmd_doctor() {
    command -v python >/dev/null 2>&1 || die "找不到 python"
    command -v nnUNetv2_train >/dev/null 2>&1 \
        || warn "找不到 nnUNetv2_train，请先执行: pip install -e ${REPO_ROOT}"

    echo "=== nnU-Net 路径环境变量 ==="
    local v val
    for v in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
        val="${!v:-}"
        if [ -z "$val" ]; then
            warn "$v 未设置"
        elif [ -d "$val" ]; then
            ok "$v = $val"
        else
            warn "$v = $val （目录不存在）"
        fi
    done

    echo
    echo "=== 可选依赖 ==="
    local pkg ver
    for pkg in monai mamba_ssm; do
        if ver="$(pkg_version "$pkg")"; then
            ok "$pkg $ver"
        else
            warn "$pkg 未安装"
        fi
    done

    echo
    echo "=== 各模型可运行性 ==="
    local t deps miss
    while read -r t; do
        [ -n "$t" ] || continue
        deps="$(deps_for "$t")"
        miss=""
        for pkg in $deps; do
            pkg_version "$pkg" >/dev/null || miss="${miss} ${pkg}"
        done
        if [ -z "$miss" ]; then
            printf '  %s可跑%s  %s\n' "$GREEN" "$NC" "$t"
        else
            printf '  %s缺依赖%s %-38s 缺少:%s\n' "$RED" "$NC" "$t" "$miss"
        fi
    done < <(list_trainers)
}

# ── 训练 ──────────────────────────────────────────────────────────────────────
cmd_train() {
    local trainer="$1"; shift
    if [ $# -lt 3 ]; then
        echo "缺少参数。" >&2
        echo >&2
        usage >&2
        exit 1
    fi
    local dataset="$1" config="$2" fold="$3"; shift 3

    if ! trainer_exists "$trainer"; then
        echo "${RED}[错误]${NC} 未知模型: ${trainer}" >&2
        echo >&2
        echo "可用模型:" >&2
        list_trainers | sed 's/^/    /' >&2
        exit 1
    fi

    command -v nnUNetv2_train >/dev/null 2>&1 \
        || die "找不到 nnUNetv2_train，请先执行: pip install -e ${REPO_ROOT}"

    [ -n "${nnUNet_preprocessed:-}" ] \
        || die "nnUNet_preprocessed 未设置，训练无法开始。
      示例: export nnUNet_preprocessed=/data/sdb/medical/nnunet_data/nnUNet_preprocessed"

    local ds_glob
    if [[ "$dataset" == Dataset* ]]; then
        ds_glob="$dataset"
    elif [[ "$dataset" =~ ^[0-9]+$ ]]; then
        ds_glob="Dataset$(printf '%03d' "$dataset")_*"
    else
        ds_glob="Dataset${dataset}_*"
    fi

    local ds_path
    ds_path="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "${ds_glob}" 2>/dev/null | head -1)"
    [ -n "$ds_path" ] \
        || die "在 ${nnUNet_preprocessed} 下找不到数据集 ${ds_glob}
      该数据集可能尚未预处理，或 nnUNet_preprocessed 指向了错误的目录。"
    [ -f "${ds_path}/nnUNetPlans.json" ] \
        || warn "${ds_path} 下没有 nnUNetPlans.json，训练大概率会失败"

    mkdir -p "${LOG_DIR}"
    local log
    log="${LOG_DIR}/${trainer}_${dataset}_${config}_fold${fold}_$(date +%Y%m%d_%H%M%S).log"

    echo "=================================================="
    info "模型   : ${trainer}"
    info "数据集 : ${dataset} (${ds_path}) / ${config} / fold ${fold}"
    info "GPU    : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置，将用第 0 号卡>}"
    info "日志   : ${log}"
    echo "--------------------------------------------------"
    info "执行   : nnUNetv2_train ${dataset} ${config} ${fold} -tr ${trainer} $*"
    echo "=================================================="

    local status=0
    set +e
    nnUNetv2_train "${dataset}" "${config}" "${fold}" -tr "${trainer}" "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    set -e

    echo "=================================================="
    if [ "$status" -eq 0 ]; then
        ok "训练结束。日志: ${log}"
    else
        # 保留原始退出码，别压成 1 —— 137=OOM kill、130=Ctrl-C 对调度器有意义
        echo "${RED}[错误]${NC} 训练失败（退出码 ${status}）。日志: ${log}" >&2
    fi
    exit "$status"
}

# ── 入口 ──────────────────────────────────────────────────────────────────────
case "${1:-}" in
    ""|-h|--help)  usage ;;
    --list)        cmd_list ;;
    --doctor)      cmd_doctor ;;
    *)             cmd_train "$@" ;;
esac
