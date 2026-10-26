#!/usr/bin/env bash
set -euo pipefail

SWIFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
: "${CANN_SET_ENV:?Set CANN_SET_ENV to the installed CANN set_env.sh}"
: "${MODEL_DIR:?Set MODEL_DIR to the official Ling3 model directory}"
: "${DATASET:?Set DATASET to the SFT JSONL file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR outside source and backup directories}"
: "${CACHE_ROOT:?Set CACHE_ROOT outside source and backup directories}"
: "${TMPDIR:?Set TMPDIR to a short writable scratch path}"
: "${DEVICE_SET:?Select eight physical devices explicitly}"
: "${MASTER_PORT:?Select an unused port explicitly}"

[[ -f "$CANN_SET_ENV" && -f "$MODEL_DIR/config.json" && -f "$DATASET" ]] || {
  echo 'Missing CANN setup, model config or dataset' >&2; exit 2;
}
IFS=, read -r -a devices <<< "$DEVICE_SET"
[[ ${#devices[@]} == 8 && "$DEVICE_SET" =~ ^[0-9]+(,[0-9]+){7}$ ]] || {
  echo 'DEVICE_SET must contain exactly eight physical device numbers' >&2; exit 2;
}
declare -A seen=()
for device in "${devices[@]}"; do
  [[ ! -v seen[$device] ]] || { echo 'DEVICE_SET contains duplicates' >&2; exit 2; }
  seen[$device]=1
done
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo 'MASTER_PORT must be an integer' >&2; exit 2; }
[[ -d "$TMPDIR" ]] || { echo 'TMPDIR must already exist' >&2; exit 2; }

set +u
source "$CANN_SET_ENV"
set -u
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_SET"
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export HF_MODULES_CACHE="$CACHE_ROOT/hf_modules"
if [[ -n "${FLA_ROOT:-}" ]]; then
  [[ -d "$FLA_ROOT/fla" ]] || { echo 'FLA_ROOT has no fla package' >&2; exit 2; }
  export PYTHONPATH="$SWIFT_ROOT:$FLA_ROOT:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SWIFT_ROOT:${PYTHONPATH:-}"
fi

packing="${PACKING:-false}"
[[ "$packing" == true || "$packing" == false ]] || { echo 'PACKING must be true or false' >&2; exit 2; }
packing_args=(--packing "$packing")
if [[ "$packing" == true ]]; then
  packing_args+=(--packing_length "${PACKING_LENGTH:-2048}" --packing_strategy sequential)
fi
step_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then step_args+=(--max_steps "$MAX_STEPS"); fi

accelerate launch \
  --config_file "$SWIFT_ROOT/examples/ascend/train/ling3/fsdp2.json" \
  --num_processes 8 --main_process_port "$MASTER_PORT" \
  "$SWIFT_ROOT/swift/cli/sft.py" \
  --model "$MODEL_DIR" --template ling3 --attn_impl flash_attn \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 \
  --dataset "$DATASET" \
  --system 'You are a helpful, precise, knowledgeable, and careful AI assistant. Carefully understand the user request, provide accurate and complete answers, explain important reasoning and technical details when necessary, and avoid irrelevant or unsupported content.' \
  --torch_dtype bfloat16 --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-false}" \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_length 2048 --padding_side "${PADDING_SIDE:-right}" --num_train_epochs 2 \
  --eval_strategy no --save_steps 500 --logging_steps 1 \
  --dataloader_num_workers 8 --dataset_num_proc 8 \
  --save_total_limit 2 --save_only_model true \
  --output_dir "$OUTPUT_DIR" "${packing_args[@]}" "${step_args[@]}"
