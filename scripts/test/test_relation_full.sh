set -x
# sh scripts/test/test_relation_full.sh
PARTITION=priority
JOB_NAME=psg
PORT=${PORT:-$((29500 + $RANDOM % 29))}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

# PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python tools/rel_test_full.py --launcher="slurm" ${PY_ARGS}

PS_TYPE=ips
MODEL_NAME=vanilla

WORK_DIR=work_dirs/relation/rel_${PS_TYPE}_${MODEL_NAME}

PYTHONPATH="$(pwd):$PYTHONPATH" \
# python tools/rel_test_full.py --launcher="none" ${PY_ARGS}
python tools/rel_test_full.py \
    --work-dir ${WORK_DIR}
