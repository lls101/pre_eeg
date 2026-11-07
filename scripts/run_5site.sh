#!/usr/bin/env bash
# 5-site pre-registered runs (refine-logs/M0M1_5SITE_REPORT.md). Outputs -> results/5site/{m1,m2,m4}
set -e
cd "$(dirname "$0")"
export PD_SITES=5 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
L=../results/5site
if [ "$1" = cpu ]; then
  python m1_features.py      > $L/features_log.txt 2>&1
  python m4_connectivity.py  > $L/conn_log.txt 2>&1
  echo DONE_CPU
else
  python m1_eegnet.py --ensemble --seeds 0 1 2 > $L/eegnet_ens5_log.txt 2>&1
  python m2_finetune.py LaBraM 0 1 2  > $L/ft_LaBraM_log.txt 2>&1
  python m2_finetune.py CBraMod 0 1 2 > $L/ft_CBraMod_log.txt 2>&1
  echo DONE_GPU
fi
