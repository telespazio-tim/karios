#!/bin/bash
export PROJ_DATA=~/miniconda3/envs/karios-qa4eo/share/proj
K=~/miniconda3/envs/karios/bin/karios
R=/home/gautier/projects/isl/issues/karios-3.0.0/2025-KARIOS_DEMO
O=${VIDEO_WORK:-work}/demo
run() { n=$1; d=$2; shift; shift; rm -rf $O/$n; (cd $R/$d && $K process "$@" --no-log-file --out $O/$n > $O/$n.log 2>&1; echo "exit=$?" >> $O/$n.log) & }
C=processing_configuration_sans_outliers.json
B=WD/BSG-103-20191005-105550-464915_ortho-ms_8b_rpc_clip_work_masked_work.tif
run 3 3 --conf $C --dem-description "cop DEM" WD/clip_ref_clip.tif WD/clip_work_work.tif - DEM/dem_win.tif
run 4a 4 --conf $C --generate-key-points-mask $B WD/SPOT5_MS_clip.tif
run 4b 4 --conf $C --generate-key-points-mask --enable-large-shift-detection $B WD/SPOT5_MS_clip.tif
run 5 5 --conf processing_configuration_sans_outliers_1C.json --generate-key-points-mask "in/ASAR ASA_IMM_1PNDSI2004030.tif" in/S2C_MSIL1C_20250304T112141_N0511_R037_T28PDU_20250304T133103.tif
wait
