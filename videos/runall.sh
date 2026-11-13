#!/bin/bash
export PROJ_DATA=~/miniconda3/envs/karios-qa4eo/share/proj
K=~/miniconda3/envs/karios/bin/karios
R=/home/gautier/projects/isl/issues/karios-3.0.0/2025-KARIOS_DEMO
O=${VIDEO_WORK:-work}/demo
run() { n=$1; shift; (cd $R/$n && $K process "$@" --no-log-file > $O/$n.log 2>&1; echo "exit=$?" >> $O/$n.log) & }
C=processing_configuration_sans_outliers.json
rm -rf $O/*/ 
run 1 --conf $C -gip --out $O/1 WD/L2H_T34RGT_20171020T085527_LS8_R181_B04_30m_clip.tif WD/LC81810402017293MTI00_B5_work.tif
run 2 --conf $C --generate-key-points-mask --out $O/2 WD/S2A_T30SYJ_20241105T105231_B04_clip.tif WD/S2C_T30SYJ_20241105T105251_B04_work.tif
run 3 --conf $C --out $O/3 --dem-file-path DEM/dem_win.tif --dem-description "cop DEM" WD/clip_ref_clip.tif WD/clip_work_work.tif
run 4a --conf $C --generate-key-points-mask --out $O/4a WD/BSG-103-20191005-105550-464915_ortho-ms_8b_rpc_clip_work_masked_work.tif WD/SPOT5_MS_clip.tif
run 4b --conf $C --generate-key-points-mask --enable-large-shift-detection --out $O/4b WD/BSG-103-20191005-105550-464915_ortho-ms_8b_rpc_clip_work_masked_work.tif WD/SPOT5_MS_clip.tif
run 6 --conf $C --out $O/6 WD/3122863_3423606_2020-02-10_2257_BGREN_Analytic_band2_clip.tif WD/3122863_3423606_2020-02-10_2257_BGREN_Analytic_band5_work.tif
wait
