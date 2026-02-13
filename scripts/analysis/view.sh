# !/bin/bash

DIR="/root/autodl-tmp/models_v1-1-metrics/sentencepo_math_qwen3_4b_ep3_epsbase0.01_Lppo0_Llen0_cmin0.5_cmax1.5"
mkdir -p $DIR/correct_vs_wrong

# python scripts/analysis/view_sentence_analysis.py \
#   --input $DIR/sentence_analysis \
#   --scatter $DIR/m_vs_delta-clip-1.png \
#   --scatter-correct $DIR/m_vs_delta-clip-correct-1.png \
#   --scatter-wrong $DIR/m_vs_delta-clip-wrong-1.png \
#   --sentence-color-by-clip \
#   --bins 1,2,3,4,5,6,7,8,15,32,64 \
#   --xlim 0,55 \
#   --ylim 0,3.5

python scripts/analysis/view_sentence_analysis.py \
  --input $DIR/sentence_analysis \
  --response-scatter $DIR/resp_m_vs_delta-2.png \
  --response-scatter-correct $DIR/resp_m_vs_delta-correct-2.png \
  --response-scatter-wrong $DIR/resp_m_vs_delta-wrong-2.png \
  --response-color-by-clip \
  --response-bins 1,2,3,4,5,6,7,8,15,32,64 \
  --response-delta-bins 0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.2 \
  --response-xlim 0,55 \
  --response-ylim 0,0.4

# for metric in response_len sentence_count response_ppl response_entropy sentence_stats.ppl_mean sentence_stats.ppl_var sentence_stats.entropy_mean sentence_stats.entropy_var; do
#   python scripts/analysis/view_sentence_analysis.py \
#     --input $DIR/sentence_analysis \
#     --compare-out $DIR/correct_vs_wrong/$metric.png \
#     --compare-metrics $metric
# done