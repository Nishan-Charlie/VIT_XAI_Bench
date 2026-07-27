conda run -n mri-diffuser python scripts/scaled_eval.py hilrp vit_base_patch16_224 200 imagenets > results_low_models.txt 2>&1
conda run -n mri-diffuser python scripts/scaled_eval.py hilrp deit_base_patch16_224 200 imagenets >> results_low_models.txt 2>&1
conda run -n mri-diffuser python scripts/scaled_eval.py hilrp mobilevit_s 200 imagenets >> results_low_models.txt 2>&1
echo "DONE" >> results_low_models.txt
