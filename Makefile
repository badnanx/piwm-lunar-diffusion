.RECIPEPREFIX := >
PYTHONPATH := src
TRAIN_DIR := $(HOME)/research/piwm/data/lunar/extracted/lunar/lunartrain
TEST_DIR := $(HOME)/research/piwm/data/lunar/extracted/lunar/lunartest

.PHONY: test-data test-pair debug-vae debug-pair inspect-pair plot-pair clean-pyc

test-data:
>PYTHONPATH=$(PYTHONPATH) python scripts/test_dataset.py \
>  --data_dir $(TRAIN_DIR) \
>  --batch_size 8 \
>  --max_files 5

test-pair:
>PYTHONPATH=$(PYTHONPATH) python scripts/test_pair_dataset.py \
>  --data_dir $(TRAIN_DIR) \
>  --batch_size 8 \
>  --max_files 5

debug-vae:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_autoencoder.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/autoencoder_p1_xytheta_debug \
>  --latent_dim 64 \
>  --batch_size 16 \
>  --epochs 3 \
>  --max_train_files 10 \
>  --max_test_files 5 \
>  --state_weight 0.1 \
>  --state_indices 0 1 4 \
>  --patience 2 \
>  --seed 42

debug-pair:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_piwm_pair_baseline.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/piwm_pair_debug \
>  --latent_dim 64 \
>  --batch_size 16 \
>  --epochs 2 \
>  --max_train_files 10 \
>  --max_test_files 5 \
>  --state_indices 0 1 2 3 4 5 \
>  --p1_weight 0.1 \
>  --p2_weight 0.1 \
>  --dynamics_weight 0.1 \
>  --pred_recon_weight 1.0 \
>  --patience 2 \
>  --seed 42

inspect-pair:
>explorer.exe outputs/piwm_pair_debug

plot-pair:
>PYTHONPATH=$(PYTHONPATH) python scripts/plot_history.py \
>  --history outputs/piwm_pair_debug/history.json \
>  --output_dir outputs/piwm_pair_debug/plots

clean-pyc:
>find . -type d -name "__pycache__" -prune -exec rm -rf {} +

.PHONY: debug-pair-physstrong eval-pair eval-pair-physstrong

debug-pair-physstrong:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_piwm_pair_baseline.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/piwm_pair_debug_physstrong \
>  --latent_dim 64 \
>  --batch_size 16 \
>  --epochs 2 \
>  --max_train_files 10 \
>  --max_test_files 5 \
>  --state_indices 0 1 2 3 4 5 \
>  --p1_weight 1.0 \
>  --p2_weight 0.5 \
>  --dynamics_weight 1.0 \
>  --pred_recon_weight 0.1 \
>  --patience 2 \
>  --seed 42

eval-pair:
>PYTHONPATH=$(PYTHONPATH) python scripts/eval_piwm_pair_baseline.py \
>  --data_dir $(TEST_DIR) \
>  --checkpoint outputs/piwm_pair_debug/best.pt \
>  --output_path outputs/piwm_pair_debug/eval_metrics.json \
>  --latent_dim 64 \
>  --state_indices 0 1 2 3 4 5

eval-pair-physstrong:
>PYTHONPATH=$(PYTHONPATH) python scripts/eval_piwm_pair_baseline.py \
>  --data_dir $(TEST_DIR) \
>  --checkpoint outputs/piwm_pair_debug_physstrong/best.pt \
>  --output_path outputs/piwm_pair_debug_physstrong/eval_metrics.json \
>  --latent_dim 64 \
>  --state_indices 0 1 2 3 4 5

.PHONY: debug-pair-noisy2 eval-pair-noisy2 eval-pair-noisy2-clean

debug-pair-noisy2:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_piwm_pair_baseline.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/piwm_pair_debug_noisy2 \
>  --state_key noisy_states_2 \
>  --latent_dim 64 \
>  --batch_size 16 \
>  --epochs 2 \
>  --max_train_files 10 \
>  --max_test_files 5 \
>  --state_indices 0 1 2 3 4 5 \
>  --p1_weight 1.0 \
>  --p2_weight 0.5 \
>  --dynamics_weight 1.0 \
>  --pred_recon_weight 0.1 \
>  --patience 2 \
>  --seed 42

eval-pair-noisy2:
>PYTHONPATH=$(PYTHONPATH) python scripts/eval_piwm_pair_baseline.py \
>  --data_dir $(TEST_DIR) \
>  --checkpoint outputs/piwm_pair_debug_noisy2/best.pt \
>  --output_path outputs/piwm_pair_debug_noisy2/eval_metrics_noisy2.json \
>  --state_key noisy_states_2 \
>  --latent_dim 64 \
>  --state_indices 0 1 2 3 4 5

eval-pair-noisy2-clean:
>PYTHONPATH=$(PYTHONPATH) python scripts/eval_piwm_pair_baseline.py \
>  --data_dir $(TEST_DIR) \
>  --checkpoint outputs/piwm_pair_debug_noisy2/best.pt \
>  --output_path outputs/piwm_pair_debug_noisy2/eval_metrics_clean.json \
>  --state_key states \
>  --latent_dim 64 \
>  --state_indices 0 1 2 3 4 5

.PHONY: debug-extractor medium-extractor plot-extractor

debug-extractor:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_state_extractor.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/state_extractor_xytheta_debug \
>  --state_key states \
>  --state_indices 0 1 4 \
>  --batch_size 16 \
>  --epochs 3 \
>  --max_train_files 10 \
>  --max_test_files 5 \
>  --patience 2 \
>  --seed 42

medium-extractor:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_state_extractor.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/state_extractor_xytheta_medium \
>  --state_key states \
>  --state_indices 0 1 4 \
>  --batch_size 32 \
>  --epochs 5 \
>  --max_train_files 50 \
>  --max_test_files 10 \
>  --patience 3 \
>  --seed 42

plot-extractor:
>PYTHONPATH=$(PYTHONPATH) python scripts/plot_history.py \
>  --history outputs/state_extractor_xytheta_medium/history.json \
>  --output_dir outputs/state_extractor_xytheta_medium/plots

.PHONY: medium-extractor-visible

medium-extractor-visible:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_state_extractor.py \
>  --train_dir $(TRAIN_DIR) \
>  --test_dir $(TEST_DIR) \
>  --output_dir outputs/state_extractor_xytheta_visible_medium \
>  --state_key states \
>  --state_indices 0 1 4 \
>  --visible_only \
>  --visible_margin 0 \
>  --batch_size 32 \
>  --epochs 5 \
>  --max_train_files 50 \
>  --max_test_files 10 \
>  --patience 3 \
>  --seed 42

.PHONY: export-latents-debug train-correction-debug eval-correction-debug

export-latents-debug:
>PYTHONPATH=$(PYTHONPATH) python scripts/export_latent_transitions.py \
>  --data_dir $(TEST_DIR) \
>  --checkpoint outputs/piwm_pair_debug_physstrong/best.pt \
>  --output_path outputs/piwm_pair_debug_physstrong/latent_transitions_test.npz \
>  --state_key states \
>  --latent_dim 64 \
>  --batch_size 64

train-correction-debug:
>PYTHONPATH=$(PYTHONPATH) python scripts/train_latent_correction.py \
>  --latent_npz outputs/piwm_pair_debug_physstrong/latent_transitions_test.npz \
>  --output_dir outputs/latent_correction_debug \
>  --latent_dim 64 \
>  --hidden_dim 256 \
>  --batch_size 128 \
>  --epochs 20 \
>  --patience 5 \
>  --seed 42

eval-correction-debug:
>PYTHONPATH=$(PYTHONPATH) python scripts/eval_latent_correction.py \
>  --latent_npz outputs/piwm_pair_debug_physstrong/latent_transitions_test.npz \
>  --checkpoint outputs/latent_correction_debug/best.pt \
>  --output_path outputs/latent_correction_debug/eval_metrics.json \
>  --latent_dim 64 \
>  --hidden_dim 256

.PHONY: summarize-piwm-v1

summarize-piwm-v1:
PYTHONPATH=$(PYTHONPATH) python scripts/summarize_piwm_eval.py \
  --metrics outputs/piwm_pair_physstrong_v1/eval_metrics.json
