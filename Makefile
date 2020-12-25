CMD := echo
CMD := sbatch submit1.sh
CMD := python

USR := $(shell whoami | head -c 2)

SID := 625
MEL := 500

MINF := 30
# setting a very large number for MEL
# will extract all common electrodes across all conversations

create-pickle:
	$(CMD) tfs_pickling.py --subjects $(SID) \
				--max-electrodes $(MEL) \
				--vocab-min-freq $(MINF) \
				--pickle;

upload-pickle: create-pickle
	gsutil -m cp -r $(SID)*.pkl gs://247-podcast-data/247_pickles/



# Change the `head` argument to run a lag multiple times.
# Each submit.sh runs a model twice by default, so `head -n 5` will give you
# 10 models. Set it to 1 to just run it once.
NL = $(shell expr $(shell echo $(LAGS) | wc -w) - 1)
LAGS := $(shell yes "{-2048..2048..256}" | head -n 1 | tr '\n' ' ')
# LAGS := 0

MODES := prod comp
MODES := prod
MODES := comp

test-ensemble:
	$(CMD) tfsdec_main.py \
	    --signal-pickle data/raw/$(SID)_binned_signal.pkl \
	    --label-pickle data/raw/$(SID)_prod_labels_MWF30.pkl \
	    --lag 0 \
	    --fine-epochs 0 \
	    --ensemble \
	    --model s_$(SID)-m_prod-e_64-u_$(USR)

run-decoding:
	for mode in $(MODES); do \
	    sbatch --array=0-$(NL) submit1.sh tfsdec_main.py \
		--signal-pickle data/raw/$(SID)_binned_signal.pkl \
		--label-pickle data/raw/$(SID)_$${mode}_labels_MWF30.pkl \
		--lags $(LAGS) \
		--ensemble \
		--model s_$(SID)-m_$$mode-e_64-u_$(USR); \
	done

plot:
	python plot.py \
	    -q "model == 's_625-m_prod-e_64-u_zz' and ensemble == True" \
	       "model == 's_625-m_comp-e_64-u_zz' and ensemble == True" \
	    -x lag \
	    -y avg_rocauc_test_w_avg
	rsync -azp results/plots ~/tigress/
