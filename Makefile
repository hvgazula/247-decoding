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


NL = $(shell expr $(words $(LAGS)) - 1)
LAGS := $(shell echo {-2048..2048..256})
LAGS := 0

MODES := prod comp
MODES := comp
MODES := prod

run-decoding:
	for lag in $(LAGS); do \
	    $(CMD) tfsdec_main.py \
		--subject $(SID) \
		--lag $$lag \
		--fine-epochs 1 \
		--model test-s_$(SID)-e_64-u_$(USR)-l_$$lag; \
	done

run-decoding2:
	for mode in $(MODES); do \
	    sbatch --array=0-$(NL) submit1.sh tfsdec_main.py \
		--signal-pickle $(SID)_binned_signal.pkl \
		--label-pickle $(SID)_$${mode}_labels_MWF30.pkl \
		--lags $(LAGS) \
		--model s_$(SID)-m_$$mode-e_64-u_$(USR); \
	done

plot:
	python plots.py \
	    -q "model == 's_625-m_prod-e_64-u_zz'" \
	       "model == 's_625-m_comp-e_64-u_zz'" \
	    -x lag \
	    -y avg_rocauc_test_w_avg
	mv -f out.png ~/tigress/
