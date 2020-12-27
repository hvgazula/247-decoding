CMD := echo
CMD := python
CMD := sbatch submit1.sh

# EMB_TYPE := glove
EMB_TYPE := bert
# EMB_TYPE := gpt2

SID := 625
MEL := 500
MINF := 30
# setting a very large number for MEL
# will extract all common electrodes across all conversations

create-pickle:
	$(CMD) tfs_pickling.py \
				--subjects $(SID) \
				--max-electrodes $(MEL) \
				--vocab-min-freq $(MINF) \
				--pickle;

upload-pickle: create-pickle
	gsutil -m cp -r $(SID)*.pkl gs://247-podcast-data/247_pickles/

generate-embeddings:
	mkdir -p logs
	$(CMD) tfs_gen_embeddings.py \
				--subject $(SID) \
				--embedding-type $(EMB_TYPE);