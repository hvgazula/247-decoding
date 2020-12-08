CMD := echo
CMD := python

SID := 625
MEL := 500
MINF := 100
# setting a very large number for MEL
# will extract all common electrodes across all conversations

create-pickle:
	$(CMD) tfs_pickling.py --subjects $(SID) \
				--max-electrodes $(MEL) \
				--vocab-min-freq $(MINF) \
				--pickle;

upload-pickle: create-pickle
	gsutil -m cp -r $(SID)*.pkl gs://247-podcast-data/247_pickles/
