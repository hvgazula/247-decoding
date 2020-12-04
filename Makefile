CMD := echo
CMD := python

SID := 625
MEL := 64

tfs-pickle:
	$(CMD) tfs_pickling.py --subjects $(SID) \
				--max-electrodes $(MEL) \
				--pickle;

upload-pickle: tfs-pickle
	gsutil -m cp -r $(SID)*.pkl gs://247-podcast-data/247_pickles/
