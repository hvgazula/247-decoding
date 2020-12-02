CMD := echo
CMD := python

SID := 625
MEL := 55

tfs-pickle:
	$(CMD) tfs_pickling.py --subjects $(SID) \
				--max-electrodes $(MEL) \
				--pickle;