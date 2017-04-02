.PHONY 		: all, clean

all			:	imshrunkG

imshrunk	:	imshrunkG.cu
				nvcc imshrunkG.cu -o imshrunkG -lopencv_core -lopencv_highgui

clean		:
				rm -f imshrunkG
