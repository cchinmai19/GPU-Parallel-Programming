.PHONY 		: all, clean

all			:	imrotateGS

imrotateGS	:	imrotateGS.cu
				nvcc imrotateGS.cu -o imrotateGS -lopencv_core -lopencv_highgui

clean		:
				rm -f imrotateGS
