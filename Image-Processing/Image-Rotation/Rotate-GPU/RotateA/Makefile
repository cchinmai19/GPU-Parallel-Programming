.PHONY 		: all, clean

all			:	imrotateG

imrotateG	:	imrotateG.cu
				nvcc imrotateG.cu -o imrotateG -lopencv_core -lopencv_highgui

clean		:
				rm -f imrotateG
