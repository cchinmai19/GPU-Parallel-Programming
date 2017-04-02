.PHONY 		: all, clean

all			:	imrotateGS2

imrotateGS2	:	imrotateGS2.cu
				nvcc imrotateGS2.cu -o imrotateGS2 -lopencv_core -lopencv_highgui

clean		:
				rm -f imrotateGS2