.PHONY : all, clean

all		:	hogfeature

hogfeature		:main.cu HogSupport.cu HogSupport.h
				nvcc main.cu HogSupport.cu -o hogfeature -lopencv_core -lcublas -lopencv_highgui

clean		:
			rm -f hogfeature
