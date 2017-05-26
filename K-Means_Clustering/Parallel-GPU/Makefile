CC = /usr/local/cuda-8.0/bin/nvcc

parallel	:	kmeans.cu
			$(CC) -x cu -arch=sm_30 kmeans.cu -o kmeansPara -lopencv_core -lopencv_highgui
clean	:
			rm -f kmeansPara
images_disappear	:
			rm -f *.jpg *.png *.bmp
