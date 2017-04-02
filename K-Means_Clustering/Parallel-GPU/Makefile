.PHONY : all,clean

kmeans	:	kmeans.cu 
			nvcc kmeans.cu -o kmeans -lopencv_core -lopencv_highgui
clean	:
			rm -f kmeans
