=========================================
K-means Clustering for Image Segmentation
=========================================

Setup
-----
Make sure that you have opencv installed and change the Makefiles so that they point to your version of CUDA compiler (nvcc)

How to Run
----------
	cd Sequential
	make sequential
	./kmeansSeq ../images/pepper.bmp output.bmp 4
	
	cd ../Parallel
	make parallel
	./kmeansPara ../images/pepper.bmp output.bmp 4

How to Test
-----------
	cd test_suite
	./autoTest

