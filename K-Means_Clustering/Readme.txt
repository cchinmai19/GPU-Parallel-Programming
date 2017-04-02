To execute sequential code:
1. make kmeansC // compile
2. ./kmeansC input_filename.bmp output_filename.bmp num_clusters //run

To execute parallel code:
1. source source_file // loads necessary modules
2. make kmeans // compile
3. salloc -p gpu --gres=gpu:1 srun kmeans input_filename.bmp output_filename.bmp num_clusters & //run

