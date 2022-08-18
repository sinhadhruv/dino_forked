Fork dino repository outside of your repository

Step 1--> create attention maps corresponding to the dive frames

python video_generation.py  --input_path <path of original dive frames> \
    --output_path <output path> resize 256


python3 video_generation.py  --input_path /net/projects/soirov/mounting_root/images/SB0318 --output_path result-SB0318/ --resize 256

The results will be stored in attention directory inside result-SB0318


Step 2--> Get key attention maps using Gaussian Clustering 
Run summarize_video.py (read the argument parsers before running)

Step 3--> Get videos corresponding to key frames 
	Run compile_video.py (read the argument parsers before running)