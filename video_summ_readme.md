Fork the dino repository outside of ROV dive processing repository

***Step 1--> create attention maps corresponding to the dive frames***

All the dive images are stores in soirov, he path of which is given in the example below. We can 
generate attention maps for each dive. The result directory will be created in the current working 
directory, usually wehre video_generation.py is located. The name of the result directory will 
given by the user. In the example below, the name of the result directory is result- SB0318

python3 video_generation.py  --input_path <path of original dive frames> \
    --output_path <output path> resize 256


python3 video_generation.py  --input_path /net/projects/soirov/mounting_root/images/SB0318 --output_path result-SB0318/ --resize 256

The results will be stored in attention directory inside result-SB0318

***Step 2--> Get key attention maps using Gaussian Clustering***
Run summarize_video.py
read arugument parsers and refer to the example given in the script


***Step 3--> Get videos corresponding to key frames***
Run compile_video.py (read the argument parsers before running)
read arugument parsers and refer to the example given in the script