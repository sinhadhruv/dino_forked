import os
import glob
import m3u8_To_MP4
import moviepy
import shutil
from moviepy.editor import *


def get_videos(path_new, image_names):
    """
    This method stores videos corresponding to each key frame

    Arguments
    --------------------
    path_new: str
            path where we want to store the video corresponding to each key frame
    image_names: list
            list of ids of all the key frames
    """
    os.chdir(path_new)
    i = 0
    for image in image_names:
        a = image.split('/')[-1].split('.')[0]
        url_new = main_url+"/"+a+".m3u8"
        a = m3u8_To_MP4.multithread_download(url_new)
        i+=1

def concatenate_videos(path_new):
    """
    This method concatenates all the shorter videos into a longer video. Each of the shorter 
    videos is of 1 minute but we only extract 4 seconds around the center of the video (29sec-31sec)

    Arguments
    ------------------
    path_new: str
            path where all the shorter videos are stored
    """
    os.chdir(path_new)
    mp4_names = glob.glob(path_new+"/*")
    mp4_names.sort()
    print(mp4_names)

        

    clips = [VideoFileClip(c) for c in mp4_names]
    subclips = [clip.subclip(29,31) for clip in clips]
    final = concatenate_videoclips(subclips)
    

    final.write_videofile("merged.mp4")



if __name__ == '__main__':

    #main_url = "https://streamcache.uc.r.appspot.com/SOI2020/FK200126/SB0318/HD_SIT/SOURCE/MASTER"
    # path = "/home/dhruvs/dino/result-SB0318/"
    # dir = "video_secs/"
    # path_new = os.path.join(path, dir)
    #image_names = glob.glob("result-SB0318/summary_frames_new/*.jpeg")

    parser = argparse.ArgumentParser(description='Inputs to compile video')
    parser.add_argument('--main_url', type=str, required=True, help='url to extract videos corresponding to the key frame')
    parser.add_argument('--path_new', type=str, required=True, help='path to store the extracted videos')
    parser.add_argument('--path_key_frames', type=str, required=True, help='path where key frames are stored')

    image_names = glob.glob(args.path_key_frames+"/*.jpeg")

    if os.path.exists(path_new):
        shutil.rmtree(path_new)
    os.makedirs(path_new)

    get_videos(path_new, image_names)
    concatenate_videos(path_new)


