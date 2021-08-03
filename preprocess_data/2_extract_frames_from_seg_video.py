import glob
import os
import json

import cv2

config = json.load(open('preprocess_config.json', 'r', encoding='utf-8'))
segmented_video_dir = config['segmented_video_dir']
frame_dir = config['frame_dir']

for idx, video_path in enumerate(sorted(glob.glob(segmented_video_dir + '*.mp4'))):

    print('video_path:', video_path)
    # image_path = video_path.replace('segmented_video', 'segmented_image')
    mp4name = video_path.split('/')[-1]
    episode = mp4name[13:15]
    scene   = mp4name[16:19]
    shot    = mp4name[20:24]

    vidcap = cv2.VideoCapture(video_path)
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    seconds = frames // fps
    
    start_seconds = seconds if (frames - fps * seconds) <= 8 else seconds + 1
    print('frames: {}\t fps: {}\t seconds: {}\tstart_seconds={}'.format(frames, fps, seconds, start_seconds))
    
    extract_list = []
    for s in range(start_seconds):
        extract_list.extend(list(range(s*fps + 1, s*fps + 8+1)))
    # print('extract_list:', extract_list)
    
    # extract_list = list(map(lambda x: int(x), np.linspace(1, frames, number)))

    count = 0

    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if frame is None:
            break
        # print('frame.shape:', frame.shape)
        frame_number = int(vidcap.get(1))

        if frame_number in extract_list:
            root_path = frame_dir + 'AnotherMissOh{}/{}/{}/'.format(episode, scene, shot)
            os.makedirs(root_path, exist_ok=True)
            path = root_path + 'IMAGE_{:010}.jpg'.format(frame_number)
            cv2.imwrite(path, frame)

            count += 1

    vidcap.release()
    

    print('{}, {}: {} frames saved'.format(idx, mp4name, count))
    # if idx>10:
    #     break


