import datetime
import json
import os
import time

from moviepy.editor import VideoFileClip

config = json.load(open('preprocess_config.json', 'r', encoding='utf-8'))
root_dir = config['root_dir']
original_video_dir = config['original_video_dir']
segmented_video_dir = config['segmented_video_dir']
shot_info_dir = config['shot_info_dir']

time_info = dict()

for i in range(1, 18+1):
    episode_number = 'AnotherMissOh{:02}'.format(i)
    # print('episode_number:', episode_number)
    shot_file_name = episode_number + '_Shotinfo_v2.json'
    time_info[episode_number] = json.load(open(shot_info_dir + shot_file_name))['shot_results']


origin_videos = [f for f in os.listdir(original_video_dir) if os.path.isfile(os.path.join(original_video_dir, f)) and '.mp4' in f]
origin_videos.sort()

video_epiclips = {video[:-4]: VideoFileClip(os.path.join(original_video_dir, video)) for video in origin_videos}

for key, clips in video_epiclips.items():
    print(key, ':', clips.duration)

os.makedirs(segmented_video_dir, exist_ok=True)


vid_shots_dict = dict()
#------------------------------------------------------------------------------------------------#
# 비디오가 필요한 vid list를 포함한 파일에 맞게 수정해야 함. vid와 shot_contained 정보가 필요.
# vid_shots_dict[vid] = shot_contained 꼴로 저장. shot_contained는 길이 1 또는 2의 int list
# e.g. vid_shots_dict['AnotherMissOh09_001_0000'] = [56, 81]

QA_filelist = [
    'AnotherMissOhQA_train_set.json',
    'AnotherMissOhQA_val_set.json',
    'AnotherMissOhQA_test_set.json',
]


"""이 부분은 임시"""

old_set = set()
cur_set = set()

with open(os.path.join(root_dir, 'AnotherMissOh_QA/', 'AnotherMissOhQA_total_set_old.json')) as f:
    qa_datum = json.load(f)
    
    for qa_data in qa_datum:
        vid = qa_data['vid']
        shots = qa_data['shot_contained']
        # vid_shots_dict_old[vid] = shots
        old_set.add(vid)

print('len(old_set):', len(old_set))
"""여기까지 임시"""



for filename in QA_filelist:
    
    with open(os.path.join(root_dir, 'AnotherMissOh_QA/', filename)) as f:
        qa_datum = json.load(f)
    
        for qa_data in qa_datum:
            vid = qa_data['vid']
            if vid in old_set:
                continue
            shots = qa_data['shot_contained']
            vid_shots_dict[vid] = shots
            cur_set.add(vid)

print('len(cur_set):', len(cur_set))

print('len(old_set - cur_set):', len(old_set - cur_set))
print('len(cur_set - old_set):', len(cur_set - old_set))

# 여기까지 수정
#------------------------------------------------------------------------------------------------#
        
print('Total vids number:', len(vid_shots_dict))
# print('\n', vid_shots_dict)
# import sys;   sys.exit(0)

def time_to_float(t, fps, start=False):
    x = time.strptime(t.split(';')[0],'%H:%M:%S')
    if start:
        y = (int(t.split(';')[1]) + 1)/fps
    else:
        y = int(t.split(';')[1])/fps
    return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds() + y


for idx, (vid, shots) in enumerate(sorted(vid_shots_dict.items())):
    episode_number = vid[:15]
    video_epiclip = video_epiclips[episode_number]
    video_fps = video_epiclip.fps

    print(idx, ':', vid, '\tfps=', video_fps, '\tshots:', shots)

    shot_number = shots[0]
    st_et = time_info[episode_number][shot_number - 1]
    st = time_to_float(st_et['start_time'], fps=video_fps, start=True)
    et = time_to_float(st_et['end_time'], fps=video_fps, start=False)


    if len(shots) == 2:
        shot_number = shots[1]
        st_et2 = time_info[episode_number][shot_number - 1]
        et = time_to_float(st_et2['end_time'], fps=video_fps, start=False)

    if et - st <= 1.1:
        et = st + 1.1
        
    print('time: {} ~ {} \t frame: {} {} \t duration: {}'
          .format(st_et['start_time'],
                  st_et['end_time'] if len(shots) == 1 else st_et2['end_time'],
                  st, et,
                  str(datetime.timedelta(seconds=et-st))))


    video_sceneclip = video_epiclip.subclip(st, et)
    video_sceneclip.write_videofile(segmented_video_dir + "%s.mp4" % vid, audio_codec="aac")
    
else:
    print('{} segmented video clips created'.format(idx+1))
