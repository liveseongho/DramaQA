import glob
import os
import json
import subprocess

import moviepy.editor as mp

config = json.load(open('preprocess_config.json', 'r', encoding='utf-8'))
segmented_video_dir = config['segmented_video_dir']
segmented_audio_dir = config['segmented_audio_dir']

os.makedirs(segmented_audio_dir, exist_ok=True)

for idx, video_path in enumerate(sorted(glob.glob(segmented_video_dir + '*.mp4'))):

    clip = mp.VideoFileClip(video_path)
    video_name = video_path.split('/')[-1]

    audio_path = segmented_audio_dir + video_name.replace('.mp4', '.wav')

    # clip.audio.write_audiofile(audio_path)


    command = 'ffmpeg -i ' + video_path + \
              ' -ab 160k -ac 2 -ar 44100 -vn -af atempo=1/0.96 ' + audio_path
    subprocess.call(command, shell=True)

    print('{:5}: {}'.format(idx, audio_path))

    '''
    Check the shape of output audio:
    import soundfile as sf

    def wav_read(wav_file):
        wav_data, sr = sf.read(wav_file, dtype='int16')
        print('*' * 120)
        print('wav_data.shape:', wav_data.shape)
        print('sr:', sr)
        return wav_data, sr
    
    wav_read(audio_path)
    
    if idx>=0:
        break
    '''