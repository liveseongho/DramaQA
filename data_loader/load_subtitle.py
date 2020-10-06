import os
import json
from pathlib import Path
from unidecode import unidecode

from tqdm import tqdm
from utils import read_json, write_json

# debug
from pprint import pprint

empty_sub = '.'

def load_subtitle(subtitle_path):
    subtitles = {}
    speakers = {}
    subtitles = read_json(subtitle_path)

    for vid, v in subtitles.items():
        for sub in subtitles[vid]["contained_subs"]:
            sub["utter"] = unidecode(sub["utter"])

    return subtitles

def merge_qa_subtitle(new_path, qa_path, subtitle_path):
    print("Processing subtitle data")

    subtitles = load_subtitle(subtitle_path)
    qa = read_json(qa_path)

    res = []
    for row in tqdm(qa):
        if row['vid'].endswith('_0000'):
            # scene question
            vid = row['vid']
            vid_prefix = vid[:vid.find('_0000')]

            shot_subtitles = []
            shot_st = []
            shot_et = []
            subtitle_sts = set()

            # if vid starts with vid_prefix,
            # add sub to shot_subtitles if the same sub has not been added yet
            for vid, subs in subtitles.items():
                if not vid.startswith(vid_prefix):
                    continue

                shot_st.append(subs['st'])
                shot_et.append(subs['et'])

                for sub in subs['contained_subs']:
                    st = sub['st']

                    if st in subtitle_sts:
                        continue

                    subtitle_sts.add(st)
                    shot_subtitles.append((float(st), sub))

            shot_st = sorted(shot_st, key=float)
            shot_et = sorted(shot_et, key=float)
            shot_subtitles.sort()

            if shot_subtitles:
                row['subtitle'] = {}
                row['subtitle']["contained_subs"] = [sub for st, sub in shot_subtitles]
                row['subtitle']["st"] = shot_st[0]
                row['subtitle']["et"] = shot_et[-1]
            else:
                row['subtitle'] = ''
        else:
            # shot question
            if row['vid'] in subtitles:
                row['subtitle'] = subtitles[row['vid']]
            else:
                row['subtitle'] = ''

        if row['subtitle'] == '':
            row['subtitle'] = empty_sub  # prevent empty string
        res.append(row)

    write_json(res, new_path)
