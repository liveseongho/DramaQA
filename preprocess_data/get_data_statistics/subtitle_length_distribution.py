import os, json
from draw_chart import draw_chart
from pprint import pprint

config = json.load(open('../preprocess_config.json', 'r', encoding='utf-8'))
root_dir = config['root_dir']

vid_subtitle_word_set = set()

max_subtitle_len = 0
max_word_len = 0



subtitle_cnt_list = [0 for _ in range(1001)]
word_cnt_list = [0 for _ in range(1001)]
total_cnt = 0

with open(os.path.join(root_dir,
                       'AnotherMissOh_QA/AnotherMissOhQA_train_set_script.json')) as f:
    qa_datum = json.load(f)
    
    for qa_data in qa_datum:
        vid = qa_data['vid']
        if vid in vid_subtitle_word_set:
            continue
        else:
            vid_subtitle_word_set.add(vid)


        subtitle = qa_data['subtitle']
        
        if subtitle == '.':
            subtitle_len = 0
            word_len = 0
        else:
            contained_subs = subtitle['contained_subs']
            
            subtitle_len = len(contained_subs)
            word_len = sum(list(map(lambda t: len(t['utter'].split()), contained_subs)))

        total_cnt += 1
        
        max_subtitle_len=max(max_subtitle_len, subtitle_len)
        max_word_len = max(max_word_len, word_len)
        
        subtitle_cnt_list[subtitle_len] += 1
        word_cnt_list[word_len] += 1
        
        
print('Total count:', total_cnt)
print('max_subtitle_len:', max_subtitle_len)
print('max_word_len:', max_word_len)

# subtitle length distribution
subtitle_cnt_list = subtitle_cnt_list[:max_subtitle_len + 1]

x = list(range(max_subtitle_len + 1))
y = subtitle_cnt_list

draw_chart(x, y, 4, 'subtitle_length_distribution', x_axis_name='subtitle length')
print('\t'.join(list(map(lambda t: str(t), x))))
print('\t'.join(list(map(lambda t: str(t), y))))


# word length distribution
word_cnt_list = word_cnt_list[1:100+1]

x = list(range(1, 100+1))
y = word_cnt_list

draw_chart(x, y, 4, 'word_length_distribution', x_axis_name='word count')
print('\t'.join(list(map(lambda t: str(t), x))))
print('\t'.join(list(map(lambda t: str(t), y))))



