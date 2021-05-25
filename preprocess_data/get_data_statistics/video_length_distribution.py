import numpy as np
import glob
from pprint import pprint
from collections import Counter
import pickle
from draw_chart import draw_chart

# '''
counter = [Counter(), Counter(), Counter()]

# _0000
i3d_rgb_list = sorted(glob.glob('/data/dataset/AnotherMissOh/i3d_rgb_v0.3/*.npy'))
i3d_flow_list = sorted(glob.glob('/data/dataset/AnotherMissOh/i3d_flow_v0.3/*.npy'))
vggish_list = sorted(glob.glob('/data/dataset/AnotherMissOh/vggish_v0.4/*.npy'))

print(len(i3d_rgb_list))
print(len(i3d_flow_list))
print(len(vggish_list))

max_r, max_f, max_v = 0, 0, 0
max_name = ''

cnt_list = [0 for _ in range(600)]


for i, (rgb, flow, vgg) in enumerate(zip(i3d_rgb_list, i3d_flow_list, vggish_list)):
    r = np.load(rgb)
    f = np.load(flow)
    v = np.load(vgg)
    
    # if i > 6000:
    #     break

    if r.shape[0]>max_r:
        max_r = r.shape[0]
        max_f = f.shape[0]
        max_v = v.shape[0]
        max_name = rgb
        
    print(rgb, '\t', r.shape[0], f.shape[0], v.shape[0])
        
    
    cnt_list[r.shape[0]] += 1
    
    counter[0][r.shape[0]] += 1
    counter[1][f.shape[0]] += 1
    counter[2][v.shape[0]] += 1

print(max_name, max_r, max_f, max_v)

for i in range(60):
    print(i, cnt_list[i])

pickle.dump(cnt_list, open('video_length_cnt_list.pkl', 'wb'))

common_counter = counter[0].most_common(25)

x = list(map(lambda t: t[0], common_counter))
y = list(map(lambda t: t[1], common_counter))

print('Total:', sum(counter[0]))
print('Most common 25:', sum([t[1] for t in common_counter]))

draw_chart(x, y, 1.5, 'video_length_distribution', x_axis_name='i3d_rgb length')


# '''

'''cnt_list = pickle.load(open('cnt_list.pkl', 'rb'))

print(cnt_list)

x = list(range(len(cnt_list)))[1:31]
y = cnt_list[1:31]

print('Total:', sum(cnt_list))
print('Top 30:', sum(cnt_list[1:31]))

df = pd.DataFrame({'i3d_rgb length': x, 'count': y})

graph = sns.catplot(data=df, x='i3d_rgb length', y='count',
                    kind='bar', height=8, aspect=1.5)
graph.savefig('length.png', dpi=800)'''