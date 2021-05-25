import pandas as pd
import seaborn as sns



def draw_chart(x, y, aspect, chart_title, x_axis_name, y_axis_name='count'):
    
    sns.set(font_scale=max(1, aspect / 2.5))
    
    df = pd.DataFrame({x_axis_name: x, y_axis_name: y})
    
    graph = sns.catplot(data=df, x=x_axis_name, y=y_axis_name,
                        kind='bar', height=8, aspect=aspect)
    graph.savefig('chart/' + chart_title + '.png', dpi=400)
