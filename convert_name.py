import os
pen_datas = os.listdir('./pen_data')
for pen_data in pen_datas:
    imgs = os.listdir(f'./pen_data/{pen_data}')
    for index, img in enumerate(imgs):
        tails_name = img.split('.')[1]
        os.rename(f'./pen_data/{pen_data}/{img}', f'./pen_data/{pen_data}/{pen_data}_{index}.{tails_name}')

