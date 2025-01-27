import os 

ds = '5K'
names = os.listdir(f'/home/szxie/DL3DV/960P-unzip/{ds}')

for name in names:
    # print(name)
    os.system(f'bash run.sh {ds} {name}')