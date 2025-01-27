import os 

ds = '1K'
names = os.listdir(f'/home/szxie/storage/DL3DV/960P-unzip/{ds}')

for name in names:
    # print(name)
    os.system(f'bash run.sh {ds} {name}')