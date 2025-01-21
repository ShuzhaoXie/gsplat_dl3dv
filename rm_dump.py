import os

for sub_dir in ['1K', '5K', '8K', '10K', '11K']:
    
    pt = f'/home/szxie/DL3DV/960P-unzip/{sub_dir}'
    
    fp = f'lost_zips/{sub_dir}.txt'
    
    cptnames = os.listdir(pt)
    for cn in cptnames:
        son_names =  os.listdir(os.path.join(pt, cn))
        if len(son_names) == 1 and cn == son_names[0]:
            os.system(f'mv {pt}/{cn}/{cn}/* {pt}/{cn}/')
            os.system(f'rm -rf {pt}/{cn}/{cn}')
            # print(cn)

    with open(fp, 'a+') as f:
        for cn in cptnames:
            son_names = os.listdir(os.path.join(pt, cn))
            if len(son_names) == 1:
                f.write(f'{cn}\n')
                print(cn, son_names[0])
                print(len(os.listdir(os.path.join(pt, cn, son_names[0]))))
    
# d2cad727a318a1645a5de505033f50ba5066055c30caeaa570ad7d88e012d795
# 9d54ee2753b0e45a3f455a7711bf855687c5d6dccfef463247f613f82192c789
# 983e2f92b3810444651f4e5a5225c080fb581161facee3a212e79da3e084ba83