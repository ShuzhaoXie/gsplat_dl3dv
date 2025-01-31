import os 
from  multiprocessing import Process, Pool, Manager
import glob

data_dir = '/home/szxie/gsplat_dl3dv/DL3DV/960P-unzip'
save_dir = '/home/szxie/gsplat_dl3dv/DL3DV/pretrained'


def fun1(q, ds, name, data_dir, save_dir):
    cid = q.get()
    print(f'using {cid}, start training: {name}')
    os.system(f'bash run.sh {ds} {name} {cid} {data_dir} {save_dir}')
    q.put(cid)

if __name__ == '__main__':
    ds = '8K'
    names = os.listdir(f'{data_dir}/{ds}')
    
    untrained_names = []
    for name in names:
        if not glob.glob(f'{save_dir}/{ds}/{name}/*/videos/traj_29999.mp4'):
            untrained_names.append(name)

    print(len(untrained_names))
    
    # download required models
    print(f'using 0, start training: {untrained_names[0]}')
    os.system(f'bash run.sh {ds} {untrained_names[0]} 0 {data_dir} {save_dir}')
    
    with Manager() as manager:
        q = manager.Queue()
        q.put(0)
        q.put(1)
        q.put(2)
        
        pl = Pool(3)
        
        # print('here')
        for i, name in enumerate(untrained_names[1:]):
            # print(name)
            pl.apply_async(func=fun1, args=(q, ds, name, data_dir, save_dir, ))
        
        pl.close()
        pl.join()