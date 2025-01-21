import os

# os.makedirs('960P-unzip', exist_ok=True)
# os.makedirs('960P-unzip/3K', exist_ok=True)
# os.makedirs('960P-unzip/4K', exist_ok=True)
# os.makedirs('960P-unzip/7K', exist_ok=True)

ori_dir = '960P'
dir_name = '960P-unzip'
os.makedirs(dir_name, exist_ok=True)

k_dirs = ['8K']
for kdir in k_dirs:
    zip_names = os.listdir(os.path.join(ori_dir, kdir))
    # print(len(zip_names))
    os.makedirs(f'{dir_name}/{kdir}', exist_ok=True)
    for zipn in zip_names:
        if zipn[-1] == 'p':
            zipn_clear = zipn.split('.')[0]
            os.system(f'unzip 960P/{kdir}/{zipn} -d {dir_name}/{kdir}/{zipn_clear}')
            