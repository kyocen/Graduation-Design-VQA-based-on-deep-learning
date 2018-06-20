import os
import glob
import numpy as np
import h5py
import progressbar
import json

root='../data/vqa02/'

def main(root):
    for split in ('train2014','val2014'):
        path='{}_feature_2/'.format(split)
        trans(os.path.join(root,path), root, split)


def trans(from_dir, to_dir, split):
    print('transfer from {} to {}'.format(from_dir,to_dir))
    print("start transfer {}".format(split))
    pattern=os.path.join(from_dir,"*.npy")
    h5file = '{}{}_img_feature.h5'.format(to_dir, split)
    bar = progressbar.ProgressBar()
    imgid_index={}
    num_file=len([name for name in os.listdir(from_dir) if os.path.isfile(os.path.join(from_dir, name))])
    print('[size] num_file: ', num_file)

    with h5py.File(h5file, 'w') as f:
        f.create_dataset(name='img',shape=(num_file,2048,7,7),dtype='float32')
        for i, filepath in enumerate(bar(glob.glob(pattern))):
            id = os.path.basename(filepath).split('.')[0]
            imgid_index[int(id)]=i
            feature = np.load(filepath)
            f['img'][i]=feature# Save an 3d ndarray (2048,7,7) id(12 string)->3d ndarray (2048,7,7)
            if i==1:
                print('[id] ',id)
                print('[filepath] ',filepath)
                print('[feature size] ',feature.shape)
                print('[feature size]',f['img'][i].shape)
    with open(os.path.join(to_dir, '{}_imgid_index.json'.format(split)), "w") as fdict:
        json.dump(imgid_index, fdict)
        print('[img dict stored] size: ',len(imgid_index))


if __name__=='__main__':
    main(root)