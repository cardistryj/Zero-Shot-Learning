

os.environ['CUDA_VISIBLE_DEVICES'] = '1'#使用GPU 1

lam_center=0.03
lam_encoder=10
featureDim=256
modification='NSRNN'
epoch_num = 200
lr = 1e-3
batchsize = 256

num_seen = 22
num_class_total = 24

zero_shot_path = 'result_0_shot/acc_{}_{}d_0.03C_10ATE_{}_{}.txt'
cluster_path = 'result_0_shot/cluster_{}_{}d_0.03C_10ATE_{}_{}.txt'
softmax_path = 'result_0_shot/softmax_{}_{}d_0.03C_10ATE_{}_{}.txt'

path_dataset = '../data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
model_path='./models/model{}_{}d_0.03C_10ATE.pkl'.format(modification,featureDim)
