import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from model import physicsbox
import scipy.misc
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='train', help='train, test')
#parser.add_argument('--phase', dest='phase', default='test', help='train, test')

parser.add_argument('--model_name', dest='model_name', default='model_gan_se_res_unet', help='the model name')
#parser.add_argument('--model_name', dest='model_name', default='model_gan_unet', help='the model name')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch,training is 64, testing can be any number!')
parser.add_argument('--input_c_dim', dest='input_c_dim', type=int, default= 3, help='# of input image channels')
parser.add_argument('--output_c_dim', dest='output_c_dim', type=int, default= 1, help='# of output image channels')
parser.add_argument('--condition_dim', dest='condition_dim', type=int, default= 6, help='# of condition channels')
parser.add_argument('--overlap_dim', dest='overlap_dim', type=int, default= 3, help='# of overlap channels') #overlap channels between input and condition
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default= 10000, help='weight on L1 term in objective')
parser.add_argument('--L2_lambda', dest='L2_lambda', type=float, default= 1, help='weight on L1 term in objective')
parser.add_argument('--epoch', dest='epoch', type=int, default= 500, help='# the total training epoch number')
parser.add_argument('--dataset_name_train_valid', dest='dataset_train_valid', default='../data/dataset_train_valid.npy', help='name of the dataset_train_valid')
parser.add_argument('--dataset_name_test', dest='dataset_test', default='../data/dataset_test.npy', help='name of the dataset_test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--input_size', dest='input_size', type=int, default=128*64, help='input image size')
parser.add_argument('--output_size', dest='output_size', type=int, default=128*64, help='output to this size')
parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=128, help='# dim of gen filters in first conv layer')
parser.add_argument('--df_dim', dest='df_dim', type=int, default=32, help='# dim of discri filters in first conv layer')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=100, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--epoch_restore', dest='epoch_restore', type=int, default= 498, help='# the epoch to be restord')
parser.add_argument('--restore_model', dest='restore_model', default='none', help='the name of the model you restore')

args = parser.parse_args()

def main():

    # # # set GPU
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    # # # set CPU
    # config = tf.ConfigProto(device_count={"CPU": 4},
                   # inter_op_parallelism_threads = 1,
                    # intra_op_parallelism_threads = 1,
                    # log_device_placement=True)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    with tf.Session(config=config) as sess:
        model = physicsbox(sess, image_size=args.input_size,
                        batch_size = args.batch_size,
                        output_size=args.output_size,
                        input_c_dim = args.input_c_dim,
                        output_c_dim = args.output_c_dim,
                        condition_dim = args.condition_dim,
                        overlap_dim = args.overlap_dim,
                        dataset_train_valid = args.dataset_train_valid,
                        dataset_test = args.dataset_test,
                        restore_model=args.restore_model,
                        epoch_restore=args.epoch_restore,
                        save_epoch_freq = args.save_epoch_freq,
                        save_latest_freq = args.save_latest_freq,
                        gf_dim=args.gf_dim,
                        df_dim=args.df_dim,
                        L1_lambda=args.L1_lambda,
                        L2_lambda=args.L2_lambda,
                        model_name=args.model_name)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test_after_training(args)
            model.train_results_after_training(args)


if __name__ == '__main__':
    main()
    #tf.app.run()
