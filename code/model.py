from __future__ import division
import os
import sys
import time
# from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from six.moves import xrange
from ops import *
from utils import *
imageshow = 0 # 0 for just save, 1 for plot and save
if imageshow == 1:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


# This is to train the 128x128 dataset
class physicsbox(object):
    def __init__(self, sess, width=128, height=64, image_size=128*64, 
                 batch_size=64, output_size=128*64, 
                 gf_dim=128, df_dim=128, L1_lambda=10000, L2_lambda=1, 
                 input_c_dim=3, output_c_dim=1, condition_dim=6, overlap_dim=3, 
                 dataset_train_valid='../data/dataset_train_valid.npy', 
                 dataset_test='../data/dataset_test.npy', 
                 checkpoint_dir='./checkpoint', 
                 sample_dir='./sample', 
                 model_name='none', 
                 save_epoch_freq= 100, 
                 save_latest_freq = 5, 
                 epoch_restore = 500,
                 restore_model='none'):
        """ all the parameters above have no action. Please update all the variables of main.py
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [128]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [128]
            input_c_dim: (optional) Dimension of input image color (channel). For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.condition_dim = condition_dim
        self.overlap_dim = overlap_dim

        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda

        self.height = height
        self.width = width
        self.resolution = height*width
        self.resolution_element = (height-1)*(width-1)

        self.dataset_train_valid = dataset_train_valid
        self.dataset_test = dataset_test
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_epoch_freq = save_epoch_freq
        self.save_latest_freq = save_latest_freq

        self.build_model()
        self.restore_model = restore_model
        self.epoch_restore = epoch_restore



    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.resolution*(self.input_c_dim+self.output_c_dim+self.condition_dim-self.overlap_dim)],name='real_A_and_B_images') # -self.overlap_dim means the overlap channel number among input, output and condition
        #input channel
        self.vf = self.real_data[:, self.resolution*0:self.resolution*1]
        #self.ux = self.real_data[:, self.resolution*1:self.resolution*2]
        #self.uy = self.real_data[:, self.resolution*2:self.resolution*3]
        self.vm_stress = self.real_data[:, self.resolution*1:self.resolution*2]
        self.strain_energy = self.real_data[:, self.resolution*2:self.resolution*3]
        #output channel
        self.real_B = self.real_data[:, self.resolution*3:self.resolution*4] #output_structure ground truth
        # print('real_B:', tf.size(self.real_B))
        
        #condition channel
        self.bc = self.real_data[:, self.resolution*4:self.resolution*5]
        self.loadx = self.real_data[:, self.resolution*5:self.resolution*6]
        self.loady = self.real_data[:, self.resolution*6:self.resolution*7]
        #self.final_stress_vm = self.real_data[:, self.resolution*7:self.resolution*8]
        #self.final_strain_energy = self.real_data[:, self.resolution*8:self.resolution*9]

        self.real_A = tf.reshape(tf.stack([self.vf,
                                     self.vm_stress,
                                     self.strain_energy,
                                     ], axis=-1), [-1, self.height, self.width, self.input_c_dim])
        # print('real_A:', tf.size(self.real_A))

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.reshape(tf.stack([self.vf,
                                      self.bc,
                                      self.loadx,
                                      self.loady,
                                      self.vm_stress,
                                      self.strain_energy,
                                      self.real_B,
                                      ], axis=-1), [-1, self.height, self.width, self.condition_dim+self.output_c_dim])
        #print('real_AB', tf.size(self.real_AB))

        self.fake_AB = tf.reshape(tf.stack([self.vf,
                                      self.bc,
                                      self.loadx,
                                      self.loady,
                                      self.vm_stress,
                                      self.strain_energy,
                                      tf.reshape(self.fake_B, [-1, self.resolution]),
                                      ], axis=-1), [-1, self.height, self.width, self.condition_dim+self.output_c_dim])
        #print('fake_AB', tf.size(self.fake_AB))

        self.mse = tf.losses.mean_squared_error(self.real_B, tf.reshape(self.fake_B, [-1, self.resolution]))
        self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.real_B, tf.reshape(self.fake_B, [-1, self.resolution]))))
        self.vfae = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(self.real_B,1),tf.reduce_sum(self.fake_B,[1,2])))/self.resolution)
        
        self.fake_B_sample = self.generator(self.real_A, is_training=False, reuse=True)
        self.mse_sample = tf.losses.mean_squared_error(self.real_B, tf.reshape(self.fake_B_sample, [-1, self.resolution]))
        self.mae_sample = tf.reduce_mean(tf.abs(tf.subtract(self.real_B, tf.reshape(self.fake_B_sample, [-1, self.resolution]))))
        self.vfae_sample = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(self.real_B,1),tf.reduce_sum(self.fake_B_sample,[1,2])))/self.resolution)
        
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.gan_loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.gan_loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        
        # gan_loss_d is the loss of discriminator in gan.
        self.gan_loss_d = self.gan_loss_d_real + self.gan_loss_d_fake
        
        # gan_loss_g is the loss of generator in gan.
        self.gan_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        # self.gan_loss_d_real_sum = tf.summary.scalar("gan_loss_d_real", self.gan_loss_d_real)
        # self.gan_loss_d_fake_sum = tf.summary.scalar("gan_loss_d_fake", self.gan_loss_d_fake)
        
        # g_loss_final is the final object for generator.
        self.g_loss_final = self.gan_loss_g + self.L1_lambda * self.mse + self.L2_lambda * self.vfae
        # self.g_loss_final_sum = tf.summary.scalar("g_loss_final", self.g_loss_final)
        # self.gan_loss_d_sum = tf.summary.scalar("gan_loss_d", self.gan_loss_d)
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1000)



    def data_processing(self, dataset):
        num_total = dataset.shape[0]
        dataset_output = np.zeros((num_total,self.resolution*(self.input_c_dim+self.output_c_dim+self.condition_dim-self.overlap_dim)),dtype = np.float16) # -self.overlap_dim means overlap channels

        for i in range(num_total):
            vf = dataset[i,0*self.resolution:1*self.resolution]

            bc = dataset[i,1*self.resolution:2*self.resolution]
            # plt.imshow(np.reshape(bc,(self.height,self.width)),cmap='jet')
            loadx = dataset[i,2*self.resolution:3*self.resolution]
            loady = dataset[i,3*self.resolution:4*self.resolution]

            output_structure = dataset[i,4*self.resolution:4*self.resolution+self.resolution_element]
            output_structure_2d = np.reshape(output_structure,(self.height-1,self.width-1))
            output_structure_2d_padded = np.pad(output_structure_2d,((0,1),(0,1)),'constant')
            output_structure_padded = np.reshape(output_structure_2d_padded,(-1,1))
            # plt.imshow(np.reshape(output_structure_padded,(self.height,self.width)),cmap='jet')

            # u = dataset[i,4*self.resolution+self.resolution_element:6*self.resolution+self.resolution_element]
            # u_2d = np.reshape(u,(self.height,self.width,2))
            # ux_2d = u_2d[:,:,0]
            # ux = np.reshape(ux_2d,(-1,1))
            # #plt.imshow(np.reshape(ux,(self.height,self.width)),cmap='jet')
            # uy_2d = u_2d[:,:,1]
            # uy = np.reshape(uy_2d,(-1,1))
            # #plt.imshow(np.reshape(uy,(self.height,self.width)),cmap='jet')

            strain = dataset[i,6*self.resolution+self.resolution_element:9*self.resolution+self.resolution_element]
            strain_2d = np.reshape(strain,(self.height,self.width,3))
            strainxx_2d = strain_2d[:,:,0]
            strainxx = np.reshape(strainxx_2d,(-1,1))
            # plt.imshow(np.reshape(strainxx,(self.height,self.width)),cmap='jet')
            strainyy_2d = strain_2d[:,:,1]
            strainyy = np.reshape(strainyy_2d,(-1,1))
            # plt.imshow(np.reshape(strainyy,(self.height,self.width)),cmap='jet')
            strainxy_2d = strain_2d[:,:,2]
            strainxy = np.reshape(strainxy_2d,(-1,1))
            # plt.imshow(np.reshape(strainxy,(self.height,self.width)),cmap='jet')

            stress = dataset[i,9*self.resolution+self.resolution_element:12*self.resolution+self.resolution_element]
            stress_2d = np.reshape(stress,(self.height,self.width,3))
            stressxx_2d = stress_2d[:,:,0]
            stressxx = np.reshape(stressxx_2d,(-1,1))
            # plt.imshow(np.reshape(stressxx,(self.height,self.width)),cmap='jet')
            stressyy_2d = stress_2d[:,:,1]
            stressyy = np.reshape(stressyy_2d,(-1,1))
            # plt.imshow(np.reshape(stressxx,(self.height,self.width)),cmap='jet')
            stressxy_2d = stress_2d[:,:,2]
            stressxy = np.reshape(stressxy_2d,(-1,1))
            # plt.imshow(np.reshape(stressxy,(self.height,self.width)),cmap='jet')

            stress_vm_2d = np.sqrt(0.5*((stressxx_2d-stressyy_2d)**2+3*stressxy_2d**2))
            stress_vm = np.reshape(stress_vm_2d,(-1,1))
            # plt.imshow(np.reshape(stress_vm,(self.height,self.width)),cmap='jet')
            # stress_vm_norm = stress_vm/np.max(stress_vm)

            stress_vm_norm = (stress_vm-np.min(stress_vm))/(np.max(stress_vm)-np.min(stress_vm)+0.0001)
            stress_vm_norm = np.power(stress_vm_norm,0.2)
            # stress_vm_norm = stress_vm
            # plt.imshow(np.reshape(stress_vm_norm,(self.height,self.width)),cmap='jet')
            # plt.contour(np.flip(np.reshape(stress_vm_norm,(self.height,self.width)),0),100)

            strain_energy_2d = 0.5*(strainxx_2d*stressxx_2d+strainyy_2d*stressyy_2d+2*strainxy_2d*stressxy_2d)
            strain_energy = np.reshape(strain_energy_2d,(-1,1))
            # plt.imshow(np.reshape(strain_energy,(self.height,self.width)),cmap='jet')
            # strain_energy_norm = strain_energy/(max(strain_energy))

            strain_energy_norm = (strain_energy-min(strain_energy))/(max(strain_energy)-min(strain_energy)+0.0001)
            strain_energy_norm = np.power(strain_energy_norm,0.2)
            # strain_energy_norm = strain_energy
            # plt.imshow(np.reshape(strain_energy_norm,(self.height,self.width)),cmap='jet')
            # plt.contour(np.flip(np.reshape(strain_energy_norm,(self.height,self.width)),0),100)

            # final_u = dataset[i,12*self.resolution+self.resolution_element:14*self.resolution+self.resolution_element]
            # final_u_2d = np.reshape(final_u,(self.height,self.width,2))
            # final_ux_2d = final_u_2d[:,:,0]
            # final_ux = np.reshape(final_ux_2d,(-1,1))
            # #plt.imshow(np.reshape(final_ux,(self.height,self.width)),cmap='jet')
            # final_uy_2d = final_u_2d[:,:,1]
            # final_uy = np.reshape(final_uy_2d,(-1,1))
            # #plt.imshow(np.reshape(final_uy,(self.height,self.width)),cmap='jet')

            ##final strain output
            final_strain = dataset[i,14*self.resolution+self.resolution_element:17*self.resolution+self.resolution_element]
            final_strain_2d = np.reshape(final_strain,(self.height,self.width,3))
            final_strainxx_2d = final_strain_2d[:,:,0]
            final_strainxx = np.reshape(final_strainxx_2d,(-1,1))
            #plt.imshow(np.reshape(final_strainxx,(self.height,self.width)),cmap='jet')
            final_strainyy_2d = final_strain_2d[:,:,1]
            final_strainyy = np.reshape(final_strainyy_2d,(-1,1))
            # plt.imshow(np.reshape(final_strainyy,(self.height,self.width)),cmap='jet')
            final_strainxy_2d = final_strain_2d[:,:,2]
            final_strainxy = np.reshape(final_strainxy_2d,(-1,1))
            # plt.imshow(np.reshape(final_strainxy,(self.height,self.width)),cmap='jet')

            final_stress = dataset[i,17*self.resolution+self.resolution_element:20*self.resolution+self.resolution_element]
            final_stress_2d = np.reshape(final_stress,(self.height,self.width,3))
            final_stressxx_2d = final_stress_2d[:,:,0]
            final_stressxx = np.reshape(final_stressxx_2d,(-1,1))
            # plt.imshow(np.reshape(final_stressxx,(self.height,self.width)),cmap='jet')
            final_stressyy_2d = final_stress_2d[:,:,1]
            final_stressyy = np.reshape(final_stressyy_2d,(-1,1))
            # plt.imshow(np.reshape(final_stressxx,(self.height,self.width)),cmap='jet')
            final_stressxy_2d = final_stress_2d[:,:,2]
            final_stressxy = np.reshape(final_stressxy_2d,(-1,1))
            # plt.imshow(np.reshape(stressxy,(self.height,self.width)),cmap='jet')

            final_stress_vm_2d = np.sqrt(0.5*((final_stressxx_2d-final_stressyy_2d)**2+3*final_stressxy_2d**2))
            final_stress_vm = np.reshape(final_stress_vm_2d,(-1,1))
            # plt.imshow(np.reshape(final_stress_vm,(self.height,self.width)),cmap='jet')
            # final_stress_vm_norm = final_stress_vm/np.max(final_stress_vm)

            final_stress_vm_norm = (final_stress_vm-np.min(final_stress_vm))/(np.max(final_stress_vm)-np.min(final_stress_vm)+0.000001)
            final_stress_vm_norm = np.power(final_stress_vm_norm,0.2)
            # final_stress_vm_norm = final_stress_vm
            #plt.imshow(np.reshape(final_stress_vm_norm,(self.height,self.width)),cmap='jet')
            #plt.contour(np.flip(np.reshape(final_stress_vm_norm,(self.height,self.width)),0),100)

            # final_strain_energy_2d = 0.5*(final_strainxx_2d*final_stressxx_2d+final_strainyy_2d*final_stressyy_2d+2*final_strainxy_2d*final_stressxy_2d)
            # final_strain_energy = np.reshape(final_strain_energy_2d,(-1,1))
            # plt.imshow(np.reshape(final_strain_energy,(self.height,self.width)),cmap='jet')
            # final_strain_energy_norm = final_strain_energy/(max(final_strain_energy))

            # final_strain_energy_norm = (final_strain_energy-min(final_strain_energy))/(max(final_strain_energy)-min(final_strain_energy)+0.000001)
            # final_strain_energy_norm = np.power(final_strain_energy_norm,0.2)
            # final_strain_energy_norm = final_strain_energy
            # plt.imshow(np.reshape(final_strain_energy_norm,(self.height,self.width)),cmap='jet')
            # plt.contour(np.flip(np.reshape(final_strain_energy_norm,(self.height,self.width)),0),100)

            dataset_output[i,0*self.resolution:1*self.resolution] = vf
            # dataset_output[i,1*self.resolution:2*self.resolution] = ux[:,0]
            # dataset_output[i,2*self.resolution:3*self.resolution] = uy[:,0]
            # dataset_output[i,1*self.resolution:2*self.resolution] = stress_vm_norm[:,0]
            dataset_output[i,1*self.resolution:2*self.resolution] = stress_vm[:,0]
            # dataset_output[i,2*self.resolution:3*self.resolution] = strain_energy_norm[:,0]
            dataset_output[i,2*self.resolution:3*self.resolution] = strain_energy[:,0]
            dataset_output[i,3*self.resolution:4*self.resolution] = output_structure_padded[:,0]
            dataset_output[i,4*self.resolution:5*self.resolution] = bc
            dataset_output[i,5*self.resolution:6*self.resolution] = loadx
            dataset_output[i,6*self.resolution:7*self.resolution] = loady
            # dataset_output[i,7*self.resolution:8*self.resolution] = final_stress_vm_norm[:,0]
            # dataset_output[i,7*self.resolution:8*self.resolution] = final_stress_vm[:,0]
            # dataset_output[i,8*self.resolution:9*self.resolution] = final_strain_energy_norm[:,0]
            # dataset_output[i,8*self.resolution:9*self.resolution] = final_strain_energy[:,0]
        return dataset_output



    def save(self, checkpoint_dir, step):
        # model_dir = "%s_%s_%s" % (self.model_name, self.input_c_dim, self.condition_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name),global_step=step)



    def get_sample_data(self,path,size):
        data_test_all = np.load(path)
        data_test = self.data_processing(data_test_all)
        np.random.shuffle(data_test)
        num_test = data_test.shape[0]
        number_test = min(size,num_test)
        sample_data_test = data_test[:int(number_test)]
        num_sample_test = sample_data_test.shape[0]
        print("Sample testing amount:", num_sample_test)
        return sample_data_test



    def sample_model_validation(self, sample_dir, epoch, validation_data, final=False):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        # print(np.shape(validation_data))
        # np.random.shuffle(validation_data)
        mse_total = 0
        mae_total = 0
        vfae_total = 0
        idx_num = int(np.ceil(np.shape(validation_data)[0]//self.batch_size))
        for idx in range(idx_num):
            starter = idx*self.batch_size
            end = min(starter + self.batch_size, 5000000)
            feed_test = {self.real_data: validation_data[starter:end]}
            samples, mse, mae, vfae = self.sess.run(
                [self.fake_B_sample, self.mse_sample, self.mae_sample, self.vfae_sample],
                feed_dict=feed_test
                )
            mse_total += mse
            mae_total += mae
            vfae_total += vfae
            if final:
                if idx == 0:
                    validation_result = samples
                else:
                    validation_result = np.concatenate((validation_result, samples), axis=0)
        validation_loss_writer(sample_dir, self.model_name, vfae_total/idx_num, mse_total/idx_num, mae_total/idx_num)
        print("[Sample] validation_vfae: {:.8f}, validation_mse: {:.8f}, validation_mae: {:.8f}".format(vfae_total/idx_num, mse_total/idx_num, mae_total/idx_num))

        if final:
            return validation_result



    def sample_model_test(self, sample_dir, epoch, test_data, final=False):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        # print(np.shape(test_data))
        # np.random.shuffle(test_data)
        vfae_total = 0
        mse_total = 0
        mae_total = 0
        idx_num = int(np.ceil(np.shape(test_data)[0]//self.batch_size))
        for idx in range(idx_num):
            starter = idx*self.batch_size
            end = min(starter + self.batch_size, 5000000)
            feed_test = {self.real_data: test_data[starter:end]}
            samples, mse, mae, vfae = self.sess.run(
                [self.fake_B_sample, self.mse_sample, self.mae_sample, self.vfae_sample],
                feed_dict=feed_test
                )
            vfae_total += vfae
            mse_total += mse
            mae_total += mae
            if final:
                if idx == 0:
                    test_result = samples
                else:
                    test_result = np.concatenate((test_result, samples), axis=0)
        testing_loss_writer(sample_dir, self.model_name, vfae_total/idx_num, mse_total/idx_num, mae_total/idx_num)
        print("[Sample] test_vfae: {:.8f}, test_mse: {:.8f}, test_mae: {:.8f}".format(vfae_total/idx_num, mse_total/idx_num, mae_total/idx_num))

        if final:
            return test_result

        # np.save(sample_dir+self.model_name+'_'+str(epoch)+'.npy', samples)



    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            # image is 64 x 128 x (condition_dim+output_c_dim)
            h0 = lrelu(conv2d(image, self.df_dim*1, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='d_h0_conv'))
            # h0 is (32 x 64 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='d_h1_conv'), is_training=True, name='d_h1_bn'))
            # h1 is (16 x 32 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='d_h2_conv'), is_training=True, name='d_h2_bn'))
            # h2 is (8 x 16 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='d_h3_conv'), is_training=True, name='d_h3_bn'))
            # h3 is (4 x 8 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            # h4 is a number for each data sample.

            return tf.nn.sigmoid(h4), h4



    # # Just the Unet for generator.
    def generator(self, image, y=None, is_training=True, reuse=False):
        with tf.variable_scope("generator") as scope:

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            hi = self.height #64
            wi = self.width #128
            hi2, hi4, hi8, hi16, hi32, hi64= int(hi/2), int(hi/4), int(hi/8), int(hi/16), int(hi/32), int(hi/64)
            wi2, wi4, wi8, wi16, wi32, wi64 = int(wi/2), int(wi/4), int(wi/8), int(wi/16), int(wi/32), int(wi/64)
            
            if self.model_name == 'model_gan_unet': ############ unet
                
                # image is (hi x wi x input_c_dim)
                e1 = conv2d(image, self.gf_dim*1, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e1_conv')
                # e1 is (hi2 x wi2 x self.gf_dim)
                e2 = batch_norm(conv2d(lrelu(e1), self.gf_dim*2, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e2_conv'), is_training=is_training, name='g_e2_bn')
                # e2 is (hi4 x wi4 x self.gf_dim*2)
                e3 = batch_norm(conv2d(lrelu(e2), self.gf_dim*4, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e3_conv'), is_training=is_training, name='g_e3_bn')
                # e3 is (hi8 x wi8 x self.gf_dim*4)
                e4 = batch_norm(conv2d(lrelu(e3), self.gf_dim*8, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e4_conv'), is_training=is_training, name='g_e4_bn')
                # e4 is (hi16 x wi16 x self.gf_dim*8)
                e5 = batch_norm(conv2d(lrelu(e4), self.gf_dim*16, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e5_conv'), is_training=is_training, name='g_e5_bn')
                # e5 is (hi32 x wi32 x self.gf_dim*16)

                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e5),[self.batch_size, hi32, wi32, self.gf_dim*16], k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02, name='g_d1', with_w=True)
                d1 = batch_norm(self.d1, is_training=is_training, name='g_d1_bn')
                d1 = tf.concat([d1, e5], 3)
                # d1 is (h32 x wi32 x self.gf_dim*16*2)

                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),[self.batch_size, hi16, wi16, self.gf_dim*8], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d2', with_w=True)
                d2 = batch_norm(self.d2, is_training=is_training, name='g_d2_bn')
                d2 = tf.concat([d2, e4], 3)
                # d2 is (h16 x wi16 x self.gf_dim*8*2)

                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),[self.batch_size, hi8, wi8, self.gf_dim*4], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d3', with_w=True)
                d3 = batch_norm(self.d3, is_training=is_training, name='g_d3_bn')
                d3 = tf.concat([d3, e3], 3)
                # d3 is (h8 x wi8 x self.gf_dim*4*2)

                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),[self.batch_size, hi4, wi4, self.gf_dim*2], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d4', with_w=True)
                d4 = batch_norm(self.d4, is_training=is_training, name='g_d4_bn')
                d4 = tf.concat([d4, e2], 3)
                # d4 is (h4 x wi4 x self.gf_dim*2*2)

                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),[self.batch_size, hi2, wi2, self.gf_dim*1], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d5', with_w=True)
                d5 = batch_norm(self.d5, is_training=is_training, name='g_d5_bn')
                d5 = tf.concat([d5, e1], 3)
                # d5 is (h2 x wi2 x self.gf_dim*1*2)

                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),[self.batch_size, hi, wi, self.output_c_dim], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d6', with_w=True)
                #self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.leaky_relu(d5,0.1),[self.batch_size, hi, wi, self.output_c_dim], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d6', with_w=True)
                # d6 is (hi x wi x output_c_dim)
                return tf.nn.sigmoid(self.d6)
            else: ############ se-res-unet
                # image is (hi x wi x input_c_dim)
                e1 = conv2d(image, self.gf_dim*1, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e1_conv')
                # e1 is (hi2 x wi2 x self.gf_dim): 32x64xself.gf_dim*1
                e2 = batch_norm(conv2d(lrelu(e1), self.gf_dim*2, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e2_conv'), is_training=is_training, name='g_e2_bn')
                # e2 is (hi4 x wi4 x self.gf_dim*2): 16x32xself.gf_dim*2
                e3 = batch_norm(conv2d(lrelu(e2), self.gf_dim*4, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='g_e3_conv'), is_training=is_training, name='g_e3_bn')
                # e3 is (hi8 x wi8 x self.gf_dim*4): 8x16xself.gf_dim*4
                
                rn1= residual_block(e3, self.gf_dim*4, "rsenet1")
                rn2= residual_block(rn1, self.gf_dim*4, "rsenet2")
                rn3= residual_block(rn2, self.gf_dim*4, "rsenet3")
                rn4= residual_block(rn3, self.gf_dim*4, "rsenet4")
                rn5= residual_block(rn4, self.gf_dim*4, "rsenet5")
                rn6= residual_block(rn5, self.gf_dim*4, "rsenet6")
                rn7= residual_block(rn6, self.gf_dim*4, "rsenet7")
                rn8= residual_block(rn7, self.gf_dim*4, "rsenet8")
                rn9= residual_block(rn8, self.gf_dim*4, "rsenet9")
                rn10= residual_block(rn9, self.gf_dim*4, "rsenet10")
                rn11= residual_block(rn10, self.gf_dim*4, "rsenet11")
                rn12= residual_block(rn11, self.gf_dim*4, "rsenet12")
                rn13= residual_block(rn12, self.gf_dim*4, "rsenet13")
                rn14= residual_block(rn13, self.gf_dim*4, "rsenet14")
                rn15= residual_block(rn14, self.gf_dim*4, "rsenet15")
                rn16= residual_block(rn15, self.gf_dim*4, "rsenet16")
                rn17= residual_block(rn16, self.gf_dim*4, "rsenet17")
                rn18= residual_block(rn17, self.gf_dim*4, "rsenet18")
                rn19= residual_block(rn18, self.gf_dim*4, "rsenet19")
                rn20= residual_block(rn19, self.gf_dim*4, "rsenet20")
                rn21= residual_block(rn20, self.gf_dim*4, "rsenet21")
                rn22= residual_block(rn21, self.gf_dim*4, "rsenet22")
                rn23= residual_block(rn22, self.gf_dim*4, "rsenet23")
                rn24= residual_block(rn23, self.gf_dim*4, "rsenet24")
                rn25= residual_block(rn24, self.gf_dim*4, "rsenet25")
                rn26= residual_block(rn25, self.gf_dim*4, "rsenet26")
                rn27= residual_block(rn26, self.gf_dim*4, "rsenet27")
                rn28= residual_block(rn27, self.gf_dim*4, "rsenet28")
                rn29= residual_block(rn28, self.gf_dim*4, "rsenet29")
                rn30= residual_block(rn29, self.gf_dim*4, "rsenet30")
                rn31= residual_block(rn30, self.gf_dim*4, "rsenet31")
                rn32= residual_block(rn31, self.gf_dim*4, "rsenet32")
                # # rnx is the same size as e3: (hi8 x wi8 x self.gf_dim*4): 8x16xself.gf_dim*4

                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(rn32),[self.batch_size, hi8, wi8, self.gf_dim*4], k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02, name='g_d1', with_w=True)
                d1 = batch_norm(self.d1, is_training=is_training, name='g_d1_bn')
                d1 = tf.concat([d1, e3], 3)
                # d1 is (h8 x wi8 x self.gf_dim*4*2)

                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),[self.batch_size, hi4, wi4, self.gf_dim*2], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d2', with_w=True)
                d2 = batch_norm(self.d2, is_training=is_training, name='g_d2_bn')
                d2 = tf.concat([d2, e2], 3)
                # d2 is (h4 x wi4 x self.gf_dim*2*2)
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),[self.batch_size, hi2, wi2, self.gf_dim*1], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d3', with_w=True)
                d3 = batch_norm(self.d3, is_training=is_training, name='g_d3_bn')
                d3 = tf.concat([d3, e1], 3)
                # d3 is (h2 x wi2 x self.gf_dim*1*2)
                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),[self.batch_size, hi, wi, self.output_c_dim], k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='g_d4', with_w=True)
                # d4 is (hi x wi x output_c_dim)
                return tf.nn.sigmoid(self.d4)



    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.gan_loss_d, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss_final, var_list=self.g_vars)
        if self.restore_model == 'none':
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            start_epoch = 0
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            # self.d_vars.initializer.run()
            # self.g_vars.initializer.run()
            self.saver.restore(self.sess, self.checkpoint_dir+'/pix2pix' + self.restore_model)
            start_epoch = int(self.restore_model.split('-')[-1])

        print('epoch start from:', start_epoch)

        start_time = time.time()

        print("model:", self.model_name)

        # load all datasets
        # generate useful datasets
        data_train_valid_all = np.load(self.dataset_train_valid)
        data_train_valid = self.data_processing(data_train_valid_all)

        data_test_all = np.load(self.dataset_test)
        data_test = self.data_processing(data_test_all)

        num_train_valid = data_train_valid.shape[0]
        np.random.shuffle(data_train_valid)

        num_train = int(num_train_valid*0.8)
        data_train = data_train_valid[:num_train]
        data_validation = data_train_valid[num_train:num_train_valid]

        num_train = data_train.shape[0]
        num_validation = data_validation.shape[0]
        num_test = data_test.shape[0]
        print(data_train.shape)
        print(data_validation.shape)
        print(data_test.shape)

        np_data_train =  data_train[:num_train // self.batch_size * self.batch_size]
        np_data_validation = data_validation[:num_validation // self.batch_size * self.batch_size]
        np_data_test = data_test[:num_test // self.batch_size * self.batch_size]

        num_train = np_data_train.shape[0]
        num_validation = np_data_validation.shape[0]
        num_test = np_data_test.shape[0]
        print("training amount:", num_train)
        print("validation amount:", num_validation)
        print("testing amount:", num_test)

        for epoch in xrange(start_epoch,args.epoch):
            batch_idxs =int(np.ceil(num_train // self.batch_size))
            errGAN_D_fake = 0
            errGAN_D_real = 0
            errG_final = 0
            errGAN_G_eval = 0
            mse_eval = 0
            mae_eval = 0
            vfae_eval = 0
            for idx in xrange(0, batch_idxs):
                batch_starter = idx*self.batch_size
                batch_end = min(batch_starter + self.batch_size, 100000)
                #print(batch_starter)
                batch = np_data_train[batch_starter:batch_end]
                #print(tf.size(batch))
                feed = {self.real_data: batch}
                
                # Update D network
                _ = self.sess.run([d_optim], feed_dict=feed)
                # self.writer.add_summary(summary_str, counter)
                
                # Run g_optim three times to make sure that gan_loss_d does not go to zero for unet.
                if self.model_name == 'model_gan_unet': ############ unet
                    _ = self.sess.run([g_optim], feed_dict=feed)
                    _ = self.sess.run([g_optim], feed_dict=feed)
                    _ = self.sess.run([g_optim], feed_dict=feed)
                else:
                    _ = self.sess.run([g_optim], feed_dict=feed)
                    # _ = self.sess.run([g_optim], feed_dict=feed)
                # self.writer.add_summary(summary_str, counter)
                
                errGAN_D_fake_t, errGAN_D_real_t, errG_final_t, errGAN_G_t, mse_eval_t, mae_eval_t, vfae_eval_t = self.sess.run([self.gan_loss_d_fake,self.gan_loss_d_real, self.g_loss_final, self.gan_loss_g, self.mse, self.mae,self.vfae], feed_dict=feed)

                errGAN_D_fake += errGAN_D_fake_t
                errGAN_D_real += errGAN_D_real_t
                errG_final += errG_final_t
                errGAN_G_eval += errGAN_G_t
                vfae_eval += vfae_eval_t
                mse_eval += mse_eval_t
                mae_eval += mae_eval_t
                
            training_loss_writer(args.sample_dir, epoch, (time.time() - start_time)/60, self.model_name, (errGAN_D_real)/batch_idxs, (errGAN_D_fake)/batch_idxs, (errGAN_D_fake+errGAN_D_real)/batch_idxs, errG_final/batch_idxs, errGAN_G_eval/batch_idxs, vfae_eval/batch_idxs, mse_eval/batch_idxs, mae_eval/batch_idxs)
            
            print("Epoch: [%2d] time: %4.4f, real_loss: %.8f, fake_loss: %.8f, gan_loss_d: %.8f, g_loss_whole: %.8f, gan_loss_g: %.8f, train_vfae: %.8f, train_mse: %.8f, train_mae: %.8f" % (epoch, (time.time() - start_time)/60, (errGAN_D_real)/batch_idxs, (errGAN_D_fake)/batch_idxs, (errGAN_D_fake+errGAN_D_real)/batch_idxs, errG_final/batch_idxs, errGAN_G_eval/batch_idxs, vfae_eval/batch_idxs, mse_eval/batch_idxs, mae_eval/batch_idxs))
            

            if np.mod(epoch, 1) == 0 and epoch <= args.epoch:
                self.sample_model_validation(args.sample_dir, epoch, np_data_validation)
                self.sample_model_test(args.sample_dir, epoch, np_data_test)

            if np.mod(epoch, args.save_epoch_freq) == 0 or (epoch >= args.epoch-self.save_latest_freq):
                self.save(args.checkpoint_dir, epoch)


    def test_after_training(self, args):
        ## path of the folder to save visualization result (note:need '/' at the end )
        image_save_path='./test/test_results_after_training/'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
            
        struc_save_path = image_save_path+'struc_real_fake.csv'
        if os.path.exists(struc_save_path):
            os.remove(struc_save_path)
        
        # saver=tf.train.import_meta_graph('./checkpoint/model_gan_unet_L1_10000_L2_1_3_6/model_gan_unet_L1_10000_L2_1-30.meta') #read the model
        # saver.restore(self.sess, "./checkpoint/model_gan_unet_L1_10000_L2_1_3_6/model_gan_unet_L1_10000_L2_1-30") #read the weights to model
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
        model_name_num = self.model_name + '-' + str(self.epoch_restore) # self.epoch_restore is the epoch number to restore
        model_path = os.path.join(checkpoint_dir, model_name_num)
        saver=tf.train.import_meta_graph(model_path + '.meta') # read the model
        saver.restore(self.sess, model_path) # read the weights to model

        fake_total=[]
        real_total=[]
        rvf_total=[]
        rvf_abs_total=[]
        fake_comp = []
        real_comp = []
        test_sample_size = max(640,1*self.batch_size) # Here 640 is the minimum test sample size which should be larger than the batch_size
        test_data=self.get_sample_data(self.dataset_test,test_sample_size)
        print(test_data.shape)
        #print ('volumn fraction',test_data[0,0],test_data[199,0],test_data[200,0],test_data[201,0])
        
        idx_num = int(np.ceil(np.shape(test_data)[0]//self.batch_size))
        
        
        for id_test in range(idx_num):
        # num_test=test_data.shape[0]
        # num_shuffle=np.random.choice(np.arange(num_test),self.batch_size)
        # test_shuffle=[test_data[i] for i in num_shuffle]
            starter = id_test*self.batch_size
            end = starter + self.batch_size
            feed_test = {self.real_data: test_data[starter:end]}
            mse, mae, fake_top_2d, real_top = self.sess.run([self.mse_sample, self.mae_sample,self.fake_B_sample,self.real_B],feed_dict=feed_test)
            fake_top = np.reshape(fake_top_2d, [-1, self.resolution])
            struc_real_fake = np.concatenate((real_top, fake_top), axis=1)
            df = pd.DataFrame(struc_real_fake)
            df.to_csv(image_save_path+'struc_real_fake.csv', mode='a', header=False)
            
            # print('MSE:',mse)
            # print('MAE:',mae)
            # print('Fake topology array shape:', fake_top.shape)
            # print('Reak topology array shape:', real_top.shape)
            # print(np.sum(fake_top[0,:]))
            # print(np.sum(real_top[0,:]))
            # sys.exit(0)
            vf_fake = np.array([np.sum(fake_top[i,:]) for i in np.arange(self.batch_size)])
            vf_true = np.array([np.sum(real_top[i,:]) for i in np.arange(self.batch_size)])
            # print(vf_fake)
            # print(vf_true)
            # sys.exit(0)
            re_vf = np.true_divide(np.subtract(vf_fake,vf_true),vf_true)
            re_vf_abs = np.true_divide(np.abs(np.subtract(vf_fake,vf_true)),vf_true)
            np.save(image_save_path+'re_vf_test', re_vf)
            # print(re_vf)
            # sys.exit(0)
            rvf_total.extend(re_vf)
            rvf_abs_total.extend(re_vf_abs)
            
            # # combine all the topology images together : fake_total and real_total.
            # if id_test==0: ## init the two variables
                # fake_total=fake_top
                # real_total=real_top
            # else: 
                # fake_total=np.concatenate((fake_total,fake_top),axis=0)
                # real_total=np.concatenate((real_total,real_top),axis=0)
            # print (fake_total.shape)
            # print (real_total.shape)
            # print (rvf_total)
            
            # sys.exit(0)
            
            ### result visualization ###  
            for num in np.arange(self.batch_size):
                # img1=np.squeeze(fake_top_2d[num])
                img1=np.reshape(fake_top[num],(self.height,self.width))
                img2=np.reshape(real_top[num],(self.height,self.width))
                img=np.concatenate((img1, img2), axis=0)
                visualization(img,'test_fake_real',id_test,num,image_save_path,save=1)
                
                print(np.shape(fake_top[num]))
                print(np.shape(test_data[num]))
 
                comp_fake = fem_compliance(fake_top[num],test_data[num],self.height,self.width)
                comp_real = fem_compliance(real_top[num],test_data[num],self.height,self.width)
                fake_comp.append(comp_fake)
                real_comp.append(comp_real)

        # sys.exit(0)
        # print(rvf_total.shape)
        plot_vf(image_save_path,rvf_total)
        plot_vf(image_save_path+'abs_',rvf_abs_total)
        np.save(image_save_path+'rvf_test', rvf_total)
        np.savetxt(image_save_path+'rvf_test.csv', rvf_total, delimiter = ',')
        
        compliance_test = np.stack((fake_comp,real_comp),axis=1)
        np.savetxt(image_save_path+'compliance_test.csv', compliance_test, delimiter = ',')
        

    def train_results_after_training(self, args):
        ## path of the folder to save visualization result (note:need '/' at the end )
        image_save_path='./test/train_results_after_training/'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        
        struc_save_path = image_save_path+'struc_real_fake.csv'
        if os.path.exists(struc_save_path):
            os.remove(struc_save_path)
        # saver=tf.train.import_meta_graph('./checkpoint/model_gan_unet_L1_10000_L2_1_3_6/model_gan_unet_L1_10000_L2_1-30.meta') #read the model
        # saver.restore(self.sess, "./checkpoint/model_gan_unet_L1_10000_L2_1_3_6/model_gan_unet_L1_10000_L2_1-30") #read the weights to model
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
        model_name_num = self.model_name + '-' + str(self.epoch_restore) # self.epoch_restore is the epoch number to restore
        model_path = os.path.join(checkpoint_dir, model_name_num)
        saver=tf.train.import_meta_graph(model_path+'.meta') # read the model
        saver.restore(self.sess, model_path) # read the weights to model
        
        fake_total=[]
        real_total=[]
        rvf_total=[]
        rvf_abs_total=[]
        fake_comp = []
        real_comp = []
        test_sample_size = max(1280,2*self.batch_size) # Here 640 is the minimum test sample size which should be larger than the batch_size
        all_data=self.get_sample_data(self.dataset_train_valid,test_sample_size)
        print(all_data.shape)
        #print ('volumn fraction',all_data[0,0],all_data[199,0],all_data[200,0],all_data[201,0])
        
        idx_num = int(np.ceil(np.shape(all_data)[0]//self.batch_size))
        
        for id_test in range(idx_num):
        # num_test=all_data.shape[0]
        # num_shuffle=np.random.choice(np.arange(num_test),self.batch_size)
        # test_shuffle=[all_data[i] for i in num_shuffle]
            starter = id_test*self.batch_size
            end = starter + self.batch_size
            feed_test = {self.real_data: all_data[starter:end]}
            mse, mae, fake_top_2d, real_top = self.sess.run([self.mse_sample, self.mae_sample,self.fake_B_sample,self.real_B],feed_dict=feed_test)
            
            fake_top = np.reshape(fake_top_2d, [-1, self.resolution])
            struc_real_fake = np.concatenate((real_top, fake_top), axis=1)
            df = pd.DataFrame(struc_real_fake)
            df.to_csv(image_save_path+'struc_real_fake.csv', mode='a', header=False)
            
            # print('MSE:',mse)
            # print('MAE:',mae)
            # print('Fake topology array shape:', fake_top.shape)
            # print('Reak topology array shape:', real_top.shape)
            # print(np.sum(fake_top[0,:]))
            # print(np.sum(real_top[0,:]))
            # sys.exit(0)
            vf_fake = np.array([np.sum(fake_top[i,:]) for i in np.arange(self.batch_size)])
            vf_true = np.array([np.sum(real_top[i,:]) for i in np.arange(self.batch_size)])
            # print(vf_fake)
            # print(vf_true)
            # sys.exit(0)
            re_vf = np.true_divide(np.subtract(vf_fake,vf_true),vf_true)
            re_vf_abs = np.true_divide(np.abs(np.subtract(vf_fake,vf_true)),vf_true)
            
            # print(re_vf)
            # sys.exit(0)
            rvf_total.extend(re_vf)
            rvf_abs_total.extend(re_vf_abs)
            
            # # combine all the topology images together : fake_total and real_total.
            # if id_test==0: ## init the two variables
                # fake_total=fake_top
                # real_total=real_top
            # else: 
                # fake_total=np.concatenate((fake_total,fake_top),axis=0)
                # real_total=np.concatenate((real_total,real_top),axis=0)
            # print (fake_total.shape)
            # print (real_total.shape)
            # print (rvf_total)
            
            # sys.exit(0)
            
            ### result visualization and compute the compliance for each image. 
            for num in np.arange(self.batch_size):
                # img1=np.squeeze(fake_top_2d[num])
                img1=np.reshape(fake_top[num],(self.height,self.width))
                img2=np.reshape(real_top[num],(self.height,self.width))
                img=np.concatenate((img1, img2), axis=0)
                visualization(img,'training_fake_real',id_test,num,image_save_path,save=1)
                
                comp_fake = fem_compliance(fake_top[num],all_data[num],self.height,self.width)
                comp_real = fem_compliance(real_top[num],all_data[num],self.height,self.width)
                fake_comp.append(comp_fake)
                real_comp.append(comp_real)
                
        # sys.exit(0)
        # print(rvf_total.shape)
        
        plot_vf(image_save_path,rvf_total)
        plot_vf(image_save_path+'abs_',rvf_abs_total)
        np.save(image_save_path+'rvf_train', rvf_total)
        np.savetxt(image_save_path+'rvf_train.csv', rvf_total, delimiter = ',')
        
        compliance_train = np.stack((fake_comp,real_comp),axis=1)
        np.savetxt(image_save_path+'compliance_train.csv', compliance_train, delimiter = ',')