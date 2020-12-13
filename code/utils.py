from __future__ import division
import math
import os
import scipy.misc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
# import tensorflow as tf
import cv2
imageshow = 0 # 0 for just save, 1 for plot and save
if imageshow == 1:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
from datetime import datetime
import solidspy.preprocesor as pre
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol

def training_loss_writer(path, i, cur_time, model_name, cur_err_GAN_D_real, cur_err_GAN_D_fake, cur_err_GAN_D_total, cur_err_G_final_total, cur_err_GAN_G_loss, cur_vfmae, cur_mse, cur_mae):
    if i == 0:
        with open(path + '/train_record_' + model_name + '.txt', 'w+') as text_file:
            text_file.write('epoch {:03d}:'.format(i) + '    ')
            text_file.write('time_min: {:04f}'.format(cur_time) + '    ')
            text_file.write('train_GAN_Dloss_real: {:04f}'.format(cur_err_GAN_D_real) + '    ')
            text_file.write('train_GAN_Dloss_fake: {:04f}'.format(cur_err_GAN_D_fake) + '    ')
            text_file.write('train_GAN_Dloss: {:04f}'.format(cur_err_GAN_D_total) + '    ')
            text_file.write('train_G_whole_loss: {:04f}'.format(cur_err_G_final_total) + '    ')
            text_file.write('train_GAN_Gloss: {:04f}'.format(cur_err_GAN_G_loss) + '    ')
            text_file.write('train_vfmae: {:04f}'.format(cur_vfmae) + '    ')
            text_file.write('train_mse: {:04f}'.format(cur_mse) + '    ')
            text_file.write('train_mae: {:04f}'.format(cur_mae) + '    ')
        text_file.close()
    else:
        with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
            text_file.write('epoch {:03d}:'.format(i) + '    ')
            text_file.write('time_min: {:04f}'.format(cur_time) + '    ')
            text_file.write('train_GAN_Dloss_real: {:04f}'.format(cur_err_GAN_D_real) + '    ')
            text_file.write('train_GAN_Dloss_fake: {:04f}'.format(cur_err_GAN_D_fake) + '    ')
            text_file.write('train_GAN_Dloss: {:04f}'.format(cur_err_GAN_D_total) + '    ')
            text_file.write('train_G_whole_loss: {:04f}'.format(cur_err_G_final_total) + '    ')
            text_file.write('train_GAN_Gloss: {:04f}'.format(cur_err_GAN_G_loss) + '    ')
            text_file.write('train_vfmae: {:04f}'.format(cur_vfmae) + '    ')
            text_file.write('train_mse: {:04f}'.format(cur_mse) + '    ')
            text_file.write('train_mae: {:04f}'.format(cur_mae) + '    ')
        text_file.close()


def validation_loss_writer(path, model_name, cur_vfmae, cur_mse, cur_mae):
    with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
        text_file.write('validation_vfmae: {:04f}'.format(cur_vfmae) + '    ')
        text_file.write('validation_mse: {:04f}'.format(cur_mse) + '    ')
        text_file.write('validation_mae: {:04f}'.format(cur_mae) + '    ')
    text_file.close()


def testing_loss_writer(path, model_name, cur_vfmae, cur_mse, cur_mae):
    with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
        text_file.write('test_vfmae: {:04f}'.format(cur_vfmae) + '    ')
        text_file.write('test_mse: {:04f}'.format(cur_mse) + '    ')
        text_file.write('test_mae: {:04f}'.format(cur_mae) + '\n')
    text_file.close()


def calc_mse(fake,real):
    # row=fake.shape[0]
    # coln=fake.shape[1]
    # return np.true_divide(np.sum(np.power(np.linalg.norm(fake-real),2)),len(fake))
    return (np.square(fake - real)).mean()


def plot_vf(save_path,re_vf_total):
    re_vf_total_sort=np.sort(re_vf_total)
    plt.plot(np.arange(len(re_vf_total)),re_vf_total_sort)
    plt.plot(np.arange(len(re_vf_total)),np.zeros(len(re_vf_total)),'--')
    # plt.plot(np.arange(len(re_vf_total)),np.mean(re_vf_total)*np.ones(len(re_vf_total)),'r')
    plt.ylim([-0.5,0.5])
    plt.show()
    plt.savefig(save_path+'vf_distribution.png',dpi=200)
    plt.close()


def visualization(element_top,name,id_test,num,save_path,save=0):
    ### element_top: img to visualize 
    ### name: image name 
    ### id_test: batch_number
    ### num: the number of images in the id_test 
    ### save_path: where to save the visualziation data 
    ### save: save data or not; 1 for save 

    element_top=(1-element_top)*255
    element_top=element_top.astype(np.float32)
    img=cv2.resize(element_top,None,fx=5,fy=5)
    # print ('results',element_top)
    
    # cv2.imshow(name,img)
    # cv2.waitKey(0)
    if save==1:
        cv2.imwrite(save_path+str(id_test)+'_'+str(num)+'.png',img)

def one_index(i_H,j_W,H):
    return H*(j_W-1)+i_H

# index starting from zero for fem
def zero_index(i_H,j_W,H):
    return H*j_W+i_H

def writefiles(fem_path,topology,input_bc,input_loadx,input_loady):
    H = 64   #height node number
    W = 128 #width node number
    
    domain = np.where(topology > 0.5, 1, 0)
    domain_2d = np.reshape(domain,(H,W))
    domain_2d = domain_2d[0:H-1,0:W-1]
#    plt.imshow(topology_2d,cmap='gray')
#    plt.show()
    input_bc_2d = np.reshape(input_bc,(H,W))
#    plt.imshow(input_bc_2d,cmap='gray')
#    plt.show()
    input_loadx_2d = np.reshape(input_loadx,(H,W))
#    plt.imshow(input_loadx_2d,cmap='gray')
#    plt.show()
    input_loady_2d = np.reshape(input_loady,(H,W))
#    plt.imshow(input_loady_2d,cmap='gray')
#    plt.show()

    node_file = np.zeros([H*W,5])
    node_file[:,0] = range(H*W)
    for i in range(H):
        for j in range(W):
            node_file[H*j+i,1] = j
            node_file[H*j+i,2] = i
            if (input_bc_2d[i,j]==1 or input_bc_2d[i,j]==3):
                node_file[H*j+i,3] = -1
            if (input_bc_2d[i,j]==2 or input_bc_2d[i,j]==3):
                node_file[H*j+i,4] = -1
    np.savetxt(fem_path+"nodes.txt", node_file, fmt=("%d", "%.4f", "%.4f", "%d", "%d"))


    ele_array = np.zeros([(H-1)*(W-1),7], dtype=int)
    ele_array[:,0] = range((H-1)*(W-1))
    ele_array[:,1] = 1
    for i in range(H-1):
        for j in range(W-1):
            if domain_2d[i,j] == 1:
                ele_array[(H-1)*j+i,2] = 1
            ele_array[(H-1)*j+i,3] = H*j+i
            ele_array[(H-1)*j+i,4] = H*(j+1)+i
            ele_array[(H-1)*j+i,5] = H*(j+1)+i+1
            ele_array[(H-1)*j+i,6] = H*j+i+1
    np.savetxt(fem_path+"eles.txt", ele_array, fmt=("%d", "%d", "%d", "%d", "%d", "%d","%d"))


    load_file = np.array([])
    for i in range(H):
        for j in range(W):
            if (input_loadx_2d[i,j]**2 + input_loady_2d[i,j]**2 >0.9):
                load_add = np.array([zero_index(i,j,H), input_loadx_2d[i,j], input_loady_2d[i,j]])
                load_file = np.append(load_file, load_add, axis=0)
#                load_add = np.array([zero_index(i,j,H)-1, input_loadx_2d[i,j], input_loady_2d[i,j]])
#                load_file = np.append(load_file, load_add, axis=0)
#                load_add = np.array([zero_index(i,j,H)+1, input_loadx_2d[i,j], input_loady_2d[i,j]])
#                load_file = np.append(load_file, load_add, axis=0)
    load_file = np.reshape(load_file,(-1,3))
    np.savetxt(fem_path+"loads.txt", load_file, fmt=("%0d", "%.4f", "%.4f"))


    mater_file = np.array([[1e-3, 0.3],[1, 0.3]])
    np.savetxt(fem_path+"mater.txt", mater_file, fmt=("%.3f", "%.3f"))

def mysolidspy(path):
    compute_strains = True
    plot_contours = False

    start_time = datetime.now()
    nodes, mats, elements, loads = pre.readin(folder=path)

    # Pre-processing
    DME , IBC , neq = ass.DME(nodes, elements)
    print("Number of nodes: {}".format(nodes.shape[0]))
    print("Number of elements: {}".format(elements.shape[0]))
    print("Number of equations: {}".format(neq))


    # System assembly
    KG = ass.assembler(elements, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)

    # System solution
    UG = sol.static_sol(KG, RHSG)
    if not(np.allclose(KG.dot(UG)/KG.max(), RHSG/KG.max())):
        print("The system is not in equilibrium!")
    end_time = datetime.now()
    print('Duration for system solution: {}'.format(end_time - start_time))

    # Post-processing
    #start_time = datetime.now()
    UC = pos.complete_disp(IBC, nodes, UG)
    E_nodes, S_nodes = None, None
    if compute_strains:
        E_nodes, S_nodes = pos.strain_nodes(nodes, elements, mats, UC)
    if plot_contours:
        pos.fields_plot(elements, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)
    #end_time = datetime.now()
    #print('Duration for post processing: {}'.format(end_time - start_time))
    #print('Analysis terminated successfully!')

    return (UC, E_nodes, S_nodes) if compute_strains else UC


def fem_compliance(topology,dataset,height,width):
    resolution = height * width
    fem_path = './test/fem/'
    if not os.path.exists(fem_path):
        os.makedirs(fem_path)
    input_bc = np.reshape(dataset[resolution*4:resolution*5],(height,width))
    input_loadx = np.reshape(dataset[resolution*5:resolution*6],(height,width))
    input_loady = np.reshape(dataset[resolution*6:resolution*7],(height,width))
    writefiles(fem_path,topology,input_bc,input_loadx,input_loady)
    UC_in, E_nodes_in, S_nodes_in = mysolidspy(fem_path)
    return np.sum(np.multiply(E_nodes_in, S_nodes_in))
    #return resolution