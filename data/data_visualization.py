# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:23:17 2018

@author: ZHENGUO
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



H = 64   #height node number
W = 128 #width node number

SN = H*W
SE = (H-1)*(W-1)
data = np.load("dataset_test.npy")
length_data = data.shape[0]
data = data.astype(np.float32)
#list(data.keys())

i = 500 # from 0 to length-1

output_shape = data[i,4*SN:4*SN+SE]
output_shape= np.reshape(output_shape,(H-1,W-1))
plt.imshow(output_shape,cmap='gray')
plt.gca().invert_yaxis()
plt.show()


intput_VF = data[i,0*SN:1*SN]
intput_VF = np.reshape(intput_VF,(H,W))
plt.imshow(intput_VF,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

intput_bd = data[i,1*SN:2*SN]
intput_bd = np.reshape(intput_bd,(H,W))
plt.imshow(intput_bd,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

intput_loadx = data[i,2*SN:3*SN]
intput_loadx= np.reshape(intput_loadx,(H,W))
plt.imshow(intput_loadx,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

intput_loady = data[i,3*SN:4*SN]
intput_loady= np.reshape(intput_loady,(H,W))
plt.imshow(intput_loady,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

output_shape = data[i,4*SN:4*SN+SE]
output_shape= np.reshape(output_shape,(H-1,W-1))
plt.imshow(output_shape,cmap='gray')
plt.gca().invert_yaxis()
plt.show()

UC_in_1d = data[i,4*SN+SE:6*SN+SE]
UC_in_2d = np.reshape(UC_in_1d,(H,W,2))
plt.imshow(UC_in_2d[:,:,0],cmap='jet')
plt.gca().invert_yaxis()
plt.show()

UC_mag_in_2d = np.sqrt(UC_in_2d[:,:,0]**2+UC_in_2d[:,:,1]**2)
plt.contour(UC_mag_in_2d,1000)
plt.show()

E_nodes_in_1d = data[i,6*SN+SE:9*SN+SE]
E_nodes_in_2d= np.reshape(E_nodes_in_1d,(H,W,3))
plt.imshow(E_nodes_in_2d[:,:,0],cmap='jet')
plt.gca().invert_yaxis()
plt.show()


#Stress
S_nodes_in_1d = data[i,9*SN+SE:12*SN+SE]
S_nodes_in_2d = np.reshape(S_nodes_in_1d,(H,W,3))
plt.imshow(S_nodes_in_2d[:,:,0],cmap='jet')
plt.gca().invert_yaxis()
plt.show()
#S_nodes_in_2d[:,:,0] is stress-xx
#S_nodes_in_2d[:,:,1] is stress-yy
#S_nodes_in_2d[:,:,2] is stress-xy
S_mises_in_2d = np.sqrt(S_nodes_in_2d[:,:,0]**2+S_nodes_in_2d[:,:,1]**2-
                        S_nodes_in_2d[:,:,0]*S_nodes_in_2d[:,:,1]+3*S_nodes_in_2d[:,:,2]**2)
plt.imshow(S_mises_in_2d,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

plt.contour(S_mises_in_2d,1000)
#plt.colorbar()
plt.show()


#for principle stress
S_max = 0.5*(S_nodes_in_2d[:,:,0]+S_nodes_in_2d[:,:,1])+np.sqrt(0.25*(S_nodes_in_2d[:,:,0]-S_nodes_in_2d[:,:,1])**2+S_nodes_in_2d[:,:,2]**2)
plt.imshow(S_max,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

plt.contour(np.flip(S_max,0),1000)
plt.show()

S_min = 0.5*(S_nodes_in_2d[:,:,0]+S_nodes_in_2d[:,:,1])-np.sqrt(0.25*(S_nodes_in_2d[:,:,0]-S_nodes_in_2d[:,:,1])**2+S_nodes_in_2d[:,:,2]**2)
plt.imshow(S_min,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

plt.contour(np.flip(S_min,0),1000)
plt.show()


#for load path
plt.quiver(S_nodes_in_2d[:,:,0],S_nodes_in_2d[:,:,1])
Ux = S_nodes_in_2d[:,:,0]/np.sqrt(S_nodes_in_2d[:,:,0]**2+S_nodes_in_2d[:,:,1]**2)
Vx = S_nodes_in_2d[:,:,1]/np.sqrt(S_nodes_in_2d[:,:,0]**2+S_nodes_in_2d[:,:,1]**2)
plt.quiver(Ux,Vx,linewidths=0.01)
plt.savefig('lpx.png', dpi = 1000)

Uy = S_nodes_in_2d[:,:,1]/np.sqrt(S_nodes_in_2d[:,:,1]**2+S_nodes_in_2d[:,:,2]**2)
Vy = S_nodes_in_2d[:,:,2]/np.sqrt(S_nodes_in_2d[:,:,1]**2+S_nodes_in_2d[:,:,2]**2)
plt.quiver(Uy,Vy,linewidths=0.01)
plt.savefig('lpy.png', dpi = 1000)

#plot strain energy
Strain_energy_2d = 0.5*(E_nodes_in_2d[:,:,0]*S_nodes_in_2d[:,:,0]+E_nodes_in_2d[:,:,0]*S_nodes_in_2d[:,:,0]+2*E_nodes_in_2d[:,:,0]*S_nodes_in_2d[:,:,0])
plt.imshow(Strain_energy_2d,cmap='jet')
plt.gca().invert_yaxis()
plt.show()

plt.contour(Strain_energy_2d,1000)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

UC_out_1d = data[i,12*SN+SE:14*SN+SE]
UC_out_2d = np.reshape(UC_out_1d,(H,W,2))
plt.imshow(UC_out_2d[:,:,0].reshape(H,W),cmap='jet')
plt.gca().invert_yaxis()
plt.show()

plt.imshow(UC_out_2d[:,:,1].reshape(H,W),cmap='jet')
plt.gca().invert_yaxis()
plt.show()



E_nodes_out_1d = data[i,14*SN+SE:17*SN+SE]
E_nodes_out_2d= np.reshape(E_nodes_out_1d,(H,W,3))
plt.imshow(E_nodes_out_2d[:,:,2],cmap='jet') #0:x; 1:y; 2：xy
plt.gca().invert_yaxis()
plt.show()

S_nodes_out_1d = data[i,17*SN+SE:20*SN+SE]
S_nodes_out_2d= np.reshape(S_nodes_out_1d,(H,W,3))
plt.imshow(S_nodes_out_2d[:,:,2],cmap='jet')
plt.gca().invert_yaxis()
plt.show()
