# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:02:46 2019

@author: Andrew Huffman
"""

# Read text file with columns to import coordinates from FLAC
import pandas as pd
data = pd.read_csv('initial_coordinates.txt')
import math
import numpy as np

data_matrix = data.values
# Define Material Properties
v = 0.3
Bulk_mod = 10000 #kPa
E = (3*Bulk_mod*(1-2*v)) #kPa

# Define number of rows and columns in grid
dimensions = np.shape(data_matrix)
n = n_r = int(data_matrix[dimensions[0]-1][1]) # Number of rows
n_c = int(data_matrix[dimensions[0]-1][0]) # Number of columns

# Define number of nodes in first column
C = (5+(3*(n-2)))

# Define total number of nodes in grid
n_n = (n*n_c)+((n-1)*(n_c-1))

# Initialize Global Stifness Matrix
K_global = np.zeros(((2*n_n),(2*n_n)))
# Define lower-left and upper-right r and z coordinates for each zone

for i in range (1,(n_c)):
    for j in range (1,(n_r)):
        r_1 = data_matrix[(j-1)+(n*(i-1))][2]
        r_2 = data_matrix[j+((n)*i)][2]
        z_1 = data_matrix[(j-1)+((n)*(i-1))][3]
        z_2 = data_matrix[j+((n)*i)][3]

        # Subzone 1
    
        # Geometry
        r_i1 = r_1
        z_i1 = z_1
        r_j1 = r_2
        z_j1 = z_1
        r_m1 = r_1+((r_2-r_1)/2)
        z_m1 = z_1+((z_2-z_1)/2)
        alpha_i1 = (r_j1*z_m1)-(z_j1*r_m1)
        alpha_j1 = (r_m1*z_i1)-(z_m1*r_i1)
        alpha_m1 = (r_i1*z_j1)-(z_i1*r_j1)
        beta_i1 = z_j1-z_m1
        beta_j1 = z_m1-z_i1
        beta_m1 = z_i1-z_j1
        gamma_i1 = r_m1-r_j1
        gamma_j1 = r_i1-r_m1
        gamma_m1 = r_j1-r_i1
        r_bar1 = (r_1+r_2)/2
        z_bar1 = z_m1/3
        A_1 = (1/2)*(r_2-r_1)*(z_m1-z_i1)
        
        # Assemble B matrix and transpose of B
        B_1 = np.zeros((4,6))
        B_1[0][0] = beta_i1
        B_1[0][2] = beta_j1
        B_1[0][4] = beta_m1
        B_1[1][1] = gamma_i1
        B_1[1][3] = gamma_j1
        B_1[1][5] = gamma_m1
        B_1[2][0] = ((alpha_i1/r_bar1)+beta_i1+(gamma_i1*z_bar1/r_bar1))
        B_1[2][2] = ((alpha_j1/r_bar1)+beta_j1+(gamma_j1*z_bar1/r_bar1))
        B_1[2][4] = ((alpha_m1/r_bar1)+beta_m1+(gamma_m1*z_bar1/r_bar1))
        B_1[3][0] = gamma_i1
        B_1[3][1] = beta_i1
        B_1[3][2] = gamma_j1
        B_1[3][3] = beta_j1
        B_1[3][4] = gamma_m1
        B_1[3][5] = beta_m1
        B_1 = (1/(2*A_1))*B_1
        B_1T = np.transpose(B_1)
        
        # Assemble D matrix
        D_1 = np.zeros((4,4))
        D_1[0][0] = (1-v)
        D_1[0][1] = v
        D_1[0][2] = v
        D_1[1][0] = v
        D_1[1][1] = (1-v)
        D_1[1][2] = v
        D_1[2][0] = v
        D_1[2][1] = v
        D_1[2][2] = (1-v)
        D_1[3][3] = ((1-2*v)/2)
        D_1 = (E/((1+v)*(1-2*v)))*D_1
        
        
        # Assemble local stiffness matrix, k
        
        k_1 = 2*math.pi*r_bar1*A_1*np.matmul(np.matmul(B_1T,D_1),B_1)  #lb/in
        k_1g = np.zeros((10,10))  #subzone 1 matrix in global degrees of freedom
        # Fill in first row
        k_1g[0][0] = k_1[0][0]
        k_1g[0][1] = k_1[0][1]
        k_1g[0][2] = k_1[0][2]
        k_1g[0][3] = k_1[0][3]
        k_1g[0][8] = k_1[0][4]
        k_1g[0][9] = k_1[0][5]
        # Fill in second row
        k_1g[1][0] = k_1[1][0]
        k_1g[1][1] = k_1[1][1]
        k_1g[1][2] = k_1[1][2]
        k_1g[1][3] = k_1[1][3]
        k_1g[1][8] = k_1[1][4]
        k_1g[1][9] = k_1[1][5]
        # Fill in third row
        k_1g[2][0] = k_1[2][0]
        k_1g[2][1] = k_1[2][1]
        k_1g[2][2] = k_1[2][2]
        k_1g[2][3] = k_1[2][3]
        k_1g[2][8] = k_1[2][4]
        k_1g[2][9] = k_1[2][5]
        # Fill in the fourth row
        k_1g[3][0] = k_1[3][0]
        k_1g[3][1] = k_1[3][1]
        k_1g[3][2] = k_1[3][2]
        k_1g[3][3] = k_1[3][3]
        k_1g[3][8] = k_1[3][4]
        k_1g[3][9] = k_1[3][5]
        # Fill in the ninth row
        k_1g[8][0] = k_1[4][0]
        k_1g[8][1] = k_1[4][1]
        k_1g[8][2] = k_1[4][2]
        k_1g[8][3] = k_1[4][3]
        k_1g[8][8] = k_1[4][4]
        k_1g[8][9] = k_1[4][5]
        # Fill in the tenth row
        k_1g[9][0] = k_1[5][0]
        k_1g[9][1] = k_1[5][1]
        k_1g[9][2] = k_1[5][2]
        k_1g[9][3] = k_1[5][3]
        k_1g[9][8] = k_1[5][4]
        k_1g[9][9] = k_1[5][5]
        
        
        #Subzone 2
        
        # Geometry
        r_i2 = r_2
        z_i2 = z_1
        r_j2 = r_2
        z_j2 = z_2
        r_m2 = (r_1+r_2)/2
        z_m2 = (z_1+z_2)/2
        alpha_i2 = (r_j2*z_m2)-(z_j2*r_m2)
        alpha_j2 = (r_m2*z_i2)-(z_m2*r_i2)
        alpha_m2 = (r_i2*z_j2)-(z_i2*r_j2)
        beta_i2 = z_j2-z_m2
        beta_j2 = z_m2-z_i2
        beta_m2 = z_i2-z_j2
        gamma_i2 = r_m2-r_j2
        gamma_j2 = r_i2-r_m2
        gamma_m2 = r_j2-r_i2
        r_bar2 = r_i2-((1/3)*(r_i2-r_m2))
        z_bar2 = z_m2
        A_2 = (1/2)*(r_2-r_1)*(z_m2-z_i2)
        
        # Assemble B matrix and transpose of B
        B_2 = np.zeros((4,6))
        B_2[0][0] = beta_i2
        B_2[0][2] = beta_j2
        B_2[0][4] = beta_m2
        B_2[1][1] = gamma_i2
        B_2[1][3] = gamma_j2
        B_2[1][5] = gamma_m2
        B_2[2][0] = ((alpha_i2/r_bar2)+beta_i2+(gamma_i2*z_bar2/r_bar2))
        B_2[2][2] = ((alpha_j2/r_bar2)+beta_j2+(gamma_j2*z_bar2/r_bar2))
        B_2[2][4] = ((alpha_m2/r_bar2)+beta_m2+(gamma_m2*z_bar2/r_bar2))
        B_2[3][0] = gamma_i2
        B_2[3][1] = beta_i2
        B_2[3][2] = gamma_j2
        B_2[3][3] = beta_j2
        B_2[3][4] = gamma_m2
        B_2[3][5] = beta_m2
        B_2 = (1/(2*A_2))*B_2
        B_2T = np.transpose(B_2)
        
        # Assemble D matrix
        D_2 = np.zeros((4,4))
        D_2[0][0] = (1-v)
        D_2[0][1] = v
        D_2[0][2] = v
        D_2[1][0] = v
        D_2[1][1] = (1-v)
        D_2[1][2] = v
        D_2[2][0] = v
        D_2[2][1] = v
        D_2[2][2] = (1-v)
        D_2[3][3] = ((1-2*v)/2)
        D_2 = (E/((1+v)*(1-2*v)))*D_2
        
        # Assemble local stiffness matrix, k
        k_2 = 2*math.pi*r_bar2*A_2*np.matmul(np.matmul(B_2T,D_2),B_2)  #lb/in
        k_2g = np.zeros((10,10))  #subzone 2 matrix in global degrees of freedom
        # Fill in third row
        k_2g[2][2] = k_2[0][0]
        k_2g[2][3] = k_2[0][1]
        k_2g[2][4] = k_2[0][2]
        k_2g[2][5] = k_2[0][3]
        k_2g[2][8] = k_2[0][4]
        k_2g[2][9] = k_2[0][5]
        # Fill in fourth row
        k_2g[3][2] = k_2[1][0]
        k_2g[3][3] = k_2[1][1]
        k_2g[3][4] = k_2[1][2]
        k_2g[3][5] = k_2[1][3]
        k_2g[3][8] = k_2[1][4]
        k_2g[3][9] = k_2[1][5]
        # Fill in fifth row
        k_2g[4][2] = k_2[2][0]
        k_2g[4][3] = k_2[2][1]
        k_2g[4][4] = k_2[2][2]
        k_2g[4][5] = k_2[2][3]
        k_2g[4][8] = k_2[2][4]
        k_2g[4][9] = k_2[2][5]
        # Fill in the sixth row
        k_2g[5][2] = k_2[3][0]
        k_2g[5][3] = k_2[3][1]
        k_2g[5][4] = k_2[3][2]
        k_2g[5][5] = k_2[3][3]
        k_2g[5][8] = k_2[3][4]
        k_2g[5][9] = k_2[3][5]
        # Fill in the ninth row
        k_2g[8][2] = k_2[4][0]
        k_2g[8][3] = k_2[4][1]
        k_2g[8][4] = k_2[4][2]
        k_2g[8][5] = k_2[4][3]
        k_2g[8][8] = k_2[4][4]
        k_2g[8][9] = k_2[4][5]
        # Fill in the tenth row
        k_2g[9][2] = k_2[5][0]
        k_2g[9][3] = k_2[5][1]
        k_2g[9][4] = k_2[5][2]
        k_2g[9][5] = k_2[5][3]
        k_2g[9][8] = k_2[5][4]
        k_2g[9][9] = k_2[5][5]
        
        
        #Subzone 3
        
        # Geometry
        r_i3 = r_2
        z_i3 = z_2
        r_j3 = r_1
        z_j3 = z_2
        r_m3 = (r_1+r_2)/2
        z_m3 = (z_1+z_2)/2
        alpha_i3 = (r_j3*z_m3)-(z_j3*r_m3)
        alpha_j3 = (r_m3*z_i3)-(z_m3*r_i3)
        alpha_m3 = (r_i3*z_j3)-(z_i3*r_j3)
        beta_i3 = z_j3-z_m3
        beta_j3 = z_m3-z_i3
        beta_m3 = z_i3-z_j3
        gamma_i3 = r_m3-r_j3
        gamma_j3 = r_i3-r_m3
        gamma_m3 = r_j3-r_i3
        r_bar3 = (r_1+r_2)/2
        z_bar3 = z_i3-(1/3*(z_i3-z_m3))
        A_3 = (1/2)*abs((r_j3-r_i3))*abs((z_m3-z_i3))
        
        # Assemble B matrix and transpose of B
        B_3 = np.zeros((4,6))
        B_3[0][0] = beta_i3
        B_3[0][2] = beta_j3
        B_3[0][4] = beta_m3
        B_3[1][1] = gamma_i3
        B_3[1][3] = gamma_j3
        B_3[1][5] = gamma_m3
        B_3[2][0] = ((alpha_i3/r_bar3)+beta_i3+(gamma_i3*z_bar3/r_bar3))
        B_3[2][2] = ((alpha_j3/r_bar3)+beta_j3+(gamma_j3*z_bar3/r_bar3))
        B_3[2][4] = ((alpha_m3/r_bar3)+beta_m3+(gamma_m3*z_bar3/r_bar3))
        B_3[3][0] = gamma_i3
        B_3[3][1] = beta_i3
        B_3[3][2] = gamma_j3
        B_3[3][3] = beta_j3
        B_3[3][4] = gamma_m3
        B_3[3][5] = beta_m3
        B_3 = (1/(2*A_3))*B_3
        B_3T = np.transpose(B_3)
        
        # Assemble D matrix
        D_3 = np.zeros((4,4))
        D_3[0][0] = (1-v)
        D_3[0][1] = v
        D_3[0][2] = v
        D_3[1][0] = v
        D_3[1][1] = (1-v)
        D_3[1][2] = v
        D_3[2][0] = v
        D_3[2][1] = v
        D_3[2][2] = (1-v)
        D_3[3][3] = ((1-2*v)/2)
        D_3 = (E/((1+v)*(1-2*v)))*D_3
        
        # Assemble local stiffness matrix, k
        k_3 = 2*math.pi*r_bar3*A_3*np.matmul(np.matmul(B_3T,D_3),B_3)  #lb/in
        k_3g = np.zeros((10,10))  #subzone 3 matrix in global degrees of freedom
        # Fill in fifth row
        k_3g[4][4] = k_3[0][0]
        k_3g[4][5] = k_3[0][1]
        k_3g[4][6] = k_3[0][2]
        k_3g[4][7] = k_3[0][3]
        k_3g[4][8] = k_3[0][4]
        k_3g[4][9] = k_3[0][5]
        # Fill in sixth row
        k_3g[5][4] = k_3[1][0]
        k_3g[5][5] = k_3[1][1]
        k_3g[5][6] = k_3[1][2]
        k_3g[5][7] = k_3[1][3]
        k_3g[5][8] = k_3[1][4]
        k_3g[5][9] = k_3[1][5]
        # Fill in seventh row
        k_3g[6][4] = k_3[2][0]
        k_3g[6][5] = k_3[2][1]
        k_3g[6][6] = k_3[2][2]
        k_3g[6][7] = k_3[2][3]
        k_3g[6][8] = k_3[2][4]
        k_3g[6][9] = k_3[2][5]
        # Fill in the eighth row
        k_3g[7][4] = k_3[3][0]
        k_3g[7][5] = k_3[3][1]
        k_3g[7][6] = k_3[3][2]
        k_3g[7][7] = k_3[3][3]
        k_3g[7][8] = k_3[3][4]
        k_3g[7][9] = k_3[3][5]
        # Fill in the ninth row
        k_3g[8][4] = k_3[4][0]
        k_3g[8][5] = k_3[4][1]
        k_3g[8][6] = k_3[4][2]
        k_3g[8][7] = k_3[4][3]
        k_3g[8][8] = k_3[4][4]
        k_3g[8][9] = k_3[4][5]
        # Fill in the tenth row
        k_3g[9][4] = k_3[5][0]
        k_3g[9][5] = k_3[5][1]
        k_3g[9][6] = k_3[5][2]
        k_3g[9][7] = k_3[5][3]
        k_3g[9][8] = k_3[5][4]
        k_3g[9][9] = k_3[5][5]
        
        #Subzone 4
        
        # Geometry
        r_i4 = r_1
        z_i4 = z_2
        r_j4 = r_1
        z_j4 = z_1
        r_m4 = (r_1+r_2)/2
        z_m4 = (z_1+z_2)/2
        alpha_i4 = (r_j4*z_m4)-(z_j4*r_m4)
        alpha_j4 = (r_m4*z_i4)-(z_m4*r_i4)
        alpha_m4 = (r_i4*z_j4)-(z_i4*r_j4)
        beta_i4 = z_j4-z_m4
        beta_j4 = z_m4-z_i4
        beta_m4 = z_i4-z_j4
        gamma_i4 = r_m4-r_j4
        gamma_j4 = r_i4-r_m4
        gamma_m4 = r_j4-r_i4
        r_bar4 = r_i4+((1/3)*(r_m4-r_i4))
        z_bar4 = z_m4
        A_4 = (1/2)*abs((r_m4-r_i4))*abs((z_i4-z_j4))
        
        # Assemble B matrix and transpose of B
        B_4 = np.zeros((4,6))
        B_4[0][0] = beta_i4
        B_4[0][2] = beta_j4
        B_4[0][4] = beta_m4
        B_4[1][1] = gamma_i4
        B_4[1][3] = gamma_j4
        B_4[1][5] = gamma_m4
        B_4[2][0] = ((alpha_i4/r_bar4)+beta_i4+(gamma_i4*z_bar4/r_bar4))
        B_4[2][2] = ((alpha_j4/r_bar4)+beta_j4+(gamma_j4*z_bar4/r_bar4))
        B_4[2][4] = ((alpha_m4/r_bar4)+beta_m4+(gamma_m4*z_bar4/r_bar4))
        B_4[3][0] = gamma_i4
        B_4[3][1] = beta_i4
        B_4[3][2] = gamma_j4
        B_4[3][3] = beta_j4
        B_4[3][4] = gamma_m4
        B_4[3][5] = beta_m4
        B_4 = (1/(2*A_4))*B_4
        B_4T = np.transpose(B_4)
        
        # Assemble D matrix
        D_4 = np.zeros((4,4))
        D_4[0][0] = (1-v)
        D_4[0][1] = v
        D_4[0][2] = v
        D_4[1][0] = v
        D_4[1][1] = (1-v)
        D_4[1][2] = v
        D_4[2][0] = v
        D_4[2][1] = v
        D_4[2][2] = (1-v)
        D_4[3][3] = ((1-2*v)/2)
        D_4 = (E/((1+v)*(1-2*v)))*D_4
        
        # Assemble local stiffness matrix, k
        k_4 = 2*math.pi*r_bar4*A_4*np.matmul(np.matmul(B_4T,D_4),B_4)  #lb/in
        k_4g = np.zeros((10,10))  #subzone 4 matrix in global degrees of freedom
        # Fill in seventh row
        k_4g[6][6] = k_4[0][0]
        k_4g[6][7] = k_4[0][1]
        k_4g[6][0] = k_4[0][2]
        k_4g[6][1]= k_4[0][3]
        k_4g[6][8] = k_4[0][4]
        k_4g[6][9] = k_4[0][5]
        # Fill in eighth row
        k_4g[7][6] = k_4[1][0]
        k_4g[7][7] = k_4[1][1]
        k_4g[7][0] = k_4[1][2]
        k_4g[7][1] = k_4[1][3]
        k_4g[7][8] = k_4[1][4]
        k_4g[7][9] = k_4[1][5]
        # Fill in first row
        k_4g[0][6] = k_4[2][0]
        k_4g[0][7] = k_4[2][1]
        k_4g[0][0] = k_4[2][2]
        k_4g[0][1] = k_4[2][3]
        k_4g[0][8] = k_4[2][4]
        k_4g[0][9] = k_4[2][5]
        # Fill in the second row
        k_4g[1][6] = k_4[3][0]
        k_4g[1][7] = k_4[3][1]
        k_4g[1][0] = k_4[3][2]
        k_4g[1][1] = k_4[3][3]
        k_4g[1][8] = k_4[3][4]
        k_4g[1][9] = k_4[3][5]
        # Fill in the ninth row
        k_4g[8][6] = k_4[4][0]
        k_4g[8][7] = k_4[4][1]
        k_4g[8][0] = k_4[4][2]
        k_4g[8][1] = k_4[4][3]
        k_4g[8][8] = k_4[4][4]
        k_4g[8][9] = k_4[4][5]
        # Fill in the tenth row
        k_4g[9][6] = k_4[5][0]
        k_4g[9][7] = k_4[5][1]
        k_4g[9][0] = k_4[5][2]
        k_4g[9][1] = k_4[5][3]
        k_4g[9][8] = k_4[5][4]
        k_4g[9][9] = k_4[5][5]
        
        # Assemble element global stifness matrix, K_1 by superposition
        K_zone = (k_1g+k_2g+k_3g+k_4g)
        
        
        # Assemble global stiffness matrix from zone stiffness matrices
        if i == 1: 
            if j == 1:
                for l in range (0,10):
                    for m in range (0,10):
                        K_global[l][m] = (K_global[l][m]+K_zone[l][m])
           
            elif j == 2:
                K_global[6][6] = (K_global[6][6]+K_zone[0][0])
                K_global[6][7] = (K_global[6][7]+K_zone[0][1])
                K_global[6][4] = (K_global[6][4]+K_zone[0][2])
                K_global[6][5] = (K_global[6][5]+K_zone[0][3])
                K_global[6][10] = (K_global[6][10]+K_zone[0][4])
                K_global[6][11] = (K_global[6][11]+K_zone[0][5])
                K_global[6][12] = (K_global[6][12]+K_zone[0][6])
                K_global[6][13] = (K_global[6][13]+K_zone[0][7])
                K_global[6][14] = (K_global[6][14]+K_zone[0][8])
                K_global[6][15] = (K_global[6][15]+K_zone[0][9])
                
                K_global[7][6] = (K_global[7][6]+K_zone[1][0])
                K_global[7][7] = (K_global[7][7]+K_zone[1][1])
                K_global[7][4] = (K_global[7][4]+K_zone[1][2])
                K_global[7][5] = (K_global[7][5]+K_zone[1][3])
                K_global[7][10] = (K_global[7][10]+K_zone[1][4])
                K_global[7][11] = (K_global[7][11]+K_zone[1][5])
                K_global[7][12] = (K_global[7][12]+K_zone[1][6])
                K_global[7][13] = (K_global[7][13]+K_zone[1][7])
                K_global[7][14] = (K_global[7][14]+K_zone[1][8])
                K_global[7][15] = (K_global[7][15]+K_zone[1][9])  
                
                K_global[4][6] = (K_global[4][6]+K_zone[2][0])
                K_global[4][7] = (K_global[4][7]+K_zone[2][1])
                K_global[4][4] = (K_global[4][4]+K_zone[2][2])
                K_global[4][5] = (K_global[4][5]+K_zone[2][3])
                K_global[4][10] = (K_global[4][10]+K_zone[2][4])
                K_global[4][11] = (K_global[4][11]+K_zone[2][5])
                K_global[4][12] = (K_global[4][12]+K_zone[2][6])
                K_global[4][13] = (K_global[4][13]+K_zone[2][7])
                K_global[4][14] = (K_global[4][14]+K_zone[2][8])
                K_global[4][15] = (K_global[4][15]+K_zone[2][9])
                
                K_global[5][6] = (K_global[5][6]+K_zone[3][0])
                K_global[5][7] = (K_global[5][7]+K_zone[3][1])
                K_global[5][4] = (K_global[5][4]+K_zone[3][2])
                K_global[5][5] = (K_global[5][5]+K_zone[3][3])
                K_global[5][10] = (K_global[5][10]+K_zone[3][4])
                K_global[5][11] = (K_global[5][11]+K_zone[3][5])
                K_global[5][12] = (K_global[5][12]+K_zone[3][6])
                K_global[5][13] = (K_global[5][13]+K_zone[3][7])
                K_global[5][14] = (K_global[5][14]+K_zone[3][8])
                K_global[5][15] = (K_global[5][15]+K_zone[3][9])
                
                K_global[10][6] = (K_global[10][6]+K_zone[4][0])
                K_global[10][7] = (K_global[10][7]+K_zone[4][1])
                K_global[10][4] = (K_global[10][4]+K_zone[4][2])
                K_global[10][5] = (K_global[10][5]+K_zone[4][3])
                K_global[10][10] = (K_global[10][10]+K_zone[4][4])
                K_global[10][11] = (K_global[10][11]+K_zone[4][5])
                K_global[10][12] = (K_global[10][12]+K_zone[4][6])
                K_global[10][13] = (K_global[10][13]+K_zone[4][7])
                K_global[10][14] = (K_global[10][14]+K_zone[4][8])
                K_global[10][15] = (K_global[10][15]+K_zone[4][9])
                
                K_global[11][6] = (K_global[11][6]+K_zone[5][0])
                K_global[11][7] = (K_global[11][7]+K_zone[5][1])
                K_global[11][4] = (K_global[11][4]+K_zone[5][2])
                K_global[11][5] = (K_global[11][5]+K_zone[5][3])
                K_global[11][10] = (K_global[11][10]+K_zone[5][4])
                K_global[11][11] = (K_global[11][11]+K_zone[5][5])
                K_global[11][12] = (K_global[11][12]+K_zone[5][6])
                K_global[11][13] = (K_global[11][13]+K_zone[5][7])
                K_global[11][14] = (K_global[11][14]+K_zone[5][8])
                K_global[11][15] = (K_global[11][15]+K_zone[5][9])
                
                K_global[12][6] = (K_global[12][6]+K_zone[6][0])
                K_global[12][7] = (K_global[12][7]+K_zone[6][1])
                K_global[12][4] = (K_global[12][4]+K_zone[6][2])
                K_global[12][5] = (K_global[12][5]+K_zone[6][3])
                K_global[12][10] = (K_global[12][10]+K_zone[6][4])
                K_global[12][11] = (K_global[12][11]+K_zone[6][5])
                K_global[12][12] = (K_global[12][12]+K_zone[6][6])
                K_global[12][13] = (K_global[12][13]+K_zone[6][7])
                K_global[12][14] = (K_global[12][14]+K_zone[6][8])
                K_global[12][15] = (K_global[12][15]+K_zone[6][9])
                
                K_global[13][6] = (K_global[13][6]+K_zone[7][0])
                K_global[13][7] = (K_global[13][7]+K_zone[7][1])
                K_global[13][4] = (K_global[13][4]+K_zone[7][2])
                K_global[13][5] = (K_global[13][5]+K_zone[7][3])
                K_global[13][10] = (K_global[13][10]+K_zone[7][4])
                K_global[13][11] = (K_global[13][11]+K_zone[7][5])
                K_global[13][12] = (K_global[13][12]+K_zone[7][6])
                K_global[13][13] = (K_global[13][13]+K_zone[7][7])
                K_global[13][14] = (K_global[13][14]+K_zone[7][8])
                K_global[13][15] = (K_global[13][15]+K_zone[7][9]) 
                
                K_global[14][6] = (K_global[14][6]+K_zone[8][0])
                K_global[14][7] = (K_global[14][7]+K_zone[8][1])
                K_global[14][4] = (K_global[14][4]+K_zone[8][2])
                K_global[14][5] = (K_global[14][5]+K_zone[8][3])
                K_global[14][10] = (K_global[14][10]+K_zone[8][4])
                K_global[14][11] = (K_global[14][11]+K_zone[8][5])
                K_global[14][12] = (K_global[14][12]+K_zone[8][6])
                K_global[14][13] = (K_global[14][13]+K_zone[8][7])
                K_global[14][14] = (K_global[14][14]+K_zone[8][8])
                K_global[14][15] = (K_global[14][15]+K_zone[8][9])
                
                K_global[15][6] = (K_global[15][6]+K_zone[9][0])
                K_global[15][7] = (K_global[15][7]+K_zone[9][1])
                K_global[15][4] = (K_global[15][4]+K_zone[9][2])
                K_global[15][5] = (K_global[15][5]+K_zone[9][3])
                K_global[15][10] = (K_global[15][10]+K_zone[9][4])
                K_global[15][11] = (K_global[15][11]+K_zone[9][5])
                K_global[15][12] = (K_global[15][12]+K_zone[9][6])
                K_global[15][13] = (K_global[15][13]+K_zone[9][7])
                K_global[15][14] = (K_global[15][14]+K_zone[9][8])
                K_global[15][15] = (K_global[15][15]+K_zone[9][9])
            elif j > 2:
                K_global[2*(4+(3*(j-2)))-2][2*(4+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-2][2*(4+(3*(j-2)))-2]+K_zone[0][0])
                K_global[2*(4+(3*(j-2)))-2][2*(4+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-2][2*(4+(3*(j-2)))-1]+K_zone[0][1])
                K_global[2*(4+(3*(j-2)))-2][2*(3+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-2][2*(3+(3*(j-2)))-2]+K_zone[0][2])
                K_global[2*(4+(3*(j-2)))-2][2*(3+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-2][2*(3+(3*(j-2)))-1]+K_zone[0][3])
                K_global[2*(4+(3*(j-2)))-2][2*(7+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-2][2*(7+(3*(j-2)))-2]+K_zone[0][4])
                K_global[2*(4+(3*(j-2)))-2][2*(7+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-2][2*(7+(3*(j-2)))-1]+K_zone[0][5])
                K_global[2*(4+(3*(j-2)))-2][2*(6+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-2][2*(6+(3*(j-2)))-2]+K_zone[0][6])
                K_global[2*(4+(3*(j-2)))-2][2*(6+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-2][2*(6+(3*(j-2)))-1]+K_zone[0][7])
                K_global[2*(4+(3*(j-2)))-2][2*(8+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-2][2*(8+(3*(j-2)))-2]+K_zone[0][8])
                K_global[2*(4+(3*(j-2)))-2][2*(8+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-2][2*(8+(3*(j-2)))-1]+K_zone[0][9])

                K_global[2*(4+(3*(j-2)))-1][2*(4+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-1][2*(4+(3*(j-2)))-2]+K_zone[1][0])
                K_global[2*(4+(3*(j-2)))-1][2*(4+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-1][2*(4+(3*(j-2)))-1]+K_zone[1][1])
                K_global[2*(4+(3*(j-2)))-1][2*(3+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-1][2*(3+(3*(j-2)))-2]+K_zone[1][2])
                K_global[2*(4+(3*(j-2)))-1][2*(3+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-1][2*(3+(3*(j-2)))-1]+K_zone[1][3])
                K_global[2*(4+(3*(j-2)))-1][2*(7+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-1][2*(7+(3*(j-2)))-2]+K_zone[1][4])
                K_global[2*(4+(3*(j-2)))-1][2*(7+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-1][2*(7+(3*(j-2)))-1]+K_zone[1][5])
                K_global[2*(4+(3*(j-2)))-1][2*(6+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-1][2*(6+(3*(j-2)))-2]+K_zone[1][6])
                K_global[2*(4+(3*(j-2)))-1][2*(6+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-1][2*(6+(3*(j-2)))-1]+K_zone[1][7])
                K_global[2*(4+(3*(j-2)))-1][2*(8+(3*(j-2)))-2] = (K_global[2*(4+(3*(j-2)))-1][2*(8+(3*(j-2)))-2]+K_zone[1][8])
                K_global[2*(4+(3*(j-2)))-1][2*(8+(3*(j-2)))-1] = (K_global[2*(4+(3*(j-2)))-1][2*(8+(3*(j-2)))-1]+K_zone[1][9])  

                K_global[2*(3+(3*(j-2)))-2][2*(4+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-2][2*(4+(3*(j-2)))-2]+K_zone[2][0])
                K_global[2*(3+(3*(j-2)))-2][2*(4+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-2][2*(4+(3*(j-2)))-1]+K_zone[2][1])
                K_global[2*(3+(3*(j-2)))-2][2*(3+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-2][2*(3+(3*(j-2)))-2]+K_zone[2][2])
                K_global[2*(3+(3*(j-2)))-2][2*(3+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-2][2*(3+(3*(j-2)))-1]+K_zone[2][3])
                K_global[2*(3+(3*(j-2)))-2][2*(7+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-2][2*(7+(3*(j-2)))-2]+K_zone[2][4])
                K_global[2*(3+(3*(j-2)))-2][2*(7+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-2][2*(7+(3*(j-2)))-1]+K_zone[2][5])
                K_global[2*(3+(3*(j-2)))-2][2*(6+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-2][2*(6+(3*(j-2)))-2]+K_zone[2][6])
                K_global[2*(3+(3*(j-2)))-2][2*(6+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-2][2*(6+(3*(j-2)))-1]+K_zone[2][7])
                K_global[2*(3+(3*(j-2)))-2][2*(8+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-2][2*(8+(3*(j-2)))-2]+K_zone[2][8])
                K_global[2*(3+(3*(j-2)))-2][2*(8+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-2][2*(8+(3*(j-2)))-1]+K_zone[2][9])
                
                K_global[2*(3+(3*(j-2)))-1][2*(4+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-1][2*(4+(3*(j-2)))-2]+K_zone[3][0])
                K_global[2*(3+(3*(j-2)))-1][2*(4+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-1][2*(4+(3*(j-2)))-1]+K_zone[3][1])
                K_global[2*(3+(3*(j-2)))-1][2*(3+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-1][2*(3+(3*(j-2)))-2]+K_zone[3][2])
                K_global[2*(3+(3*(j-2)))-1][2*(3+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-1][2*(3+(3*(j-2)))-1]+K_zone[3][3])
                K_global[2*(3+(3*(j-2)))-1][2*(7+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-1][2*(7+(3*(j-2)))-2]+K_zone[3][4])
                K_global[2*(3+(3*(j-2)))-1][2*(7+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-1][2*(7+(3*(j-2)))-1]+K_zone[3][5])
                K_global[2*(3+(3*(j-2)))-1][2*(6+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-1][2*(6+(3*(j-2)))-2]+K_zone[3][6])
                K_global[2*(3+(3*(j-2)))-1][2*(6+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-1][2*(6+(3*(j-2)))-1]+K_zone[3][7])
                K_global[2*(3+(3*(j-2)))-1][2*(8+(3*(j-2)))-2] = (K_global[2*(3+(3*(j-2)))-1][2*(8+(3*(j-2)))-2]+K_zone[3][8])
                K_global[2*(3+(3*(j-2)))-1][2*(8+(3*(j-2)))-1] = (K_global[2*(3+(3*(j-2)))-1][2*(8+(3*(j-2)))-1]+K_zone[3][9])
            
                K_global[2*(7+(3*(j-2)))-2][2*(4+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-2][2*(4+(3*(j-2)))-2]+K_zone[4][0])
                K_global[2*(7+(3*(j-2)))-2][2*(4+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-2][2*(4+(3*(j-2)))-1]+K_zone[4][1])
                K_global[2*(7+(3*(j-2)))-2][2*(3+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-2][2*(3+(3*(j-2)))-2]+K_zone[4][2])
                K_global[2*(7+(3*(j-2)))-2][2*(3+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-2][2*(3+(3*(j-2)))-1]+K_zone[4][3])
                K_global[2*(7+(3*(j-2)))-2][2*(7+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-2][2*(7+(3*(j-2)))-2]+K_zone[4][4])
                K_global[2*(7+(3*(j-2)))-2][2*(7+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-2][2*(7+(3*(j-2)))-1]+K_zone[4][5])
                K_global[2*(7+(3*(j-2)))-2][2*(6+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-2][2*(6+(3*(j-2)))-2]+K_zone[4][6])
                K_global[2*(7+(3*(j-2)))-2][2*(6+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-2][2*(6+(3*(j-2)))-1]+K_zone[4][7])
                K_global[2*(7+(3*(j-2)))-2][2*(8+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-2][2*(8+(3*(j-2)))-2]+K_zone[4][8])
                K_global[2*(7+(3*(j-2)))-2][2*(8+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-2][2*(8+(3*(j-2)))-1]+K_zone[4][9])
                
                K_global[2*(7+(3*(j-2)))-1][2*(4+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-1][2*(4+(3*(j-2)))-2]+K_zone[5][0])
                K_global[2*(7+(3*(j-2)))-1][2*(4+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-1][2*(4+(3*(j-2)))-1]+K_zone[5][1])
                K_global[2*(7+(3*(j-2)))-1][2*(3+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-1][2*(3+(3*(j-2)))-2]+K_zone[5][2])
                K_global[2*(7+(3*(j-2)))-1][2*(3+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-1][2*(3+(3*(j-2)))-1]+K_zone[5][3])
                K_global[2*(7+(3*(j-2)))-1][2*(7+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-1][2*(7+(3*(j-2)))-2]+K_zone[5][4])
                K_global[2*(7+(3*(j-2)))-1][2*(7+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-1][2*(7+(3*(j-2)))-1]+K_zone[5][5])
                K_global[2*(7+(3*(j-2)))-1][2*(6+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-1][2*(6+(3*(j-2)))-2]+K_zone[5][6])
                K_global[2*(7+(3*(j-2)))-1][2*(6+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-1][2*(6+(3*(j-2)))-1]+K_zone[5][7])
                K_global[2*(7+(3*(j-2)))-1][2*(8+(3*(j-2)))-2] = (K_global[2*(7+(3*(j-2)))-1][2*(8+(3*(j-2)))-2]+K_zone[5][8])
                K_global[2*(7+(3*(j-2)))-1][2*(8+(3*(j-2)))-1] = (K_global[2*(7+(3*(j-2)))-1][2*(8+(3*(j-2)))-1]+K_zone[5][9])
                
                K_global[2*(6+(3*(j-2)))-2][2*(4+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-2][2*(4+(3*(j-2)))-2]+K_zone[6][0])
                K_global[2*(6+(3*(j-2)))-2][2*(4+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-2][2*(4+(3*(j-2)))-1]+K_zone[6][1])
                K_global[2*(6+(3*(j-2)))-2][2*(3+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-2][2*(3+(3*(j-2)))-2]+K_zone[6][2])
                K_global[2*(6+(3*(j-2)))-2][2*(3+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-2][2*(3+(3*(j-2)))-1]+K_zone[6][3])
                K_global[2*(6+(3*(j-2)))-2][2*(7+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-2][2*(7+(3*(j-2)))-2]+K_zone[6][4])
                K_global[2*(6+(3*(j-2)))-2][2*(7+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-2][2*(7+(3*(j-2)))-1]+K_zone[6][5])
                K_global[2*(6+(3*(j-2)))-2][2*(6+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-2][2*(6+(3*(j-2)))-2]+K_zone[6][6])
                K_global[2*(6+(3*(j-2)))-2][2*(6+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-2][2*(6+(3*(j-2)))-1]+K_zone[6][7])
                K_global[2*(6+(3*(j-2)))-2][2*(8+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-2][2*(8+(3*(j-2)))-2]+K_zone[6][8])
                K_global[2*(6+(3*(j-2)))-2][2*(8+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-2][2*(8+(3*(j-2)))-1]+K_zone[6][9])
                
                K_global[2*(6+(3*(j-2)))-1][2*(4+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-1][2*(4+(3*(j-2)))-2]+K_zone[7][0])
                K_global[2*(6+(3*(j-2)))-1][2*(4+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-1][2*(4+(3*(j-2)))-1]+K_zone[7][1])
                K_global[2*(6+(3*(j-2)))-1][2*(3+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-1][2*(3+(3*(j-2)))-2]+K_zone[7][2])
                K_global[2*(6+(3*(j-2)))-1][2*(3+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-1][2*(3+(3*(j-2)))-1]+K_zone[7][3])
                K_global[2*(6+(3*(j-2)))-1][2*(7+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-1][2*(7+(3*(j-2)))-2]+K_zone[7][4])
                K_global[2*(6+(3*(j-2)))-1][2*(7+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-1][2*(7+(3*(j-2)))-1]+K_zone[7][5])
                K_global[2*(6+(3*(j-2)))-1][2*(6+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-1][2*(6+(3*(j-2)))-2]+K_zone[7][6])
                K_global[2*(6+(3*(j-2)))-1][2*(6+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-1][2*(6+(3*(j-2)))-1]+K_zone[7][7])
                K_global[2*(6+(3*(j-2)))-1][2*(8+(3*(j-2)))-2] = (K_global[2*(6+(3*(j-2)))-1][2*(8+(3*(j-2)))-2]+K_zone[7][8])
                K_global[2*(6+(3*(j-2)))-1][2*(8+(3*(j-2)))-1] = (K_global[2*(6+(3*(j-2)))-1][2*(8+(3*(j-2)))-1]+K_zone[7][9])
                
                K_global[2*(8+(3*(j-2)))-2][2*(4+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-2][2*(4+(3*(j-2)))-2]+K_zone[8][0])
                K_global[2*(8+(3*(j-2)))-2][2*(4+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-2][2*(4+(3*(j-2)))-1]+K_zone[8][1])
                K_global[2*(8+(3*(j-2)))-2][2*(3+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-2][2*(3+(3*(j-2)))-2]+K_zone[8][2])
                K_global[2*(8+(3*(j-2)))-2][2*(3+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-2][2*(3+(3*(j-2)))-1]+K_zone[8][3])
                K_global[2*(8+(3*(j-2)))-2][2*(7+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-2][2*(7+(3*(j-2)))-2]+K_zone[8][4])
                K_global[2*(8+(3*(j-2)))-2][2*(7+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-2][2*(7+(3*(j-2)))-1]+K_zone[8][5])
                K_global[2*(8+(3*(j-2)))-2][2*(6+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-2][2*(6+(3*(j-2)))-2]+K_zone[8][6])
                K_global[2*(8+(3*(j-2)))-2][2*(6+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-2][2*(6+(3*(j-2)))-1]+K_zone[8][7])
                K_global[2*(8+(3*(j-2)))-2][2*(8+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-2][2*(8+(3*(j-2)))-2]+K_zone[8][8])
                K_global[2*(8+(3*(j-2)))-2][2*(8+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-2][2*(8+(3*(j-2)))-1]+K_zone[8][9])
                
                K_global[2*(8+(3*(j-2)))-1][2*(4+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-1][2*(4+(3*(j-2)))-2]+K_zone[9][0])
                K_global[2*(8+(3*(j-2)))-1][2*(4+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-1][2*(4+(3*(j-2)))-1]+K_zone[9][1])
                K_global[2*(8+(3*(j-2)))-1][2*(3+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-1][2*(3+(3*(j-2)))-2]+K_zone[9][2])
                K_global[2*(8+(3*(j-2)))-1][2*(3+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-1][2*(3+(3*(j-2)))-1]+K_zone[9][3])
                K_global[2*(8+(3*(j-2)))-1][2*(7+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-1][2*(7+(3*(j-2)))-2]+K_zone[9][4])
                K_global[2*(8+(3*(j-2)))-1][2*(7+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-1][2*(7+(3*(j-2)))-1]+K_zone[9][5])
                K_global[2*(8+(3*(j-2)))-1][2*(6+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-1][2*(6+(3*(j-2)))-2]+K_zone[9][6])
                K_global[2*(8+(3*(j-2)))-1][2*(6+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-1][2*(6+(3*(j-2)))-1]+K_zone[9][7])
                K_global[2*(8+(3*(j-2)))-1][2*(8+(3*(j-2)))-2] = (K_global[2*(8+(3*(j-2)))-1][2*(8+(3*(j-2)))-2]+K_zone[9][8])
                K_global[2*(8+(3*(j-2)))-1][2*(8+(3*(j-2)))-1] = (K_global[2*(8+(3*(j-2)))-1][2*(8+(3*(j-2)))-1]+K_zone[9][9])
                
        elif i == 2:
            if j == 1:
                K_global[2][2] = (K_global[2][2]+K_zone[0][0])
                K_global[2][3] = (K_global[2][3]+K_zone[0][1])
                K_global[2][(2*(C+1))-2] = (K_global[2][(2*(C+1))-2]+K_zone[0][2])
                K_global[2][(2*(C+1))-1] = (K_global[2][(2*(C+1))-1]+K_zone[0][3])
                K_global[2][(2*(C+2))-2] = (K_global[2][(2*(C+2))-2]+K_zone[0][4])
                K_global[2][(2*(C+2))-1] = (K_global[2][(2*(C+2))-1]+K_zone[0][5])
                K_global[2][4] = (K_global[2][4]+K_zone[0][6])
                K_global[2][5] = (K_global[2][5]+K_zone[0][7])
                K_global[2][(2*(C+3))-2] = (K_global[2][(2*(C+3))-2]+K_zone[0][8])
                K_global[2][(2*(C+3))-1] = (K_global[2][(2*(C+3))-1]+K_zone[0][9])
                
                K_global[3][2] = (K_global[3][2]+K_zone[1][0])
                K_global[3][3] = (K_global[3][3]+K_zone[1][1])
                K_global[3][(2*(C+1))-2] = (K_global[3][(2*(C+1))-2]+K_zone[1][2])
                K_global[3][(2*(C+1))-1] = (K_global[3][(2*(C+1))-1]+K_zone[1][3])
                K_global[3][(2*(C+2))-2] = (K_global[3][(2*(C+2))-2]+K_zone[1][4])
                K_global[3][(2*(C+2))-1] = (K_global[3][(2*(C+2))-1]+K_zone[1][5])
                K_global[3][4] = (K_global[3][4]+K_zone[1][6])
                K_global[3][5] = (K_global[3][5]+K_zone[1][7])
                K_global[3][(2*(C+3))-2] = (K_global[3][(2*(C+3))-2]+K_zone[1][8])
                K_global[3][(2*(C+3))-1] = (K_global[3][(2*(C+3))-1]+K_zone[1][9])
                
                K_global[(2*(C+1))-2][2] = (K_global[(2*(C+1))-2][2]+K_zone[2][0])
                K_global[(2*(C+1))-2][3] = (K_global[(2*(C+1))-2][3]+K_zone[2][1])
                K_global[(2*(C+1))-2][(2*(C+1))-2] = (K_global[(2*(C+1))-2][(2*(C+1))-2]+K_zone[2][2])
                K_global[(2*(C+1))-2][(2*(C+1))-1] = (K_global[(2*(C+1))-2][(2*(C+1))-1]+K_zone[2][3])
                K_global[(2*(C+1))-2][(2*(C+2))-2] = (K_global[(2*(C+1))-2][(2*(C+2))-2]+K_zone[2][4])
                K_global[(2*(C+1))-2][(2*(C+2))-1] = (K_global[(2*(C+1))-2][(2*(C+2))-1]+K_zone[2][5])
                K_global[(2*(C+1))-2][4] = (K_global[(2*(C+1))-2][4]+K_zone[2][6])
                K_global[(2*(C+1))-2][5] = (K_global[(2*(C+1))-2][5]+K_zone[2][7])
                K_global[(2*(C+1))-2][(2*(C+3))-2] = (K_global[(2*(C+1))-2][(2*(C+3))-2]+K_zone[2][8])
                K_global[(2*(C+1))-2][(2*(C+3))-1] = (K_global[(2*(C+1))-2][(2*(C+3))-1]+K_zone[2][9])
                
                K_global[(2*(C+1))-1][2] = (K_global[(2*(C+1))-1][2]+K_zone[3][0])
                K_global[(2*(C+1))-1][3] = (K_global[(2*(C+1))-1][3]+K_zone[3][1])
                K_global[(2*(C+1))-1][(2*(C+1))-2] = (K_global[(2*(C+1))-1][(2*(C+1))-2]+K_zone[3][2])
                K_global[(2*(C+1))-1][(2*(C+1))-1] = (K_global[(2*(C+1))-1][(2*(C+1))-1]+K_zone[3][3])
                K_global[(2*(C+1))-1][(2*(C+2))-2] = (K_global[(2*(C+1))-1][(2*(C+2))-2]+K_zone[3][4])
                K_global[(2*(C+1))-1][(2*(C+2))-1] = (K_global[(2*(C+1))-1][(2*(C+2))-1]+K_zone[3][5])
                K_global[(2*(C+1))-1][4] = (K_global[(2*(C+1))-1][4]+K_zone[3][6])
                K_global[(2*(C+1))-1][5] = (K_global[(2*(C+1))-1][5]+K_zone[3][7])
                K_global[(2*(C+1))-1][(2*(C+3))-2] = (K_global[(2*(C+1))-1][(2*(C+3))-2]+K_zone[3][8])
                K_global[(2*(C+1))-1][(2*(C+3))-1] = (K_global[(2*(C+1))-1][(2*(C+3))-1]+K_zone[3][9])
                
                K_global[(2*(C+2))-2][2] = (K_global[(2*(C+2))-2][2]+K_zone[4][0])
                K_global[(2*(C+2))-2][3] = (K_global[(2*(C+2))-2][3]+K_zone[4][1])
                K_global[(2*(C+2))-2][(2*(C+1))-2] = (K_global[(2*(C+2))-2][(2*(C+1))-2]+K_zone[4][2])
                K_global[(2*(C+2))-2][(2*(C+1))-1] = (K_global[(2*(C+2))-2][(2*(C+1))-1]+K_zone[4][3])
                K_global[(2*(C+2))-2][(2*(C+2))-2] = (K_global[(2*(C+2))-2][(2*(C+2))-2]+K_zone[4][4])
                K_global[(2*(C+2))-2][(2*(C+2))-1] = (K_global[(2*(C+2))-2][(2*(C+2))-1]+K_zone[4][5])
                K_global[(2*(C+2))-2][4] = (K_global[(2*(C+2))-2][4]+K_zone[4][6])
                K_global[(2*(C+2))-2][5] = (K_global[(2*(C+2))-2][5]+K_zone[4][7])
                K_global[(2*(C+2))-2][(2*(C+3))-2] = (K_global[(2*(C+2))-2][(2*(C+3))-2]+K_zone[4][8])
                K_global[(2*(C+2))-2][(2*(C+3))-1] = (K_global[(2*(C+2))-2][(2*(C+3))-1]+K_zone[4][9])
                
                K_global[(2*(C+2))-1][2] = (K_global[(2*(C+2))-1][2]+K_zone[5][0])
                K_global[(2*(C+2))-1][3] = (K_global[(2*(C+2))-1][3]+K_zone[5][1])
                K_global[(2*(C+2))-1][(2*(C+1))-2] = (K_global[(2*(C+2))-1][(2*(C+1))-2]+K_zone[5][2])
                K_global[(2*(C+2))-1][(2*(C+1))-1] = (K_global[(2*(C+2))-1][(2*(C+1))-1]+K_zone[5][3])
                K_global[(2*(C+2))-1][(2*(C+2))-2] = (K_global[(2*(C+2))-1][(2*(C+2))-2]+K_zone[5][4])
                K_global[(2*(C+2))-1][(2*(C+2))-1] = (K_global[(2*(C+2))-1][(2*(C+2))-1]+K_zone[5][5])
                K_global[(2*(C+2))-1][4] = (K_global[(2*(C+2))-1][4]+K_zone[5][6])
                K_global[(2*(C+2))-1][5] = (K_global[(2*(C+2))-1][5]+K_zone[5][7])
                K_global[(2*(C+2))-1][(2*(C+3))-2] = (K_global[(2*(C+2))-1][(2*(C+3))-2]+K_zone[5][8])
                K_global[(2*(C+2))-1][(2*(C+3))-1] = (K_global[(2*(C+2))-1][(2*(C+3))-1]+K_zone[5][9])
                
                K_global[4][2] = (K_global[4][2]+K_zone[6][0])
                K_global[4][3] = (K_global[4][3]+K_zone[6][1])
                K_global[4][(2*(C+1))-2] = (K_global[4][(2*(C+1))-2]+K_zone[6][2])
                K_global[4][(2*(C+1))-1] = (K_global[4][(2*(C+1))-1]+K_zone[6][3])
                K_global[4][(2*(C+2))-2] = (K_global[4][(2*(C+2))-2]+K_zone[6][4])
                K_global[4][(2*(C+2))-1] = (K_global[4][(2*(C+2))-1]+K_zone[6][5])
                K_global[4][4] = (K_global[4][4]+K_zone[6][6])
                K_global[4][5] = (K_global[4][5]+K_zone[6][7])
                K_global[4][(2*(C+3))-2] = (K_global[4][(2*(C+3))-2]+K_zone[6][8])
                K_global[4][(2*(C+3))-1] = (K_global[4][(2*(C+3))-1]+K_zone[6][9])
                
                K_global[5][2] = (K_global[5][2]+K_zone[7][0])
                K_global[5][3] = (K_global[5][3]+K_zone[7][1])
                K_global[5][(2*(C+1))-2] = (K_global[5][(2*(C+1))-2]+K_zone[7][2])
                K_global[5][(2*(C+1))-1] = (K_global[5][(2*(C+1))-1]+K_zone[7][3])
                K_global[5][(2*(C+2))-2] = (K_global[5][(2*(C+2))-2]+K_zone[7][4])
                K_global[5][(2*(C+2))-1] = (K_global[5][(2*(C+2))-1]+K_zone[7][5])
                K_global[5][4] = (K_global[5][4]+K_zone[7][6])
                K_global[5][5] = (K_global[5][5]+K_zone[7][7])
                K_global[5][(2*(C+3))-2] = (K_global[5][(2*(C+3))-2]+K_zone[7][8])
                K_global[5][(2*(C+3))-1] = (K_global[5][(2*(C+3))-1]+K_zone[7][9])
                
                K_global[(2*(C+3))-2][2] = (K_global[(2*(C+3))-2][2]+K_zone[8][0])
                K_global[(2*(C+3))-2][3] = (K_global[(2*(C+3))-2][3]+K_zone[8][1])
                K_global[(2*(C+3))-2][(2*(C+1))-2] = (K_global[(2*(C+3))-2][(2*(C+1))-2]+K_zone[8][2])
                K_global[(2*(C+3))-2][(2*(C+1))-1] = (K_global[(2*(C+3))-2][(2*(C+1))-1]+K_zone[8][3])
                K_global[(2*(C+3))-2][(2*(C+2))-2] = (K_global[(2*(C+3))-2][(2*(C+2))-2]+K_zone[8][4])
                K_global[(2*(C+3))-2][(2*(C+2))-1] = (K_global[(2*(C+3))-2][(2*(C+2))-1]+K_zone[8][5])
                K_global[(2*(C+3))-2][4] = (K_global[(2*(C+3))-2][4]+K_zone[8][6])
                K_global[(2*(C+3))-2][5] = (K_global[(2*(C+3))-2][5]+K_zone[8][7])
                K_global[(2*(C+3))-2][(2*(C+3))-2] = (K_global[(2*(C+3))-2][(2*(C+3))-2]+K_zone[8][8])
                K_global[(2*(C+3))-2][(2*(C+3))-1] = (K_global[(2*(C+3))-2][(2*(C+3))-1]+K_zone[8][9])
                
                K_global[(2*(C+3))-1][2] = (K_global[(2*(C+3))-1][2]+K_zone[9][0])
                K_global[(2*(C+3))-1][3] = (K_global[(2*(C+3))-1][3]+K_zone[9][1])
                K_global[(2*(C+3))-1][(2*(C+1))-2] = (K_global[(2*(C+3))-1][(2*(C+1))-2]+K_zone[9][2])
                K_global[(2*(C+3))-1][(2*(C+1))-1] = (K_global[(2*(C+3))-1][(2*(C+1))-1]+K_zone[9][3])
                K_global[(2*(C+3))-1][(2*(C+2))-2] = (K_global[(2*(C+3))-1][(2*(C+2))-2]+K_zone[9][4])
                K_global[(2*(C+3))-1][(2*(C+2))-1] = (K_global[(2*(C+3))-1][(2*(C+2))-1]+K_zone[9][5])
                K_global[(2*(C+3))-1][4] = (K_global[(2*(C+3))-1][4]+K_zone[9][6])
                K_global[(2*(C+3))-1][5] = (K_global[(2*(C+3))-1][5]+K_zone[9][7])
                K_global[(2*(C+3))-1][(2*(C+3))-2] = (K_global[(2*(C+3))-1][(2*(C+3))-2]+K_zone[9][8])
                K_global[(2*(C+3))-1][(2*(C+3))-1] = (K_global[(2*(C+3))-1][(2*(C+3))-1]+K_zone[9][9])
                
                
            
            elif j >= 2:
                K_global[(2*(3*(j-1))-2)][(2*(3*(j-1))-2)] = (K_global[(2*(3*(j-1))-2)][(2*(3*(j-1))-2)]+K_zone[0][0])
                K_global[(2*(3*(j-1))-2)][(2*(3*(j-1))-1)] = (K_global[(2*(3*(j-1))-2)][(2*(3*(j-1))-1)]+K_zone[0][1])
                K_global[(2*(3*(j-1))-2)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-2)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[0][2])
                K_global[(2*(3*(j-1))-2)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-2)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[0][3])
                K_global[(2*(3*(j-1))-2)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-2)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[0][4])
                K_global[(2*(3*(j-1))-2)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-2)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[0][5])
                K_global[(2*(3*(j-1))-2)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*(3*(j-1))-2)][((2*(6+(3*(j-2))))-2)]+K_zone[0][6])
                K_global[(2*(3*(j-1))-2)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*(3*(j-1))-2)][((2*(6+(3*(j-2))))-1)]+K_zone[0][7])
                K_global[(2*(3*(j-1))-2)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-2)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[0][8])
                K_global[(2*(3*(j-1))-2)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-2)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[0][9])
                
                K_global[(2*(3*(j-1))-1)][(2*(3*(j-1))-2)] = (K_global[(2*(3*(j-1))-1)][(2*(3*(j-1))-2)]+K_zone[1][0])
                K_global[(2*(3*(j-1))-1)][(2*(3*(j-1))-1)] = (K_global[(2*(3*(j-1))-1)][(2*(3*(j-1))-1)]+K_zone[1][1])
                K_global[(2*(3*(j-1))-1)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-1)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[1][2])
                K_global[(2*(3*(j-1))-1)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-1)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[1][3])
                K_global[(2*(3*(j-1))-1)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-1)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[1][4])
                K_global[(2*(3*(j-1))-1)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-1)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[1][5])
                K_global[(2*(3*(j-1))-1)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*(3*(j-1))-1)][((2*(6+(3*(j-2))))-2)]+K_zone[1][6])
                K_global[(2*(3*(j-1))-1)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*(3*(j-1))-1)][((2*(6+(3*(j-2))))-1)]+K_zone[1][7])
                K_global[(2*(3*(j-1))-1)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*(3*(j-1))-1)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[1][8])
                K_global[(2*(3*(j-1))-1)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*(3*(j-1))-1)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[1][9])
                
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(3*(j-1))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(3*(j-1))-2)]+K_zone[2][0])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(3*(j-1))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(3*(j-1))-1)]+K_zone[2][1])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[2][2])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[2][3])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[2][4])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[2][5])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-2)]+K_zone[2][6])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-1)]+K_zone[2][7])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[2][8])
                K_global[(2*((C+2)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[2][9])
                
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(3*(j-1))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(3*(j-1))-2)]+K_zone[3][0])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(3*(j-1))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(3*(j-1))-1)]+K_zone[3][1])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[3][2])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[3][3])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[3][4])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[3][5])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-2)]+K_zone[3][6])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-1)]+K_zone[3][7])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[3][8])
                K_global[(2*((C+2)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+2)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[3][9])
                
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(3*(j-1))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(3*(j-1))-2)]+K_zone[4][0])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(3*(j-1))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(3*(j-1))-1)]+K_zone[4][1])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[4][2])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[4][3])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[4][4])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[4][5])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-2)]+K_zone[4][6])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-1)]+K_zone[4][7])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[4][8])
                K_global[(2*((C+4)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[4][9])
                
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(3*(j-1))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(3*(j-1))-2)]+K_zone[5][0])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(3*(j-1))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(3*(j-1))-1)]+K_zone[5][1])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[5][2])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[5][3])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[5][4])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[5][5])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-2)]+K_zone[5][6])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-1)]+K_zone[5][7])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[5][8])
                K_global[(2*((C+4)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+4)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[5][9])
                
                K_global[(2*(6+(3*(j-2)))-2)][(2*(3*(j-1))-2)] = (K_global[(2*(6+(3*(j-2)))-2)][(2*(3*(j-1))-2)]+K_zone[6][0])
                K_global[(2*(6+(3*(j-2)))-2)][(2*(3*(j-1))-1)] = (K_global[(2*(6+(3*(j-2)))-2)][(2*(3*(j-1))-1)]+K_zone[6][1])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[6][2])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[6][3])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[6][4])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[6][5])
                K_global[(2*(6+(3*(j-2)))-2)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*(6+(3*(j-2))))-2)]+K_zone[6][6])
                K_global[(2*(6+(3*(j-2)))-2)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*(6+(3*(j-2))))-1)]+K_zone[6][7])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[6][8])
                K_global[(2*(6+(3*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[6][9])
                
                K_global[(2*(6+(3*(j-2)))-1)][(2*(3*(j-1))-2)] = (K_global[(2*(6+(3*(j-2)))-1)][(2*(3*(j-1))-2)]+K_zone[7][0])
                K_global[(2*(6+(3*(j-2)))-1)][(2*(3*(j-1))-1)] = (K_global[(2*(6+(3*(j-2)))-1)][(2*(3*(j-1))-1)]+K_zone[7][1])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[7][2])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[7][3])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[7][4])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[7][5])
                K_global[(2*(6+(3*(j-2)))-1)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*(6+(3*(j-2))))-2)]+K_zone[7][6])
                K_global[(2*(6+(3*(j-2)))-1)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*(6+(3*(j-2))))-1)]+K_zone[7][7])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[7][8])
                K_global[(2*(6+(3*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*(6+(3*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[7][9])
                
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(3*(j-1))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(3*(j-1))-2)]+K_zone[8][0])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(3*(j-1))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(3*(j-1))-1)]+K_zone[8][1])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[8][2])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[8][3])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[8][4])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[8][5])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-2)]+K_zone[8][6])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*(6+(3*(j-2))))-1)]+K_zone[8][7])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[8][8])
                K_global[(2*((C+5)+(2*(j-2)))-2)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-2)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[8][9])
                
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(3*(j-1))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(3*(j-1))-2)]+K_zone[9][0])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(3*(j-1))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(3*(j-1))-1)]+K_zone[9][1])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-2)]+K_zone[9][2])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+2)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+2)+(2*(j-2))))-1)]+K_zone[9][3])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-2)]+K_zone[9][4])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+4)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+4)+(2*(j-2))))-1)]+K_zone[9][5])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-2)]+K_zone[9][6])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*(6+(3*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*(6+(3*(j-2))))-1)]+K_zone[9][7])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-2)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-2)]+K_zone[9][8])
                K_global[(2*((C+5)+(2*(j-2)))-1)][(2*((C+5)+(2*(j-2)))-1)] = (K_global[(2*((C+5)+(2*(j-2)))-1)][((2*((C+5)+(2*(j-2))))-1)]+K_zone[9][9])
        elif i > 2:
            if j == 1:
                a1 = (2*((C+1)+(21*(i-3)))-2) 
                a2 = (2*((C+1)+(21*(i-3)))-1)
                b1 = (2*((C+22)+(21*(i-3)))-2)
                b2 = (2*((C+22)+(21*(i-3)))-1)
                c1 = (2*((C+23+(21*(i-3))))-2)
                c2 = (2*((C+23+(21*(i-3))))-1)
                d1 = (2*((C+2+(21*(i-3))))-2)
                d2 = (2*((C+2+(21*(i-3))))-1)
                e1 = (2*((C+24+(21*(i-3))))-2)
                e2 = (2*((C+24+(21*(i-3))))-1)

                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[0][0])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[0][1])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[0][2])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[0][3])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[0][4])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[0][5])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[0][6])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[0][7])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[0][8])
                K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[0][9])
                
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[1][0])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[1][1])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[1][2])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[1][3])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[1][4])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[1][5])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[1][6])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[1][7])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[1][8])
                K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+1)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[1][9])
                
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[2][0])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[2][1])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[2][2])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[2][3])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[2][4])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[2][5])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[2][6])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[2][7])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[2][8])
                K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[2][9])
                
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[3][0])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[3][1])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[3][2])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[3][3])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[3][4])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[3][5])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[3][6])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[3][7])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[3][8])
                K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+22)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[3][9])
                
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[4][0])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[4][1])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[4][2])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[4][3])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[4][4])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[4][5])
                K_global[(2*((C+23+(21*(i-3))))-2)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[4][6])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[4][7])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[4][8])
                K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[4][9])
                
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[5][0])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[5][1])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[5][2])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[5][3])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[5][4])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[5][5])
                K_global[(2*((C+23+(21*(i-3))))-1)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[5][6])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[5][7])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[5][8])
                K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+23)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[5][9])
                
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[6][0])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[6][1])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[6][2])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[6][3])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[6][4])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[6][5])
                K_global[(2*((C+2+(21*(i-3))))-2)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[6][6])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[6][7])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[6][8])
                K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[6][9])
                
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[7][0])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[7][1])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[7][2])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[7][3])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[7][4])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[7][5])
                K_global[(2*((C+2+(21*(i-3))))-1)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[7][6])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[7][7])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[7][8])
                K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+2)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[7][9])
                
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[8][0])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[8][1])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[8][2])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[8][3])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[8][4])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[8][5])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[8][6])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[8][7])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[8][8])
                K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-2)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[8][9])
                
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-2)]+K_zone[9][0])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+1)+(21*(i-3)))-1)]+K_zone[9][1])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-2)]+K_zone[9][2])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+22)+(21*(i-3)))-1)]+K_zone[9][3])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-2)]+K_zone[9][4])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+23+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+23)+(21*(i-3)))-1)]+K_zone[9][5])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-2)]+K_zone[9][6])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+2+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+2)+(21*(i-3)))-1)]+K_zone[9][7])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-2)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-2)]+K_zone[9][8])
                K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+24+(21*(i-3))))-1)] = (K_global[(2*((C+24)+(21*(i-3)))-1)][(2*((C+24)+(21*(i-3)))-1)]+K_zone[9][9])
                
            elif j >= 2:
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[0][0])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[0][1])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[0][2])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[0][3])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[0][4])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[0][5])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[0][6])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[0][7])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[0][8])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[0][9])

                 
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[1][0])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[1][1])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[1][2])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[1][3])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[1][4])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[1][5])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[1][6])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[1][7])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[1][8])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[1][9])
                    
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[2][0])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[2][1])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[2][2])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[2][3])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[2][4])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[2][5])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[2][6])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[2][7])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[2][8])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[2][9])

                    
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[3][0])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[3][1])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[3][2])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[3][3])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[3][4])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[3][5])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[3][6])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[3][7])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[3][8])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-2)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[3][9])
                
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[4][0])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[4][1])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[4][2])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[4][3])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[4][4])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[4][5])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[4][6])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[4][7])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[4][8])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[4][9])
                
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[5][0])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[5][1])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[5][2])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[5][3])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[5][4])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[5][5])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[5][6])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[5][7])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[5][8])
                K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[5][9])
                
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[6][0])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[6][1])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[6][2])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[6][3])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[6][4])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[6][5])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[6][6])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[6][7])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[6][8])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[6][9])
                
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[7][0])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[7][1])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[7][2])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[7][3])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[7][4])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[7][5])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[7][6])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[7][7])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[7][8])
                K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+2+(21*(i-3))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[7][9])
                
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[8][0])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[8][1])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[8][2])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[8][3])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[8][4])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[8][5])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[8][6])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[8][7])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[8][8])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-2)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[8][9])
                
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-2)]+K_zone[9][0])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-2)))-1)]+K_zone[9][1])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-2)]+K_zone[9][2])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-2)))-1)]+K_zone[9][3])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-2)]+K_zone[9][4])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-2))+(2*(j-1)))-1)]+K_zone[9][5])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-2)]+K_zone[9][6])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+2+(21*(i-3))+(2*(j-1)))-1)]+K_zone[9][7])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-2)]+K_zone[9][8])
                K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)] = (K_global[(2*(C+3+(21*(i-2))+(2*(j-1)))-1)][(2*(C+3+(21*(i-2))+(2*(j-1)))-1)]+K_zone[9][9])
               
K_globalex = pd.DataFrame (K_global/1000000)

# Save to xlsx file

filepath = 'Global_Stiffness_Matrix.xlsx'

K_globalex.to_excel(filepath, index=False)

# Read text file with columns to import coordinates from FLAC

undeformed = pd.read_csv('initial_coordinates.txt')
bc = pd.read_csv('new_boundary_coordinates.txt')
undeformed_matrix = undeformed.values
bc_matrix = bc.values
dimensions_bc = np.shape(bc_matrix)


# Define number of rows, columns, and boundary nodes in grid
D = (3+2*(n_r-2)) 
BC = dimensions_bc[0] 
F = np.zeros([len(K_global),1])
UC = n_n-BC # Number of Unconstrained Nodes

Total_nodes = n_n # Total number of nodes in system
# Initialize undeformed coordinates vector
coord_0 = np.zeros([len(K_global),1])

# Initialize deformed boundary nodes' coordinates vector
coord_bc = np.zeros([len(K_global),1])

# Initialize deformed coordinates vector
coord_new = np.zeros([len(K_global),1])


# Define Boundary Node degrees of freedom and coordinates and assemble Boundary Node and Unconstrained Node DOF vectors

#Initialize vector to store Boundary Node DOF's
DOF_BC = np.zeros([BC,1])
DOF = -1

#Left-side Boundary 
for x in range (0,n_r):
   if x == 0:
       node = 1
       r = bc_matrix[x][2]
       z = bc_matrix[x][3]
       coord_bc[(2*node)-2][0] = r
       coord_bc[(2*node)-1][0] = z
       DOF = DOF+1
       DOF_BC[DOF][0] = node
       
   else:
       node = (4+(3*(x-1)))
       r = bc_matrix[x][2]
       z = bc_matrix[x][3]
       coord_bc[(2*node)-2][0] = r
       coord_bc[(2*node)-1][0] = z
       DOF = DOF+1
       DOF_BC[DOF][0] = node
       

# Bottom Boundary
for x in range (1,(n_c-1)):
    if x == 1:
        node = 2
        r = bc_matrix[n_r][2]
        z = bc_matrix[n_r][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    elif x == 2:
        node = (C+1)
        r = bc_matrix[n_r+x-1][2]
        z = bc_matrix[n_r+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    elif x > 2:
        node = ((C+1)+(D*(x-2)))
        r = bc_matrix[n_r+x-1][2]
        z = bc_matrix[n_r+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
# Right-side Boundary
for x in range (1,(n_r+1)):
    if x == 1:
        node = (C+1+(D*(n_c-3))+(2*(x-1)))
        r = bc_matrix[(n_r+n_c-2)+x-1][2]
        z = bc_matrix[(n_r+n_c-2)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    elif x == 2:
        node = (C+1+(D*(n_c-3))+1)
        r = bc_matrix[(n_r+n_c-2)+x-1][2]
        z = bc_matrix[(n_r+n_c-2)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    elif x > 2:
        node = ((C+1+(D*(n_c-3))+1)+(2*(x-2)))
        r = bc_matrix[(n_r+n_c-2)+x-1][2]
        z = bc_matrix[(n_r+n_c-2)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node

# Top Boundary
for x in range (1, (n_c-1)):
    if x == 1:
        node = C-2
        r = bc_matrix[((2*(n-1))+n_c)+x-1][2]
        z = bc_matrix[((2*(n-1))+n_c)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    if x == 2:
        node = ((C-2)+(22*(x-1)))
        r = bc_matrix[((2*(n-1))+n_c)+x-1][2]
        z = bc_matrix[((2*(n-1))+n_c)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node
        
    if x > 2:
        node = (((C-2)+22)+(21*(x-2)))
        r = bc_matrix[((2*(n-1))+n_c)+x-1][2]
        z = bc_matrix[((2*(n-1))+n_c)+x-1][3]
        coord_bc[(2*node)-2][0] = r
        coord_bc[(2*node)-1][0] = z
        DOF = DOF+1
        DOF_BC[DOF][0] = node  # This vectore contains the node numbers for the Boundary Nodes
       
Deformed_Boundary_Coordinates = pd.DataFrame(coord_bc)

#Assemble Unconstrained Node's DOF vector
DOF = np.zeros([(BC+UC),1])
for i in range(0,len(DOF_BC)):
    k = int(DOF_BC[i][0])
    DOF[k-1][0] = k
for i in range(0,len(DOF)):
    if DOF[i][0] == 0:
        DOF[i][0] = ((DOF[i-1][0])+1)
M = np.isin(DOF,DOF_BC)
N = np.invert(M)
DOF_UC = np.multiply(DOF,N)
DOF_UC = np.sort(DOF_UC,0)
zeros = np.zeros([BC,1])
for i in range(0, BC):
    DOF_UC = np.delete(DOF_UC,0,0)  # This vector contains the node numbers for the Unconstrained Nodes


# Assemble undeformed coordinates vector
for i in range (1,n_c):
    for j in range (1,n_r):
        #Pull lower-left and upper-right, 1 and 2 respectively, r and z coordinates for each zone
        r_1 = undeformed_matrix[(j-1)+(n*(i-1))][2]
        r_2 = undeformed_matrix[j+(n)+(n*(i-1))][2]
        z_1 = undeformed_matrix[j-1+((n)*(i-1))][3]
        z_2 = undeformed_matrix[j+((n)*i)][3]
        
        # Calculate r and z coordinates of corners (a-d) and midpoint (e) for each zone
        a_r = r_1
        a_z = z_1
        b_r = r_2
        b_z = z_1
        c_r = r_2
        c_z = z_2
        d_r = r_1
        d_z = z_2
        e_r = (r_1+r_2)/2
        e_z = (z_1+z_2)/2
        
        # Calculate node number for corners and midpoint of each zone and store coordinates in vector according to DOF
        if i == 1:
            if j == 1:
                a = 1
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = 2
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = 3
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = 4
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = 5
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j == 2:
                a = 4
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = 3
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = 6
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = 7
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = 8
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j > 2:
                a = (4+(3*(j-2)))
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = (3+(3*(j-2)))
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = (6+(3*(j-2)))
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = (7+(3*(j-2)))
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = (8+(3*(j-2)))
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
        elif i == 2:
            if j == 1:
                a = 2
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = C+1
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = C+2
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = 3
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = C+3
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j == 2:
                a = 3
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = C+2
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = C+4
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = 6
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = C+5
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j >= 3:
                a = (3+(3*(j-2)))
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = ((C+2)+(2*(j-2)))
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = ((C+4)+(2*(j-2)))
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = (6+(3*(j-2)))
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = ((C+5)+(2*(j-2)))
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
        elif i > 2:
            if j == 1:
                a = ((C+1)+(D*(i-3)))
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = ((C+1)+(D*(i-2)))
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = ((C+2)+(D*(i-2)))
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = ((C+2)+(D*(i-3)))
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = ((C+3)+(D*(i-2)))
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j == 2:
                a = ((C+2)+(D*(i-3)))
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = ((C+2)+(D*(i-2)))
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = ((C+4)+(D*(i-2)))
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = ((C+4)+(D*(i-3)))
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = ((C+5)+(D*(i-2)))
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
            elif j >= 3:
                a = (((C+4)+(2*(j-3)))+(D*(i-3)))
                coord_0[(2*a)-2][0] = a_r
                coord_0[(2*a)-1][0] = a_z
                b = (((C+4)+(2*(j-3)))+(D*(i-2)))
                coord_0[(2*b)-2][0] = b_r
                coord_0[(2*b)-1][0] = b_z
                c = (((C+4)+(2*(j-2)))+(D*(i-2)))
                coord_0[(2*c)-2][0] = c_r
                coord_0[(2*c)-1][0] = c_z
                d = (((C+4)+(2*(j-2)))+(D*(i-3)))
                coord_0[(2*d)-2][0] = d_r
                coord_0[(2*d)-1][0] = d_z
                e = (((C+5)+(2*(j-2)))+(D*(i-2)))
                coord_0[(2*e)-2][0] = e_r
                coord_0[(2*e)-1][0] = e_z
Undeformed_Coordinates = pd.DataFrame(coord_0)

#Update deformed coordinates vector, i.e. calculate and store displacments at boundary DOF's
coord_new = coord_bc-coord_0 
Deformed_Coordinates = pd.DataFrame(coord_new)

# Calculate Displacements

# Initialize displacement vector
u = np.zeros([len(K_global),1])

# Calculate displacements for DOF's corresponding to Boundary Nodes
for i in range(0,BC):
    a = int(DOF_BC[i][0])
    u[(2*a)-2][0] = coord_bc[(2*a)-2][0]-coord_0[(2*a)-2][0]
    u[(2*a)-1][0] = coord_bc[(2*a)-1][0]-coord_0[(2*a)-1][0]


# Assemble modified, equivalent linear system

g = np.zeros([len(K_global),1])
for i in range(0,len(K_global)):
    g[i][0] = F[i][0]

dof_BC = np.zeros([2*BC,1])
for i in range(0,BC):
    if i==0:
        dof_BC[0][0] = 2*DOF_BC[i][0]-2
        dof_BC[1][0] = 2*DOF_BC[i][0]-1
    else:
        dof_BC[2*i][0] = 2*DOF_BC[i][0]-2
        dof_BC[(2*i)+1][0] = 2*DOF_BC[i][0]-1

for i in range(0,len(dof_BC)):
    a = int(dof_BC[i][0])
    for j in range(0,len(K_global)):
        g[j][0] = g[j][0]-(K_global[j][a]*u[a][0])
        K_global[a][j] = 0
        K_global[j][a] = 0
        K_global[a][a] = 1

for i in range(0,len(dof_BC)):
    a = int(dof_BC[i][0])
    g[a][0] = u[a][0]
    
# Calculate displacments from "equivalent" system
u_calc = np.linalg.solve(K_global,g)

coord_new = coord_0+u_calc

K_globalmodex = pd.DataFrame (K_global)
filepath = 'Global_Stiffness_Matrix_Equivalent.xlsx'
K_globalmodex.to_excel(filepath, index=False)


# Convert new coordinates vector into r and z columns
coord_r = np.zeros([int(len(K_global)/2),1])
coord_r0 = np.zeros([int(len(K_global)/2),1])
coord_z = np.zeros([int(len(K_global)/2),1])
coord_z0 = np.zeros([int(len(K_global)/2),1])
for i in range(0,int(len(K_global)/2)):
    if i == 0:
        r = coord_new[0][0]
        r0 = coord_0[0][0]
        z = coord_new[1][0]
        z0 = coord_0[1][0]
        coord_r[0][0] = r
        coord_r0[0][0] = r0
        coord_z[0][0] = z
        coord_z0[0][0] = z0
    else:
        r = coord_new[2*i][0]
        r0 = coord_0[2*i][0]
        z = coord_new[(2*i)+1][0]
        z0 = coord_0[(2*i)+1][0]
        coord_r[i][0] = r
        coord_r0[i][0] = r0
        coord_z[i][0] = z
        coord_z0[i][0] = z0
radial_coordinates = pd.DataFrame(coord_r)
filepath='Radial Coordinates.xlsx'
radial_coordinates.to_excel(filepath,index=False)
axial_coordinates = pd.DataFrame(coord_z)
filepath='Axial Coordinates.xlsx'
axial_coordinates.to_excel(filepath,index=False)


# Rearrange coordinates into FLAC format and write to txt file
r_FLAC = np.zeros([(n_r*n_c),1])
z_FLAC = np.zeros([(n_r*n_c),1])
a = -1
for i in range(0,n_c):
    for j in range(0,n_r):
        a = a+1
        if i == 0:
            node = int(DOF_BC[j][0]) # Left-edge boundary node of each row
            r_FLAC[a][0] = coord_new[(2*node)-2][0]
            z_FLAC[a][0] = coord_new[(2*node)-1][0]
        elif i == 1:
            if j == 0:
                node = 2
                r_FLAC[a][0] = coord_new[(2*node)-2][0]
                z_FLAC[a][0] = coord_new[(2*node)-1][0]
            elif j >= 1:
                node = 3*j
                r_FLAC[a][0] = coord_new[(2*node)-2][0]
                z_FLAC[a][0] = coord_new[(2*node)-1][0]
        elif i > 1:
            if j == 0:
                node = (C+1)+(D*(i-2))
                r_FLAC[a][0] = coord_new[(2*node)-2][0]
                z_FLAC[a][0] = coord_new[(2*node)-1][0]
            elif j > 0:
                node = (C+2)+(D*(i-2))+(2*(j-1))
                r_FLAC[a][0] = coord_new[(2*node)-2][0]
                z_FLAC[a][0] = coord_new[(2*node)-1][0]
coord_FLAC = np.zeros([(n_r*n_c),2])
for i in range(0,2):
    for j in range(0,len(r_FLAC)):
        if i == 0:
            coord_FLAC[j][i] = r_FLAC[j][0]
        elif i == 1:
            coord_FLAC[j][i] = z_FLAC[j][0]
coord_FLAC_df = pd.DataFrame(coord_FLAC)
csv = coord_FLAC_df.to_csv('updated_coordinates.txt') # Write csv file containing updated coordinates for FLAC to read