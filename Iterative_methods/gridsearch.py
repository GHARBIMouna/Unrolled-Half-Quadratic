import numpy as np
import time
import matplotlib.pyplot as plt
import math
import torch 
import torch.nn as nn
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--Mode', type=str,required=True, help="SC,ES")
parser.add_argument('--function', type=str,required=True, help="Hybrid or test, convex, non-convex or hybrid, double check parameters.")
parser.add_argument('--Dataset', type=str,required=True, help="Dataset name")
parser.add_argument('--file_name', type=str,required=False, help= "file_name to save gridsearch results")
args = parser.parse_args()





def SNR(x_true,x_pred):
    return 20*torch.log10(torch.linalg.norm(x_true)/torch.linalg.norm(x_true-x_pred))

def TSNR(x_true,x_pred):
    return 20*torch.log10(torch.linalg.norm(x_true[x_true!=0])/torch.linalg.norm(x_true[x_true!=0]-x_pred[x_true!=0]))


"""Begin Fair potential"""
def phi_s(input, delta_s):
    y = delta_s * input / (abs(input) + delta_s)
    return y
def omega_s(u, delta_s):
    return delta_s / (torch.abs(u) + delta_s)
def psi_s(input, delta):
    return delta * (torch.abs(input) - delta * torch.log(torch.abs(input) / delta + 1))
"""End Fair potential"""


"""Cauchy penalization"""
def phi_s1(input,delta_s):
    return (input*delta_s**2)/(delta_s**2+input**2)
def psi_s1(input,delta_s):
    return 0.5*delta_s**2*torch.log(1+(input**2)/delta_s**2)
def omega_s1(input,delta_s):
    return (delta_s**2)/(delta_s**2+input**2)
"""End Cauchy penalization"""




def MM_quadratic(delta_s,lamda,delta_s1,lamda1,H,L,x_true,x_degraded,x0,N,p,iterations,mode):
    x0_b=x0
    MSE=[]
    mse=loss(x0_b,x_true)
    MSE.append(mse)
    number_iterations=1
    diag=torch.diag(omega_s(torch.mm(L,x0_b).squeeze(),delta_s))
    diag1=torch.diag(omega_s1(torch.mm(L,x0_b).squeeze(),delta_s1))
    a=torch.mm(torch.transpose(H,0,1),H)+  lamda*torch.mm(torch.mm(torch.transpose(L,0,1),diag),L) +lamda1*torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L)
    grad=torch.mm(torch.transpose(H,0,1),torch.mm(H,x0_b)-x_degraded)+ lamda*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag),L),x0_b) +lamda1*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L),x0_b)
    bk=torch.mm(torch.inverse(a),grad)
    x1_b=x0_b-bk
    mse=loss(x1_b,x_true)
    MSE.append(mse)
    if mode=="SC":
        while  (((torch.linalg.norm(x1_b-x0_b)**2)/(torch.linalg.norm(x0_b)**2))>1e-5) & (number_iterations<iterations):
            diag=torch.diag(omega_s(torch.mm(L,x1_b).squeeze(),delta_s))
            diag1=torch.diag(omega_s1(torch.mm(L,x1_b).squeeze(),delta_s1))
            a=torch.mm(torch.transpose(H,0,1),H)+  lamda*torch.mm(torch.mm(torch.transpose(L,0,1),diag),L)+lamda1*torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L)
            grad=torch.mm(torch.transpose(H,0,1),torch.mm(H,x1_b)-x_degraded)+ lamda*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag),L),x1_b)+  lamda1*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L),x1_b)
            bk=torch.mm(torch.inverse(a),grad)
            x0_b=x1_b
            x1_b=x1_b-bk
            mse=loss(x1_b,x_true)
            MSE.append(mse)
            number_iterations=number_iterations+1
    elif mode=="ES":    
        while (number_iterations<iterations):
            diag=torch.diag(omega_s(torch.mm(L,x1_b).squeeze(),delta_s))
            diag1=torch.diag(omega_s1(torch.mm(L,x1_b).squeeze(),delta_s1))
            a=torch.mm(torch.transpose(H,0,1),H)+  lamda*torch.mm(torch.mm(torch.transpose(L,0,1),diag),L)+lamda1*torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L)
            grad=torch.mm(torch.transpose(H,0,1),torch.mm(H,x1_b)-x_degraded)+ lamda*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag),L),x1_b)+  lamda1*torch.mm(torch.mm(torch.mm(torch.transpose(L,0,1),diag1),L),x1_b)
            bk=torch.mm(torch.inverse(a),grad)
            x0_b=x1_b
            x1_b=x1_b-bk
            mse=loss(x1_b,x_true)
            MSE.append(mse)
            number_iterations=number_iterations+1

    snr=SNR(x_true,x1_b)
    tsnr=TSNR(x_true,x1_b)
    return(MSE,x1_b,snr,tsnr,number_iterations)



N=2000 #Groudtruth signal size
p=2049 #Degraded signal size
L=torch.eye(N).type(torch.cuda.DoubleTensor)
loss=nn.MSELoss()

#Gridsearch
if args.function=="Gridsearch_cvx":
    num_signals=200
    deltas_cvx=[0.01,0.1,0.5,1,2,4,6,8]
    lamdas_cvx=[0.0,0.01,0.1,1,2,4,6,8,10,20]
    lamda_ncvx=0.0
    delta_ncvx=0.01 #not evaluated

if args.function=="Gridsearch_ncvx":
    num_signals=200
    deltas_ncvx=[0.01,0.1,0.5,1,2,4,6,8]
    lamdas_ncvx=[0.0,0.01,0.1,1,2,4,6,8,10,20]
    lamda_cvx=0.0
    delta_cvx=0.01 #not evaluated

if args.function=="Gridsearch_Hybrid":
    num_signals=200
    print("Make sure that the retained delta values are correct manually.")
    lamdas_ncvx=[0.0,0.01,0.1,1,2,4,6,8,10,20]
    lamdas_cvx=[0.0,0.01,0.1,1,2,4,6,8,10,20]
    delta_cvx=0.01
    delta_ncvx=0.1 

if args.function=="Test_cvx":
    print("Input manually values to be tested")
    lamda_cvx=20
    lamda_ncvx=0.0
    delta_cvx=0.01
    delta_ncvx=0.1 #Not evaluated
    num_signals=100

if args.function=="Test_ncvx":
    print("Input manually values to be tested")
    lamda_ncvx=20
    lamda_cvx=0.0
    delta_ncvx=0.01
    delta_cvx=0.1 #Not evaluated
    num_signals=100


if args.function=="Test_Hybrid":
    print("Input manually values to be tested")
    lamda_ncvx=20
    lamda_cvx=20.0
    delta_ncvx=0.01
    delta_cvx=0.1 
    num_signals=100



if args.mode=="SC":
    iterations=100
elif args.mode =="ES":
    print("Input manually the number of iterations for ES.")
    iterations =8


mse_min=1000000
for lamda_cvx in lamdas_cvx:
    for lamda_ncvx in lamdas_ncvx:
        for delta_s_ncvx in deltas_ncvx:
            for delta_s_cvx in deltas_cvx:
                
                print('lamda_cvx=',lamda_cvx)
                print('lamda_ncvx=',lamda_ncvx)
                print('delta_s_cvx=',delta_s_cvx)
                print('delta_s_ncvx=',delta_s_ncvx)
                MSE_avg=0
                iter_avg=0
                time_avg=0
                snr_avg=0
                tsnr_avg=0
                MSE_list=[]
                SNR_list=[]
                TSNR_list=[]

                for i in range(num_signals):
                    if args.function in ["Gridsearch_cvx", "Gridsearch_ncvx", "Gridsearch_Hybrid"]:
                        H=np.load(os.path.join('Datasets'+args.Dataset,'training','H','H_tr_'+str(i)+'.npy'),allow_pickle=True)
                        H=torch.from_numpy(H).type(torch.cuda.DoubleTensor)
                        x_true=torch.unsqueeze(torch.tensor(np.load(os.path.join('Datasets',args.Dataset,'training','Groundtruth','x_Gr_tr_'+str(i)+'.npy'),allow_pickle=True)),1).type(torch.cuda.DoubleTensor)
                        x_degraded=torch.unsqueeze(torch.tensor(np.load(os.path.join('Datasets'+args.Dataset+'training','Degraded','x_De_tr_'+str(i)+'.npy'),allow_pickle=True)),1).type(torch.cuda.DoubleTensor)
                    if args.function in ["Test_cvx","Test_ncvx","Test_Hybrid"]:
                        H=np.load(os.path.join('Datasets'+args.Dataset,'test','H','H_te_'+str(i)+'.npy'),allow_pickle=True)
                        H=torch.from_numpy(H).type(torch.cuda.DoubleTensor)
                        x_true=torch.unsqueeze(torch.tensor(np.load(os.path.join('Datasets',args.Dataset,'test','Groundtruth','x_Gr_te_'+str(i)+'.npy'),allow_pickle=True)),1).type(torch.cuda.DoubleTensor)
                        x_degraded=torch.unsqueeze(torch.tensor(np.load(os.path.join('Datasets'+args.Dataset+'test','Degraded','x_De_te_'+str(i)+'.npy'),allow_pickle=True)),1).type(torch.cuda.DoubleTensor)

                        

                   
                    x0=torch.zeros((2000,1)).type(torch.cuda.DoubleTensor) 
                    t0=time.time()
                    MSE_values,x_pred,snr,tsnr,num_iterations=MM_quadratic(delta_s_cvx,lamda_cvx,delta_s_ncvx,lamda_ncvx,H,L,x_true,x_degraded,x0,N,p,iterations,args.mode)
                    t1=time.time()
                    MSE_list.append(MSE_values[-1])
                    SNR_list.append(snr)
                    TSNR_list.append(tsnr)
                    MSE_avg+=MSE_values[-1]
                    iter_avg+=num_iterations
                    time_avg+=t1-t0
                    snr_avg+=snr
                    tsnr_avg+=tsnr
                    
                    

                     

                MSE_avg=MSE_avg/num_signals
                iter_avg=iter_avg/num_signals
                time_avg=time_avg/num_signals
                snr_avg=snr_avg/num_signals
                tsnr_avg=tsnr_avg/num_signals
                
                
                #compute metrics STD
                mse_std=0
                for l in MSE_list:
                    mse_std=mse_std+((l-MSE_avg)**2)
                mse_std=torch.sqrt(mse_std/(num_signals-1))
                
                snr_std=0
                for l in SNR_list:
                    snr_std=snr_std+((l-snr_avg)**2)
                snr_std=torch.sqrt(snr_std/(num_signals-1))
                
                tsnr_std=0
                for l in TSNR_list:
                    tsnr_std=tsnr_std+((l-tsnr_avg)**2)
                tsnr_std=torch.sqrt(tsnr_std/(num_signals-1))
                ##########
                
                
                
                
                if MSE_avg < mse_min:
                    lamda_cvx_min=lamda_cvx
                    delta_s_cvx_min=delta_s_cvx
                    lamda_ncvx_min=lamda_ncvx
                    delta_s_ncvx_min=delta_s_ncvx
                    mse_min=MSE_avg
                    iter_min=iter_avg
                    time_min=time_avg
                    snr_min=snr_avg
                    tsnr_min=tsnr_avg
                    mse_std_min=mse_std
                    snr_std_min=snr_std
                    tsnr_std_min=tsnr_std
                print("average MSE:",float(MSE_avg))
                print("MSE std is:", float(mse_std))
                print("time_avg", float(time_avg))
                print("iter_avg",float(iter_avg))
                print("snr_avg",snr_avg)
                print("tsnr_avg",tsnr_avg)
                print("snr_std_min",float(snr_std))
                print("tsnr_std_min",float(tsnr_std))


file_object=open(os.path.join('Iterative_methods', args.Dataset + '_'+args.file_name, "a"))
file_object.writelines([ 'lamda_cvx= '+str(lamda_cvx_min)+'\n',
                          'delta_s_cvx= '+str(delta_s_cvx_min)+'\n',
                           'lamda_ncvx= '+str(lamda_ncvx_min)+'\n',
                           'delta_s_ncvx= '+str(delta_s_ncvx_min)+'\n',
                           'Avg MSE= '+str(float(mse_min))+'\n',
                           'SNR avg= '+str(float(snr_min))+'\n',
                            'TSNR avg= '+str(float(tsnr_min))+'\n',
                        ' MSE STD= '+str(float(mse_std_min))+'\n',
                           'SNR STD= '+str(float(snr_std_min))+'\n',
                            'TSNR STD= '+str(float(tsnr_std_min))+'\n',
                        'Avg Num_iterations= '+str(iter_min)+'\n',
                           'Avg Exec_time= '+str(time_min)+'\n'])
file_object.close()
