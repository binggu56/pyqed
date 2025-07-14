#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:53:53 2025

@author: Yao Wang

dissipaton quantum master equation

# mailto: wy2010@ustc dot edu dot cn
"""



import numpy as np

class DQME:
    """
    dissipaton quantum master equation, an exact quasiparticle representation of
    the HEOM
    """
    def __init__(self, H):
        self.H = H

    def decompose(self):
        pass



class DQMEFermion(DQME):
    # def __init__(self, H):
        # self.H = H



    def run(self, dt, nt):
        pass

class DMQEBoson(DQME):
    def run(self, dt, nt):
        pass


Nsys=2 # system spin
Ndis=6 # number of env spins
Ntot=Nsys+Ndis

creat = np.zeros((2, 2), dtype=complex)
annih = np.zeros((2, 2), dtype=complex)

Id=np.zeros((2, 2), dtype=complex)
sigmaz=np.zeros((2, 2), dtype=complex)
sigmaplus=np.zeros((2, 2), dtype=complex)
sigmaminus=np.zeros((2, 2), dtype=complex)

Esys=-0.5
U=1

creat[0,1]=1
annih[1,0]=1
Id[0,0]=Id[1,1]=1
sigmaz[0,0]=1
sigmaz[1,1]=-1
sigmaplus[0,1]=1
sigmaminus[1,0]=1

gamma=np.zeros((Ndis), dtype=complex)
gamma[0]=1.0
gamma[1]=0.39280835
gamma[2]=1.63039922
gamma[3]=1.0
gamma[4]=0.39280835
gamma[5]=1.63039922


# exponential decomposition
eta=np.zeros((Ndis), dtype=complex)
eta[0]=0.0625-0.03830084*1.j
eta[1]=-0.03703797*1.j
eta[2]=0.07533881*1.j
eta[3]=0.0625-0.03830084*1.j
eta[4]=-0.03703797*1.j
eta[5]=0.07533881*1.j

# gamma=np.zeros((Ndis), dtype=complex)
# gamma[0]=1.0
# gamma[1]=2.0

# eta=np.zeros((Ndis), dtype=complex)
# eta[0]=0.5+1.j
# eta[1]=0.3-1.j

# Jordan-Wigner

def direct_p(a, b):
        c = np.zeros((len(a) * len(b), len(a) * len(b)), dtype=complex)
        for i in range(len(a)):
            for j in range(len(a)):
                for k in range(len(b)):
                    for l in range(len(b)):
                        c[i + k * len(a), j + l * len(a)] = a[i, j] * b[k, l]
        return c

qmdsa=np.zeros((Ntot,2**Ntot, 2**Ntot), dtype=complex)
qmdsc=np.zeros((Ntot,2**Ntot, 2**Ntot), dtype=complex)
qmdsl=np.zeros((Ntot,2**Ntot, 2**Ntot), dtype=complex)
qmdsh=np.zeros((Ntot,2**Ntot, 2**Ntot), dtype=complex)

for i in range(Ntot):
    if (i == 0):
        tema = annih
        temc = creat
    else:
        tema = sigmaz
        temc = sigmaz
    for j in range(1, Ntot):
        if (j < i):
            tema = direct_p(tema, sigmaz)
            temc = direct_p(temc, sigmaz)
        if (j == i):
            tema = direct_p(tema, annih)
            temc = direct_p(temc, creat)
        if (j > i):
            tema = direct_p(tema, Id)
            temc = direct_p(temc, Id)
    qmdsa[i] = tema
    qmdsc[i] = temc

for i in range(Ntot):
    qmdsl[i]=np.dot(qmdsa[i], qmdsc[i])
    qmdsh[i]=np.dot(qmdsc[i], qmdsa[i])


hs=np.zeros((2**Ntot, 2**Ntot), dtype=complex)
for i in range(Nsys):
    hs+=Esys*qmdsc[i].dot(qmdsa[i])

hs+=U*(qmdsc[0].dot(qmdsa[0])).dot(qmdsc[1].dot(qmdsa[1]))

def rem(rho):
    rho1 = np.zeros_like(rho, dtype=complex)

    rho1 = -1.j * (hs.dot(rho) - rho.dot(hs))

    for i in range(Nsys, Nsys + Ndis):
        rho1 -= gamma[i-Nsys] * (qmdsc[i].dot(qmdsa[i])).dot(rho) + gamma[i-Nsys] * (rho.dot(
            qmdsc[i])).dot(qmdsa[i])
        sys_label=int(Nsys*(i-Nsys)//Ndis)
        rho1 -= (1.j) * ((qmdsc[sys_label].dot(qmdsa[i])).dot(rho) -
                         (qmdsa[i].dot(rho)).dot(qmdsc[sys_label]))
        rho1 -= (1.j) * ((qmdsa[sys_label].dot(rho)).dot(qmdsc[i]) -
                         (rho.dot(qmdsc[i])).dot(qmdsa[sys_label]))
        rho1 -= (1.j) * (-eta[i-Nsys] *
                         (qmdsa[sys_label].dot(qmdsc[i])).dot(rho) - np.conj(eta[i-Nsys]) *
                         (qmdsc[i].dot(rho)).dot(qmdsa[sys_label]))
        rho1 -= (1.j) * (eta[i-Nsys] *
                         (qmdsc[sys_label].dot(rho)).dot(qmdsa[i]) + np.conj(eta[i-Nsys]) *
                         (rho.dot(qmdsa[i])).dot(qmdsc[sys_label]))
    return rho1


dt = 0.02
ti = 0
tf = 20
N = int((tf - ti) / dt)
dt2 = dt / 2
dt6 = dt / 6

rhot = np.zeros((2**Ntot, 2**Ntot), dtype=complex)

vac_dis=qmdsl[Nsys]
for a in range(Nsys+1,Ntot):
    vac_dis=np.dot(vac_dis,qmdsl[a])

rhot = ((qmdsc[0].dot(qmdsa[0])).dot(qmdsc[1].dot(qmdsa[1]))).dot(vac_dis)
project = ((qmdsa[0].dot(qmdsc[0])).dot(qmdsa[1].dot(qmdsc[1]))).dot(vac_dis)

prop_t = []
prop_rho = []


for ii in range(N):

    prop_t.append(ti + (ii + 1) * dt)
    prop_rho.append((project.dot(rhot)).trace())

    rho0 = rhot
    rho1 = rem(rho0)
    rho3 = rho0 + dt2 * rho1
    rho2 = rem(rho3)
    rho1 += 2 * rho2
    rho3 = rho0 + dt2 * rho2
    rho2 = rem(rho3)
    rho1 += 2 * rho2
    rho3 = rho0 + dt * rho2
    rho2 = rem(rho3)
    rho1 += rho2
    rhot = rho0 + dt6 * rho1


t = np.array(prop_t)
rhos = np.array(prop_rho)

np.savetxt('t.dat',prop_t)
np.savetxt('output.dat',rhos)


beta1 = 8
npsd1 = 2

# import DQME result
tdqme=np.loadtxt('t.dat')
outdqme=np.loadtxt('output.dat',dtype=complex)

# import HEOM result
magic_str = '{}-{}'.format(beta1, npsd1)
# data1 = fastread_np('./prop-rho-eq-{}-dt'.format(magic_str))
# t1 = data1[:, 0]
# data1 = np.reshape(data1[:, 1:], (len(t1), 4, 4, 2))
# rhot1 = data1[:, :, :, 0] + 1j * data1[:, :, :, 1]
import ultraplot as plt

plt.plot(tdqme, outdqme.real,'b-')
#plt.plot(t, rhos.imag,'r-')
# plt.plot(t1, rhot1[:, 0, 0].real,'r--')
#plt.plot(heom_pop[0].real, heom_pop[1].imag,'k--')
plt.xlim(0,20)
plt.ylim(0,0.1)