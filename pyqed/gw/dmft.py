#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:09:22 2023

@author: bing
"""

# /*----------------------------------------------------------------------------

#    author: Naoto Tsuji <tsuji@cms.phys.s.u-tokyo.ac.jp>

#            Department of Physics, University of Tokyo

#    date:   February 28, 2013

# ----------------------------------------------------------------------------*/
# #include <iostream>
# #include <fstream>
# #include <string.h>
# #include "dmft.h"
# #include "fft.h"
# #include "integral.h"

# dmft::dmft(parm parm_) : local_green(parm_), weiss_green(parm_), self_energy(parm_), weiss_green_new(parm_), weiss_green_T(parm_), self_energy_T(parm_),
#                          density(parm_.N_t+1), double_occupancy(parm_.N_t+1), kinetic_energy(parm_.N_t+1)
# {
#   if (strcmp(parm_.dos,"semicircular")==0){
#     weiss_green_der=KB_derivative(parm_);
#     weiss_green_der_new=KB_derivative(parm_);
#   }
# }

class Parm:
    def __init__(self, dt, nt, beta=None):
        self.dt = dt
        self.nt = nt
        self.temperature = 1./beta

class DMFT:
    def __init__(self):
        pass




    def start_eq_dmft(parm_):
        converged=false;
        G0_diff=1.0;
        time_step=0;
        iteration=1;
        initialize_eq_green(parm_);
        initialize_output_file();

        # equilibrium DMFT self-consistency loop
        while !converged && iteration <= parm_.N_iter:
            eq_impurity_solution(parm_);
            eq_dmft_self_consistency(parm_);
            iteration+=1;
            if (G0_diff<=parm_.tolerance):
                converged=true;
                print("Equilibrium DMFT is converged")

                if iteration==parm_.N_iter+1 && G0_diff>parm_.tolerance:
                    print("Equilibrium DMFT is NOT converged !!!!!")


  # Fourier transformation: G(w) -> G(t)
  # int i, j;
  # double w, tau;
  for (j=0;j<parm_.N_tau;j++){
    w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
    local_green.matsubara_w[j]-=1.0/(xj*w)+(parm_.e2+0.25*parm_.U_i*parm_.U_i)/(xj*w)/(xj*w)/(xj*w);
  }
  fft_w2t(parm_,local_green.matsubara_w,local_green.matsubara_t);
  for (i=0;i<=parm_.N_tau;i++){
    tau=i*parm_.dtau;
    local_green.matsubara_t[i]+=-0.5+0.25*(parm_.e2+0.25*parm_.U_i*parm_.U_i)*tau*(parm_.beta-tau);
  }
  for (j=0;j<parm_.N_tau;j++){
    w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
    local_green.matsubara_w[j]+=1.0/(xj*w)+(parm_.e2+0.25*parm_.U_i*parm_.U_i)/(xj*w)/(xj*w)/(xj*w);
  }

  measure_density(parm_);
  measure_double_occupancy(parm_);
  measure_kinetic_energy(parm_);
  measure_interaction_energy(parm_);
  measure_total_energy(parm_);



    def eq_dmft_self_consistency(parm parm_):
  if (strcmp(parm_.dos,"semicircular")==0){
    // solve the lattice Dyson equation: G0(w)=1/[i*w-G(w)]
    for (int j=0;j<parm_.N_tau;j++){
      double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
      weiss_green.matsubara_w[j]=1.0/(xj*w-local_green.matsubara_w[j]);
    }
  }

  // Fourier transformation: G0(w) -> G0(t)
  for (int j=0;j<parm_.N_tau;j++){
      double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
      weiss_green.matsubara_w[j]-=1.0/(xj*w)+parm_.e2/(xj*w)/(xj*w)/(xj*w);
  }
  fft_w2t(parm_,weiss_green.matsubara_w,weiss_green_new.matsubara_t);
  for (int i=0;i<=parm_.N_tau;i++){
    double tau=i*parm_.dtau;
    weiss_green_new.matsubara_t[i]+=-0.5+0.25*parm_.e2*tau*(parm_.beta-tau);
  }
  for (int j=0;j<parm_.N_tau;j++){
      double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
      weiss_green.matsubara_w[j]+=1.0/(xj*w)+parm_.e2/(xj*w)/(xj*w)/(xj*w);
  }

  // evaluate |G0_{new}-G0_{old}|
  G0_diff=0.0;
  for (int i=0;i<=parm_.N_tau;i++) G0_diff+=abs(weiss_green_new.matsubara_t[i]-weiss_green.matsubara_t[i]);
  cout << "  Iteration #" << iteration << endl;
  cout << "  |G0_new-G0_old|=" << G0_diff << endl;
  // G0_{old} <= G0_{new}
  weiss_green.matsubara_t=weiss_green_new.matsubara_t;
}


    def initialize_eq_green(parm parm_):
{
  // initialize G0(w)
  if (strcmp(parm_.dos,"semicircular")==0){
    double e_max=2.0;
    double de=e_max/parm_.N_e;
    for (int j=0;j<parm_.N_tau;j++){
      weiss_green.matsubara_w[j]=0.0;
      double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
      for (int k=1;k<2*parm_.N_e;k++){
	double ek=(k-parm_.N_e)*de;
	weiss_green.matsubara_w[j]+=de*sqrt(4.0-ek*ek)/(2.0*pi)/(xj*w-ek);
      }
    }
  }

  // Fourier transformation: G0(w) -> G0(t)
  for (int j=0;j<parm_.N_tau;j++){
    double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
    weiss_green.matsubara_w[j]-=1.0/(xj*w)+parm_.e2/(xj*w)/(xj*w)/(xj*w);
  }
  fft_w2t(parm_,weiss_green.matsubara_w,weiss_green.matsubara_t);
  for (int i=0;i<=parm_.N_tau;i++){
    double tau=i*parm_.dtau;
    weiss_green.matsubara_t[i]+=-0.5+0.25*parm_.e2*tau*(parm_.beta-tau);
  }
  for (int j=0;j<parm_.N_tau;j++){
    double w=(2*j-parm_.N_tau+1)*pi/parm_.beta;
    weiss_green.matsubara_w[j]+=1.0/(xj*w)+parm_.e2/(xj*w)/(xj*w)/(xj*w);
  }
}


    def initialize_output_file():

        ofstream ofs1, ofs2, ofs3, ofs4, ofs5;
        ofs1.open("density");
        ofs2.open("double-occupancy");
        ofs3.open("kinetic-energy");
        ofs4.open("interaction-energy");
        ofs5.open("total-energy");
        ofs1.close();
        ofs2.close();
        ofs3.close();
        ofs4.close();
        ofs5.close();



    def start_noneq_dmft(parm parm_):
  initialize_self_energy(parm_);
  for (time_step=1;time_step<=parm_.N_t;time_step++){
    converged=false;
    G0_diff=1.0;
    iteration=1;
    initialize_noneq_green(parm_);
    cout << "t = " << time_step*parm_.dt << endl;
    // nonequilibrium DMFT self-consistency loop
    while(!converged && iteration<=parm_.N_iter){
      noneq_impurity_solution(parm_);
      noneq_dmft_self_consistency(parm_);
      iteration+=1;
      if (G0_diff<=parm_.tolerance){
	converged=true;
	cout << "Nonequilibrium DMFT is converged" << endl;
      }
      if (iteration==parm_.N_iter+1 && G0_diff>parm_.tolerance){
	cout << "Nonequilibrium DMFT is NOT converged !!!!!" << endl;
      }
    }

    int n=time_step;
    // d/dt G0_{old} <= d/dt G0_{new}
    if (strcmp(parm_.dos,"semicircular")==0){
      weiss_green_der.update(parm_, n, weiss_green_der_new);
    }

    measure_density(parm_);
    measure_double_occupancy(parm_);
    measure_kinetic_energy(parm_);
    measure_interaction_energy(parm_);
    measure_total_energy(parm_);
  }
}


    def noneq_dmft_self_consistency(parm parm_):
  int n=time_step;

  if (strcmp(parm_.dos,"semicircular")==0){
    // solve the lattice Dyson equation: id/dt G0(t,t')-G*G0(t,t')=delta(t,t')
    vector<complex<double> > h(n+1);
    for (int i=0;i<=n;i++) h[i]=0.0;
    weiss_green_new.Volterra_intdiff(parm_,n,h,local_green,weiss_green_der,weiss_green_der_new);
  }

  // evaluate |G0_{new}-G0_{old}|
  G0_diff=0.0;
  for (int j=0;j<=n;j++){
    G0_diff+=abs(weiss_green_new.retarded[n][j]-weiss_green.retarded[n][j]);
    G0_diff+=abs(weiss_green_new.lesser[n][j]-weiss_green.lesser[n][j]);
  }
  for (int j=0;j<=parm_.N_tau;j++) G0_diff+=abs(weiss_green_new.left_mixing[n][j]-weiss_green.left_mixing[n][j]);
  for (int i=0;i<=n-1;i++) G0_diff+=abs(weiss_green_new.lesser[i][n]-weiss_green.lesser[i][n]);
  cout << "  Iteration #" << iteration << endl;
  cout << "  |G0_new-G0_old|=" << G0_diff << endl;

  // G0_{old} <= G0_{new}
  for (int j=0;j<=n;j++){
    weiss_green.retarded[n][j]=weiss_green_new.retarded[n][j];
    weiss_green.lesser[n][j]=weiss_green_new.lesser[n][j];
  }
  for (int j=0;j<=parm_.N_tau;j++) weiss_green.left_mixing[n][j]=weiss_green_new.left_mixing[n][j];
  for (int i=0;i<=n-1;i++) weiss_green.lesser[i][n]=weiss_green_new.lesser[i][n];

  // Kadanoff-Baym G0 => contour-ordered G0
  for (int j=0;j<=n;j++){
    weiss_green_T.c12[n][j]=weiss_green.lesser[n][j];
    weiss_green_T.c21[n][j]=weiss_green.lesser[n][j]+weiss_green.retarded[n][j];
  }
  for (int i=0;i<=n-1;i++){
    weiss_green_T.c12[i][n]=weiss_green.lesser[i][n];
    weiss_green_T.c21[i][n]=weiss_green.lesser[i][n]-conj(weiss_green.retarded[n][i]);
  }
  for (int j=0;j<=parm_.N_tau;j++) weiss_green_T.c13[n][j]=weiss_green.left_mixing[n][j];
  for (int i=0;i<=parm_.N_tau;i++) weiss_green_T.c31[i][n]=conj(weiss_green.left_mixing[n][parm_.N_tau-i]);

  // Hermite conjugate
  for (int j=0;j<=n;j++){
    weiss_green_T.c12[n][j]=0.5*(weiss_green_T.c12[n][j]-conj(weiss_green_T.c12[j][n]));
    weiss_green_T.c21[n][j]=0.5*(weiss_green_T.c21[n][j]-conj(weiss_green_T.c21[j][n]));
  }
  for (int i=0;i<=n-1;i++){
    weiss_green_T.c12[i][n]=-conj(weiss_green_T.c12[n][i]);
    weiss_green_T.c21[i][n]=-conj(weiss_green_T.c21[n][i]);
  }
  for (int j=0;j<=parm_.N_tau;j++) weiss_green_T.c13[n][j]=0.5*(weiss_green_T.c13[n][j]+conj(weiss_green_T.c31[parm_.N_tau-j][n]));
  for (int i=0;i<=parm_.N_tau;i++) weiss_green_T.c31[i][n]=conj(weiss_green_T.c13[n][parm_.N_tau-i]);
}


    def initialize_noneq_green(parm parm_):
  int n=time_step;
  if (n==1){
    // initialize G0(t,t') and G(t,t')
    weiss_green.retarded[0][0]=-xj;
    local_green.retarded[0][0]=-xj;
    weiss_green_new.retarded[0][0]=-xj;
    for (int j=0;j<=parm_.N_tau;j++){
      weiss_green.left_mixing[0][j]=-xj*weiss_green.matsubara_t[parm_.N_tau-j];
      local_green.left_mixing[0][j]=-xj*local_green.matsubara_t[parm_.N_tau-j];
      weiss_green_new.left_mixing[0][j]=-xj*weiss_green.matsubara_t[parm_.N_tau-j];
    }
    weiss_green.lesser[0][0]=-xj*weiss_green.matsubara_t[parm_.N_tau];
    local_green.lesser[0][0]=-xj*local_green.matsubara_t[parm_.N_tau];
    weiss_green_new.lesser[0][0]=-xj*weiss_green.matsubara_t[parm_.N_tau];

    if (strcmp(parm_.dos,"semicircular")==0){
      // initialize d/dt G0(t,t') = -i*(G*G0)(t,t')
      weiss_green_der.retarded[0]=0.0;
      for (int j=0;j<=parm_.N_tau;j++) weiss_green_der.left_mixing[j]=0.0;
      vector<complex<double> > SxG(parm_.N_tau+1);
      for (int j=0;j<=parm_.N_tau;j++){
	for (int k=0;k<=j;k++) SxG[k]=local_green.left_mixing[0][k]*weiss_green.matsubara_t[parm_.N_tau+k-j];
	weiss_green_der.left_mixing[j]=xj*parm_.dtau*trapezoid(SxG,0,j);
	for (int k=j;k<=parm_.N_tau;k++) SxG[k]=local_green.left_mixing[0][k]*weiss_green.matsubara_t[k-j];
	weiss_green_der.left_mixing[j]+=-xj*parm_.dtau*trapezoid(SxG,j,parm_.N_tau);
      }
      for (int k=0;k<=parm_.N_tau;k++) SxG[k]=local_green.left_mixing[0][k]*conj(weiss_green.left_mixing[0][parm_.N_tau-k]);
      weiss_green_der.lesser[0]=-xj*(-xj)*parm_.dtau*trapezoid(SxG,0,parm_.N_tau);
    }

    // guess G0(t,t') in the next step
    for (int j=0;j<=n;j++) weiss_green.retarded[n][j]=weiss_green.retarded[0][0];
    for (int j=0;j<=parm_.N_tau;j++) weiss_green.left_mixing[n][j]=weiss_green.left_mixing[0][j];
    for (int j=0;j<=n;j++) weiss_green.lesser[n][j]=weiss_green.lesser[0][0];
    for (int i=0;i<=n-1;i++) weiss_green.lesser[i][n]=weiss_green.lesser[0][0];

    // contour-ordered G0 <= Kadanoff-Baym G0
    for (int i=0;i<=n;i++){
      for (int j=0;j<=n;j++){
	weiss_green_T.c12[i][j]=weiss_green.lesser[i][j];
	weiss_green_T.c21[i][j]=weiss_green.lesser[i][j]-xj;
      }
    }
    for (int i=0;i<=n;i++){
      for (int j=0;j<=parm_.N_tau;j++){
	weiss_green_T.c13[i][j]=-xj*weiss_green.matsubara_t[parm_.N_tau-j];
	weiss_green_T.c31[j][i]=xj*weiss_green.matsubara_t[j];
      }
    }
  }
  else{
    // guess G0(t,t') in the next step by quadratic extrapolation
    weiss_green.retarded[n][n]=-xj;
    for (int k=0;k<=n-2;k++) weiss_green.retarded[n][k]=2.0*weiss_green.retarded[n-1][k]-weiss_green.retarded[n-2][k];
    weiss_green.retarded[n][n-1]=0.5*(weiss_green.retarded[n][n]+weiss_green.retarded[n][n-2]);
    for (int k=0;k<=parm_.N_tau;k++) weiss_green.left_mixing[n][k]=2.0*weiss_green.left_mixing[n-1][k]-weiss_green.left_mixing[n-2][k];
    for (int k=0;k<=n-1;k++){
      weiss_green.lesser[n][k]=2.0*weiss_green.lesser[n-1][k]-weiss_green.lesser[n-2][k];
      weiss_green.lesser[k][n]=2.0*weiss_green.lesser[k][n-1]-weiss_green.lesser[k][n-2];
    }
    weiss_green.lesser[n][n]=2.0*weiss_green.lesser[n-1][n-1]-weiss_green.lesser[n-2][n-2];

    // contour-ordered G0 <= Kadanoff-Baym G0
    for (int k=0;k<=n-1;k++){
      weiss_green_T.c12[n][k]=weiss_green.lesser[n][k];
      weiss_green_T.c12[k][n]=weiss_green.lesser[k][n];
      weiss_green_T.c21[n][k]=weiss_green.lesser[n][k]+weiss_green.retarded[n][k];
      weiss_green_T.c21[k][n]=weiss_green.lesser[k][n]-conj(weiss_green.retarded[n][k]);
    }
    weiss_green_T.c12[n][n]=weiss_green.lesser[n][n];
    weiss_green_T.c21[n][n]=weiss_green.lesser[n][n]-xj;
    for (int k=0;k<=parm_.N_tau;k++){
      weiss_green_T.c13[n][k]=weiss_green.left_mixing[n][k];
      weiss_green_T.c31[k][n]=conj(weiss_green.left_mixing[n][parm_.N_tau-k]);
    }
  }
}


    def measure_density(parm parm_):
  //  n(t)=<c^+(t)c(t)>
  int n=time_step;
  if (n==0) density[0]=-local_green.matsubara_t[parm_.N_tau];
  else density[n]=local_green.lesser[n][n].imag();

  ofstream ofs;
  ofs.open("density", ios::app);
  ofs.precision(10);
  double t=n*parm_.dt;
  ofs << t << " " << fixed << density[n] << endl;
  ofs.close();
}


    def measure_double_occupancy(parm parm_):
  //  d(t)=<n_up(t)*n_do(t)>
  int n=time_step;
  if (n==0){
    // d(t)=n_up(t)*n_do(t)-1/U*\int dtau self_energy^{M}(beta-tau)*G^{M}(tau)
    if (parm_.U_i==0.0){
      double_occupancy[n]=density[n]*density[n];
    }
    else{
      double_occupancy[n]=density[n]*density[n];
      vector<double> SxG(parm_.N_tau+1);
      for (int k=0;k<=parm_.N_tau;k++) SxG[k]=self_energy.matsubara_t[parm_.N_tau-k]*local_green.matsubara_t[k];
      double_occupancy[n]-=1.0/parm_.U_i*parm_.dtau*trapezoid(SxG,0,parm_.N_tau);
    }
  }
  else{
    double t=n*parm_.dt;
    // d(t)=n_up(t)*n_do(t)-i/U*[self_energy*G]^{<}(t,t)
    //     =n_up(t)*n_do(t)-i/U*[self_energy^{R}*G^{<}+self_energy^{<}*G^{A}+self_energy^{Left}*G^{Right}](t,t)
    if (parm_.U(t)==0.0){
      double_occupancy[n]=density[n]*density[n];
    }
    else{
      double_occupancy[n]=density[n]*density[n];
      vector<complex<double> > SxG(max(parm_.N_tau+1,n+1));
      for (int k=0;k<=parm_.N_tau;k++) SxG[k]=self_energy.left_mixing[n][k]*conj(local_green.left_mixing[n][parm_.N_tau-k]);
      double_occupancy[n]+=1.0/parm_.U(t)*parm_.dtau*((-xj)*trapezoid(SxG,0,parm_.N_tau)).imag();
      for (int k=0;k<=n;k++) SxG[k]=self_energy.retarded[n][k]*local_green.lesser[k][n];
      double_occupancy[n]+=1.0/parm_.U(t)*parm_.dt*trapezoid(SxG,0,n).imag();
      for (int k=0;k<=n;k++) SxG[k]=self_energy.lesser[n][k]*conj(local_green.retarded[n][k]);
      double_occupancy[n]+=1.0/parm_.U(t)*parm_.dt*trapezoid(SxG,0,n).imag();
    }
  }

  ofstream ofs;
  ofs.open("double-occupancy", ios::app);
  ofs.precision(10);
  double t=n*parm_.dt;
  ofs << t << " " << fixed << double_occupancy[n] << endl;
  ofs.close();
}


    def measure_kinetic_energy(parm parm_):
  //  E_{kin}(t)=2\sum_{k} E(k)<c_{k}^+(t)c_{k}(t)>
  int n=time_step;
  kinetic_energy[n]=0.0;
  if (n==0){
    if (strcmp(parm_.dos,"semicircular")==0){
      // E_{kin}(0)=-2*\int_0^{beta} dtau G^M(tau)*G^M(beta-tau)
      vector<double> GxG(parm_.N_tau+1);
      for (int k=0;k<=parm_.N_tau;k++) GxG[k]=local_green.matsubara_t[parm_.N_tau-k]*local_green.matsubara_t[k];
      kinetic_energy[n]=-2.0*parm_.dtau*trapezoid(GxG,0,parm_.N_tau);
    }
  }
  else{
    if (strcmp(parm_.dos,"semicircular")==0){
      // E_{kin}(t)=2*Im[G^{R}*G^{<}+G^{<}*G^{A}+G^{Left}*G^{Right}](t,t)
      vector<complex<double> > GxG(max(parm_.N_tau+1,n+1));
      for (int k=0;k<=parm_.N_tau;k++) GxG[k]=local_green.left_mixing[n][k]*conj(local_green.left_mixing[n][parm_.N_tau-k]);
      kinetic_energy[n]+=2.0*parm_.dtau*((-xj)*trapezoid(GxG,0,parm_.N_tau)).imag();
      for (int k=0;k<=n;k++) GxG[k]=local_green.retarded[n][k]*local_green.lesser[k][n];
      kinetic_energy[n]+=2.0*parm_.dt*trapezoid(GxG,0,n).imag();
      for (int k=0;k<=n;k++) GxG[k]=local_green.lesser[n][k]*conj(local_green.retarded[n][k]);
      kinetic_energy[n]+=2.0*parm_.dt*trapezoid(GxG,0,n).imag();
    }
  }

  ofstream ofs;
  ofs.open("kinetic-energy", ios::app);
  ofs.precision(10);
  double t=n*parm_.dt;
  ofs << t << " " << fixed << kinetic_energy[n] << endl;
  ofs.close();
}


    def measure_interaction_energy(parm parm_):
  int n=time_step;
  ofstream ofs;
  ofs.open("interaction-energy", ios::app);
  ofs.precision(10);
  double t=n*parm_.dt;
  if (n==0) ofs << t << " " << parm_.U_i*(double_occupancy[n]-density[n]+0.25) << endl;
  else ofs << t << " " << parm_.U(t)*(double_occupancy[n]-density[n]+0.25) << endl;
  ofs.close();



    def measure_total_energy(parm parm_):
        int n=time_step;
        ofstream ofs;
        ofs.open("total-energy", ios::app);
        ofs.precision(10);
        double t=n*parm_.dt;
        if (n==0) ofs << t << " " << kinetic_energy[n]+parm_.U_i*(double_occupancy[n]-density[n]+0.25) << endl;
        else ofs << t << " " << kinetic_energy[n]+parm_.U(t)*(double_occupancy[n]-density[n]+0.25) << endl;
        ofs.close();

