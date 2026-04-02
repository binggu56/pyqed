"""\
 MINDO3.py: Dewar's MINDO/3 Semiempirical Method

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

#MINDO/3 Parameters: Thru Ar
# in eV
Uss = [ None, -12.505, None,
        None, None, -33.61, -51.79, -66.06,
        -91.73, -129.86, None,
        None, None, None, -39.82, -56.23, -73.39, -98.99, None] 
Upp = [ None, None, None,
        None, None, -25.11, -39.18, -56.40, -78.80, -105.93, None,
        None, None, None, -29.15, -42.31, -57.25, -76.43, None]
gss = [ None, 12.848, None,
        None, None, 10.59, 12.23, 13.59, 15.42, 16.92, None,
        None, None, None, 9.82, 11.56, 12.88, 15.03, None]
gpp = [ None, None, None,
        None, None, 8.86, 11.08, 12.98, 14.52, 16.71, None,
        None, None, None, 7.31, 8.64, 9.90, 11.30, None]
gsp = [ None, None, None,
        None, None, 9.56, 11.47, 12.66, 14.48, 17.25, None,
        None, None, None, 8.36, 10.08, 11.26, 13.16, None]
gppp = [ None, None, None,
         None, None, 7.86, 9.84, 11.59, 12.98, 14.91, None,
         None, None, None, 6.54, 7.68, 8.83, 9.97, None]
hsp = [ None, None, None,
        None, None, 1.81, 2.43, 3.14, 3.94, 4.83, None,
        None, None, None, 1.32, 1.92, 2.26, 2.42, None]
hppp = [ None, None, None,
         None, None, 0.50, 0.62, 0.70, 0.77, 0.90, None,
         None, None, None, 0.38, 0.48, 0.54, 0.67, None]

f03 = [ None, 12.848, 10.0, #averaged repulsion integral for use in gamma
        10.0, 0.0, 8.958, 10.833, 12.377, 13.985, 16.250,
        10.000, 10.000, 0.000, 0.000,7.57 ,  9.00 ,10.20 , 11.73]

IPs = [ None, -13.605, None,
        None, None, -15.160, -21.340, -27.510, -35.300, -43.700, -17.820,
        None, None, None, None, -21.100, -23.840, -25.260, None]
IPp = [ None, None, None,
        None, None, -8.520, -11.540, -14.340, -17.910, -20.890, -8.510,
        None, None, None, None, -10.290, -12.410, -15.090, None]

# slater exponents
zetas = [ None, 1.30, None,
          None, None, 1.211156, 1.739391, 2.704546, 3.640575, 3.111270, None,
          None, None, None, 1.629173, 1.926108, 1.719480, 3.430887, None]
zetap = [ None, None, None,
          None, None, 0.972826, 1.709645, 1.870839, 2.168448, 1.419860, None,
          None, None, None, 1.381721, 1.590665, 1.403205, 1.627017, None]

# Bxy resonance coefficients
Bxy = {
    (1,1) : 0.244770, (1,5) : 0.185347, (1,6) : 0.315011, (1,7) : 0.360776,
    (1,8) : 0.417759, (1,9) : 0.195242, (1,14) : 0.289647, (1,15) : 0.320118,
    (1,16) : 0.220654, (1,17) : 0.231653,
    (5,5) : 0.151324, (5,6) : 0.250031, (5,7) : 0.310959, (5,8) : 0.349745,
    (5,9) : 0.219591,
    (6,6) : 0.419907, (6,7) : 0.410886, (6,8) : 0.464514, (6,9) : 0.247494,
    (6,14) : 0.411377, (6,15) : 0.457816, (6,16) : 0.284620, (6,17) : 0.315480,
    (7,7) : 0.377342, (7,8) : 0.458110, (7,9) : 0.205347,
    (8,8) : 0.659407, (8,9) : 0.334044, (9,9) : 0.197464,
    (14,14) : 0.291703, (15,15) : 0.311790, (16,16) : 0.202489,
    (17,17) : 0.258969,
    (7,15) : 0.457816, # Rick hacked this to be the same as 6,15
    (8,15) : 0.457816, # Rick hacked this to be the same as 6,15
    }

# axy Core repulsion function terms
axy = {     
    (1,1) : 1.489450, (1,5) : 2.090352, (1,6) : 1.475836, (1,7) : 0.589380,
    (1,8) : 0.478901, (1,9) : 3.771362, (1,14) : 0.940789, (1,15) : 0.923170,
    (1,16) : 1.700689, (1,17) : 2.089404,
    (5,5) : 2.280544, (5,6) : 2.138291, (5,7) : 1.909763, (5,8) : 2.484827,
    (5,9) : 2.862183,
    (6,6) : 1.371208, (6,7) : 1.635259, (6,8) : 1.820975, (6,9) : 2.725913,
    (6,14) : 1.101382, (6,15) : 1.029693, (6,16) : 1.761370, (6,17) : 1.676222,
    (7,7) : 2.209618, (7,8) : 1.873859, (7,9) : 2.861667,
    (8,8) : 1.537190, (8,9) : 2.266949, (9,9) : 3.864997,
    (14,14) : 0.918432, (15,15) : 1.186652, (16,16) : 1.751617,
    (17,17) : 1.792125,
    (7,15) : 1.029693, # Rick hacked this to be the same as 6,15
    (8,15) : 1.029693, # Rick hacked this to be the same as 6,15
    }

# Atomic heat of formations: Mopac got from CRC
Hfat = [ None, 52.102, None,
         None, None, 135.7, 170.89, 113.0, 59.559, 18.86, None,
         None, None, None, 106.0, 79.8, 65.65, 28.95, None]

# Default isolated atomic energy values from Mopac:EISOL3
Eat = [None, -12.505, None,
       None ,None,-61.70,-119.47,-187.51,-307.07,-475.00,None,
       None,None,None,-90.98,-150.81,-229.15,-345.93,None]

nbfat = [ None, 1, None,
          None, None, 4, 4, 4, 4, 4, None,
          None, None, None, 4, 4, 4, 4, None]

CoreQ = [ None, 1, None,
          None, None, 3, 4, 5, 6, 7, None,
          None, None, None, 4, 5, 6, 7, None]

NQN = [ None, 1, 1, # principle quantum number N
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3]

