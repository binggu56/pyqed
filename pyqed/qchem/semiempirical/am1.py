#!/usr/bin/env python
# flake8: noqa

'''

Native RHF-AM1
(In testing)

Ref:
[1] J. J. Stewart, J. Comp. Chem. 10, 209 (1989)
[2] J. J. Stewart, J. Mol. Model 10, 155 (2004)
'''

import numpy
import types
import warnings
from scipy.linalg import eigh
from numba import njit

from pyqed import au2angstrom
from pyqed.qchem.basis import ContractedGaussian, overlap as gaussian_overlap
from pyqed.qchem.ci.fci import CI_H, SlaterCondon
from pyqed.qchem.mcscf.direct_ci import build_direct_connectivity, _compute_diag_compact
from periodictable import elements

warnings.warn('AM1 model is under testing')

BOHR = au2angstrom
HARTREE2EV = 27.21138602
HARTREE2KCAL = 627.52177300722
EV2KCAL = 23.061
E2 = 0.5291534944018261


def _square_mat_in_trilu_indices(n):
    idx = numpy.empty((n, n), dtype=int)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            idx[i, j] = idx[j, i] = k
            k += 1
    return idx


CORE = numpy.zeros(108, dtype=int)
CORE[[1, 3, 11, 19]] = 1
CORE[[4, 12, 20]] = 2
CORE[[5, 13, 31, 49, 81]] = 3
CORE[[6, 14, 32, 50, 82]] = 4
CORE[[7, 15, 33, 51, 83]] = 5
CORE[[8, 16, 34, 52, 84]] = 6
CORE[[9, 17, 35, 53, 85]] = 7

NATORB = numpy.zeros(108, dtype=int)
NATORB[[1, 2, 95]] = 1
NATORB[[3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 15, 16, 17, 18,
        30, 31, 32, 33, 34, 35, 36,
        37, 38, 48, 49, 50, 51, 52, 53, 54,
        80, 81, 82, 83, 84, 85, 86,
        96, 97, 98, 99]] = 4

gexps = {
    (1, 0): [23.10303149, 4.235915534, 1.185056519, 0.4070988982, 0.1580884151, 0.06510953954],
    (2, 0): [27.68496241, 5.077140627, 1.42678605, 0.2040335729, 0.09260298399, 0.04416183978],
    (2, 1): [5.868285913, 1.530329631, 0.5475665231, 0.2288932733, 0.1046655969, 0.04948220127],
    (3, 0): [3.273031938, 0.9200611311, 0.3593349765, 0.08636686991, 0.04797373812, 0.02724741144],
    (3, 1): [5.077973607, 1.34078694, 0.2248434849, 0.1131741848, 0.06076408893, 0.03315424265],
}

gcoefs = {
    (1, 0): [0.00916359628, 0.04936149294, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841],
    (2, 0): [-0.004151277819, -0.02067024148, -0.05150303337, 0.3346271174, 0.5621061301, 0.1712994697],
    (2, 1): [0.007924233646, 0.05144104825, 0.189840006, 0.4049863191, 0.4012362861, 0.1051855189],
    (3, 0): [-0.006775596947, -0.05639325779, -0.1587856086, 0.5534527651, 0.501535102, 0.07223633674],
    (3, 1): [-0.00332992984, -0.0141948834, 0.163939577, 0.4485358256, 0.390881305, 0.07411456232],
}

EHEAT3 = numpy.zeros(108)
EHEAT3[[1, 5, 6, 7, 8, 9, 14, 15, 16, 17]] = [
    52.102,
    135.7,
    170.89,
    113.0,
    59.559,
    18.86,
    106.0,
    79.8,
    65.65,
    28.95,
]
EISOL3 = numpy.zeros(108)
EISOL3[[1, 5, 6, 7, 8, 9, 14, 15, 16, 17]] = [
    -12.505,
    -61.70,
    -119.47,
    -187.51,
    -307.07,
    -475.00,
    -90.98,
    -150.81,
    -229.15,
    -345.93,
]

mopac_param = types.SimpleNamespace(
    BOHR=BOHR,
    HARTREE2EV=HARTREE2EV,
    HARTREE2KCAL=HARTREE2KCAL,
    EV2KCAL=EV2KCAL,
    E2=E2,
    CORE=CORE,
    EHEAT3=EHEAT3,
    EISOL3=EISOL3,
    gexps=gexps,
    gcoefs=gcoefs,
)


MOPAC_DD = numpy.array((0.,
    0.       , 0.       ,
    2.0549783, 1.4373245, 0.9107622, 0.8236736, 0.6433247, 0.4988896, 0.4145203, 0.,
    0.       , 0.       , 1.4040443, 1.1631107, 1.0452022, 0.9004265, 0.5406286, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3581113, 0., 1.2472095, 0., 0.       , 0.8458104, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.4878778, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.8750829, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4078712, 0.8231596, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.0684105, 0.       , 0., 0., 0., 0.,
))

MOPAC_QQ = numpy.array((0.,
    0.       , 0.       ,
    1.7437069, 1.2196103, 0.7874223, 0.7268015, 0.5675528, 0.4852322, 0.4909446, 0.,
    0.       , 0.       , 1.2809154, 1.3022422, 0.8923660, 1.0036329, 0.8057208, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.5457406, 0., 1.0698642, 0., 0.       , 1.0407133, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.1887388, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.5424241, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.1658281, 0.8225156, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 1.0540926, 0.       , 0., 0., 0., 0.,
))

MOPAC_AM = numpy.array((0.,
    0.4721793, 0.       ,
    0.2682837, 0.3307607, 0.3891951, 0.4494671, 0.4994487, 0.5667034, 0.6218302, 0.,
    0.5      , 0.       , 0.2973172, 0.3608967, 0.4248440, 0.4331617, 0.5523705, 0.,
    0.5      , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4336641, 0., 0.3737084, 0., 0.       , 0.5526071, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.5527544, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3969129, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3608967, 0.4733554, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.4721793, 0.5      , 0.5      ,0.5      , 0.5      , 0.       ,
))

MOPAC_AD = numpy.array((0.,
    0.4721793, 0.       ,
    0.2269793, 0.3356142, 0.5045152, 0.6082946, 0.7820840, 0.9961066, 1.2088792, 0.,
    0.       , 0.       , 0.2630229, 0.3829813, 0.3275319, 0.5907115, 0.7693200, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2317423, 0., 0.3180309, 0., 0.       , 0.6024598, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4497523, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.2926605, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3441817, 0.5889395, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.9262742, 0.       , 0., 0., 0., 0.,
))

MOPAC_AQ = numpy.array((0.,
    0.4721793, 0.       ,
    0.2614581, 0.3846373, 0.5678856, 0.6423492, 0.7883498, 0.9065223, 0.9449355, 0.,
    0.       , 0.       , 0.3427832, 0.3712106, 0.4386854, 0.6454943, 0.6133369, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2621165, 0., 0.3485612, 0., 0.       , 0.5307555, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4631775, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3360599, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3999442, 0.5632724, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.2909059, 0.       , 0., 0., 0., 0.,
))

MOPAC_ALP = numpy.array((0.,
    2.8823240, 0.       ,
    1.2501400, 1.6694340, 2.4469090, 2.6482740, 2.9472860, 4.4553710, 5.5178000, 0.,
    1.6680000, 0.       , 1.9765860, 2.2578160, 2.4553220, 2.4616480, 2.9193680, 0.,
    1.4050000, 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.4845630, 0., 2.1364050, 0., 0.       , 2.5765460, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.2994240, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.4847340, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 2.1961078, 2.4916445, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 2.5441341, 1.5      , 1.5      ,1.5      , 1.5      , 0.       ,
))

MOPAC_ZS = numpy.array((0.,
    1.1880780, 0.       ,
    0.7023800, 1.0042100, 1.6117090, 1.8086650, 2.3154100, 3.1080320, 3.7700820, 0.,
    0.       , 0.       , 1.5165930, 1.8306970, 1.9812800, 2.3665150, 3.6313760, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.9542990, 0., 1.2196310, 0., 0.       , 3.0641330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1028580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 2.0364130, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.6135910, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 4.0000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZP = numpy.array((0.,
    0.       , 0.       ,
    0.7023800, 1.0042100, 1.5553850, 1.6851160, 2.1579400, 2.5240390, 2.4946700, 0.,
    0.       , 0.       , 1.3063470, 1.2849530, 1.8751500, 1.6672630, 2.0767990, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3723650, 0., 1.9827940, 0., 0.       , 2.0383330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1611530, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.9557660, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.0343930, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZD = numpy.array((0.,
    0.       , 0.       ,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.,
    0.       , 0.       , 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.0000000, 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.       , 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.0000000, 1.0000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_USS = numpy.array((0.,
    -11.396427, 0.       ,
    -5.128000,-16.602378,-34.492870,-52.028658,-71.860000,-97.830000,-136.105579,0.,
    0.       , 0.       ,-24.353585,-33.953622,-42.029863,-56.694056,-111.613948,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-21.040008, 0.,-34.183889, 0., 0.       ,-104.656063,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-103.589663,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-19.941578, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-40.568292,-75.239152, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,-11.906276, 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_UPP = numpy.array((0.,
    0.       , 0.       ,
    -2.721200,-10.703771,-22.631525,-39.614239,-57.167581,-78.26238,-104.889885, 0.,
    0.       , 0.       ,-18.363645,-28.934749,-34.030709,-48.717049,-76.640107, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-17.655574, 0.,-28.640811, 0., 0.       ,-74.930052, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-74.429997, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-11.110870, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-28.089187,-57.832013, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_BETAS = numpy.array((0.,
    -6.1737870, 0.       ,
    -1.4598822, -4.4963564, -9.5991140,-15.7157830,-20.2991100,-29.2727730,-69.5902770, 0.,
    -1.1375097,-1.1883090, -3.8668220, -3.7848520, -6.3537640, -3.9205660,-24.5946700, 0.,
    -0.2601130,-4.2657396, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., -1.9974290, 0., -4.3566070, -5.6481504, -3.1470826,-19.3998800, 0.,
    -1.9999892,-9.6008645, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., -6.1333658, -1.9999126, -7.3823300, -8.3897294, -8.4433270, 0.,
    -4.4412054,-9.9997673, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., -0.9086570, 0., -6.6096803, -6.5924919, -0.9993474, 0., 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,-9999999.0000000, 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_BETAP = numpy.array((0.,
    0.       , 0.       ,
    -1.5278541, -2.6466323, -6.2737570, -7.7192830,-18.2386660,-29.2727730,-27.9223600, 0.,
    -2.1005594,-5.2849791, -2.3171460, -1.9681230, -6.5907090, -7.9052780,-14.6372160, 0.,
    -1.6603661,-6.2934710, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., -4.7581190, 0., -0.9910910, -4.9979109, -6.1468406, -8.9571950, 0.,
    -4.4131246,-3.0661804, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., -0.9999602, -2.1702085, -3.6331190, -5.1065429, -6.3234050, 0.,
    -4.3246899,-9.7724365, 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., -4.9093840, 0., -6.5157709, -1.3368867, -1.8948197, 0., 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_GSS = numpy.array((0.,
    12.8480000, 0.       ,
    7.3000000, 9.0000000,10.5900000,12.2300000,13.5900000,15.4200000,16.9200000, 0.,
    0.       , 0.       , 8.0900000, 9.8200000,11.5600050,11.7863290,15.0300000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.8000000, 0.,10.1686050, 0., 0.       ,15.0364395, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,15.0404486, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 10.800000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 9.8200000,12.8800000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,12.8480000, 0.       , 0., 0., 0., 0.,
))

MOPAC_GSP = numpy.array((0.,
    0.       , 0.       ,
    5.4200000, 7.4300000, 9.5600000,11.4700000,12.6600000,14.4800000,17.2500000, 0.,
    0.       , 0.       , 6.6300000, 8.3600000, 5.2374490, 8.6631270,13.1600000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.1820180, 0., 8.1444730, 0., 0.       ,13.0346824, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,13.0565580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 9.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 8.3600000,11.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GPP = numpy.array((0.,
    0.       , 0.       ,
    5.0000000, 6.9700000, 8.8600000,11.0800000,12.9800000,14.5200000,16.7100000, 0.,
    0.       , 0.       , 5.9800000, 7.3100000, 7.8775890,10.0393080,11.3000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,13.3000000, 0., 6.6719020, 0., 0.       ,11.2763254, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,11.1477837, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,14.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 7.3100000, 9.9000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GP2 = numpy.array((0.,
    0.       , 0.       ,
    4.5200000, 6.2200000, 7.8600000, 9.8400000,11.5900000,12.9800000,14.9100000, 0.,
    0.       , 0.       , 5.4000000, 6.5400000, 7.3076480, 7.7816880, 9.9700000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,12.9305200, 0., 6.2697060, 0., 0.       , 9.8544255, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 9.9140907, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,13.5000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 6.5400000, 8.8300000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_HSP = numpy.array((0.,
    0.       , 0.       ,
    0.8300000, 1.2800000, 1.8100000, 2.4300000, 3.1400000, 3.9400000, 4.8300000, 0.,
    0.       , 0.       , 0.7000000, 1.3200000, 0.7792380, 2.5321370, 2.4200000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4846060, 0., 0.9370930, 0., 0.       , 2.4558683, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.4563820, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.3200000, 2.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.1000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_EHEAT = mopac_param.EHEAT3.copy()
MOPAC_EHEAT[[1, 6, 7, 8, 9]] = [52.102000, 170.890000, 113.000000, 59.559000, 18.890000]

MOPAC_EISOL = mopac_param.EISOL3.copy()
MOPAC_EISOL[[1, 6, 7, 8, 9]] = [-11.396427, -120.815794, -202.407743, -316.099520, -482.290583]

MOPAC_IDEA_FN1 = numpy.zeros((108,10))
dat = (
    1 , 1,  0.1227960,
    1 , 2,  0.0050900,
    1 , 3, -0.0183360,
    6 , 1,  0.0113550,
    6 , 2,  0.0459240,
    6 , 3, -0.0200610,
    6 , 4, -0.0012600,
    7 , 1,  0.0252510,
    7 , 2,  0.0289530,
    7 , 3, -0.0058060,
    8 , 1,  0.2809620,
    8 , 2,  0.0814300,
    9 , 1,  0.2420790,
    9 , 2,  0.0036070,
    13, 1,  0.0900000,
    14, 1,  0.25,
    14, 2,  0.061513,
    14, 3,  0.0207890,
    15, 1, -0.0318270,
    15, 2,  0.0184700,
    15, 3,  0.0332900,
    16, 1, -0.5091950,
    16, 2, -0.0118630,
    16, 3,  0.0123340,
    17, 1,  0.0942430,
    17, 2,  0.0271680,
    35, 1,  0.0666850,
    35, 2,  0.0255680,
    53, 1,  0.0043610,
    53, 2,  0.0157060,
)
MOPAC_IDEA_FN1[dat[0::3],dat[1::3]] = numpy.array(dat[2::3]) / mopac_param.HARTREE2EV

MOPAC_IDEA_FN2 = numpy.zeros((108,10))
dat = (
    1 , 1,  5.0000000,
    1 , 2,  5.0000000,
    1 , 3,  2.0000000,
    6 , 1,  5.0000000,
    6 , 2,  5.0000000,
    6 , 3,  5.0000000,
    6 , 4,  5.0000000,
    7 , 1,  5.0000000,
    7 , 2,  5.0000000,
    7 , 3,  2.0000000,
    8 , 1,  5.0000000,
    8 , 2,  7.0000000,
    9 , 1,  4.8000000,
    9 , 2,  4.6000000,
    13, 1, 12.3924430,
    14, 1,  9.000,
    14, 2,  5.00,
    14, 3,  5.00,
    15, 1,  6.0000000,
    15, 2,  7.0000000,
    15, 3,  9.0000000,
    16, 1,  4.5936910,
    16, 2,  5.8657310,
    16, 3, 13.5573360,
    17, 1,  4.0000000,
    17, 2,  4.0000000,
    35, 1,  4.0000000,
    35, 2,  4.0000000,
    53, 1,  2.3000000,
    53, 2,  3.0000000,
)
MOPAC_IDEA_FN2[dat[0::3],dat[1::3]] = dat[2::3]

MOPAC_IDEA_FN3 = numpy.zeros((108,10))
dat = (
    1 , 1,  1.2000000,
    1 , 2,  1.8000000,
    1 , 3,  2.1000000,
    6 , 1,  1.6000000,
    6 , 2,  1.8500000,
    6 , 3,  2.0500000,
    6 , 4,  2.6500000,
    7 , 1,  1.5000000,
    7 , 2,  2.1000000,
    7 , 3,  2.4000000,
    8 , 1,  0.8479180,
    8 , 2,  1.4450710,
    9 , 1,  0.9300000,
    9 , 2,  1.6600000,
    13, 1,  2.0503940,
    14, 1,  0.911453,
    14, 2,  1.995569,
    14, 3,  2.990610,
    15, 1,  1.4743230,
    15, 2,  1.7793540,
    15, 3,  3.0065760,
    16, 1,  0.7706650,
    16, 2,  1.5033130,
    16, 3,  2.0091730,
    17, 1,  1.3000000,
    17, 2,  2.1000000,
    35, 1,  1.5000000,
    35, 2,  2.3000000,
    53, 1,  1.8000000,
    53, 2,  2.2400000,
)
MOPAC_IDEA_FN3[dat[0::3],dat[1::3]] = dat[2::3]
del(dat)


def get_hcore(mol):
    nao = mol.nao
    atom_charges = mol.atom_charges()
    ao_atom_indices = numpy.asarray(mol.ao_atom_indices, dtype=int)
    ao_atom_charges = atom_charges[ao_atom_indices]
    ao_l = numpy.asarray(mol.ao_l, dtype=int)

    basis_u = numpy.where(ao_l == 0, MOPAC_USS[ao_atom_charges], MOPAC_UPP[ao_atom_charges])
    ao_beta = numpy.where(ao_l == 0, MOPAC_BETAS[ao_atom_charges], MOPAC_BETAP[ao_atom_charges])

    # Off-diagonal AM1 resonance terms use the STO overlap as in MOPAC.
    hcore = mol.overlap.copy()
    hcore *= ao_beta[:,None] + ao_beta
    hcore *= .5
    hcore[ao_atom_indices[:,None] == ao_atom_indices] = 0

    # U term
    hcore[numpy.diag_indices(nao)] = basis_u

    # if method == 'INDO' or 'CINDO'
    #    # Nuclear attraction
    #    gamma = _get_gamma(mol)
    #    z_eff = mopac_param.CORE[atom_charges]
    #    vnuc = numpy.einsum('ij,j->i', gamma, z_eff)
    #    aoslices = mol.aoslice_by_atom()
    #    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
    #        idx = numpy.arange(p0, p1)
    #        hcore[idx,idx] -= vnuc[ia]

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        for ja in range(ia):
            w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ia, ja)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
    return hcore


_PACKED_PAIRS_4 = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1),
                   (2, 2), (3, 0), (3, 1), (3, 2), (3, 3))


def _mopac_repp(ni, nj, rij, ev=1.0):
    ev1 = ev / 2.0
    ev2 = ev / 4.0
    ev3 = ev / 8.0
    ev4 = ev / 16.0
    ri = numpy.zeros(22, dtype=float)
    core = numpy.zeros(8, dtype=float)
    si = NATORB[ni] >= 3
    sj = NATORB[nj] >= 3

    if not si and not sj:
        aee = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AM[nj]) ** 2
        ri[0] = ev / numpy.sqrt(rij * rij + aee)
        core[0] = CORE[nj] * ri[0]
        core[4] = CORE[ni] * ri[0]
        return ri, core

    if si and not sj:
        da = MOPAC_DD[ni]
        qa = 2.0 * MOPAC_QQ[ni]
        aee = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AM[nj]) ** 2
        ade = (0.5 / MOPAC_AD[ni] + 0.5 / MOPAC_AM[nj]) ** 2
        aqe = (0.5 / MOPAC_AQ[ni] + 0.5 / MOPAC_AM[nj]) ** 2
        rsq = rij * rij
        sqr = numpy.sqrt([
            rsq + aee,
            (rij + da) ** 2 + ade,
            (rij - da) ** 2 + ade,
            (rij + qa) ** 2 + aqe,
            (rij - qa) ** 2 + aqe,
            rsq + aqe,
            rsq + aqe + qa * qa,
        ])
        ee = ev / sqr[0]
        ri[0] = ee
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2]
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5]
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5]
        core[0] = CORE[nj] * ri[0]
        core[4] = CORE[ni] * ri[0]
        core[1] = CORE[nj] * ri[1]
        core[2] = CORE[nj] * ri[2]
        core[3] = CORE[nj] * ri[3]
        return ri, core

    if not si and sj:
        db = MOPAC_DD[nj]
        qb = 2.0 * MOPAC_QQ[nj]
        aee = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AM[nj]) ** 2
        aed = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AD[nj]) ** 2
        aeq = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AQ[nj]) ** 2
        rsq = rij * rij
        sqr = numpy.sqrt([
            rsq + aee,
            (rij - db) ** 2 + aed,
            (rij + db) ** 2 + aed,
            (rij - qb) ** 2 + aeq,
            (rij + qb) ** 2 + aeq,
            rsq + aeq,
            rsq + aeq + qb * qb,
        ])
        ee = ev / sqr[0]
        ri[0] = ee
        ri[4] = ev1 / sqr[1] - ev1 / sqr[2]
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5]
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5]
        core[0] = CORE[nj] * ri[0]
        core[4] = CORE[ni] * ri[0]
        core[5] = CORE[ni] * ri[4]
        core[6] = CORE[ni] * ri[10]
        core[7] = CORE[ni] * ri[11]
        return ri, core

    da = MOPAC_DD[ni]
    db = MOPAC_DD[nj]
    qa2 = 2.0 * MOPAC_QQ[ni]
    qb2 = 2.0 * MOPAC_QQ[nj]
    aee = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AM[nj]) ** 2
    ade = (0.5 / MOPAC_AD[ni] + 0.5 / MOPAC_AM[nj]) ** 2
    aqe = (0.5 / MOPAC_AQ[ni] + 0.5 / MOPAC_AM[nj]) ** 2
    aed = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AD[nj]) ** 2
    aeq = (0.5 / MOPAC_AM[ni] + 0.5 / MOPAC_AQ[nj]) ** 2
    axx = (0.5 / MOPAC_AD[ni] + 0.5 / MOPAC_AD[nj]) ** 2
    adq = (0.5 / MOPAC_AD[ni] + 0.5 / MOPAC_AQ[nj]) ** 2
    aqd = (0.5 / MOPAC_AQ[ni] + 0.5 / MOPAC_AD[nj]) ** 2
    aqq = (0.5 / MOPAC_AQ[ni] + 0.5 / MOPAC_AQ[nj]) ** 2
    rsq = rij * rij
    arg = numpy.empty(72, dtype=float)
    arg[0] = rsq + aee
    arg[1] = (rij + da) ** 2 + ade
    arg[2] = (rij - da) ** 2 + ade
    arg[3] = (rij - qa2) ** 2 + aqe
    arg[4] = (rij + qa2) ** 2 + aqe
    arg[5] = rsq + aqe
    arg[6] = arg[5] + qa2 * qa2
    arg[7] = (rij - db) ** 2 + aed
    arg[8] = (rij + db) ** 2 + aed
    arg[9] = (rij - qb2) ** 2 + aeq
    arg[10] = (rij + qb2) ** 2 + aeq
    arg[11] = rsq + aeq
    arg[12] = arg[11] + qb2 * qb2
    arg[13] = rsq + axx + (da - db) ** 2
    arg[14] = rsq + axx + (da + db) ** 2
    arg[15] = (rij + da - db) ** 2 + axx
    arg[16] = (rij - da + db) ** 2 + axx
    arg[17] = (rij - da - db) ** 2 + axx
    arg[18] = (rij + da + db) ** 2 + axx
    arg[19] = (rij + da) ** 2 + adq
    arg[20] = arg[19] + qb2 * qb2
    arg[21] = (rij - da) ** 2 + adq
    arg[22] = arg[21] + qb2 * qb2
    arg[23] = (rij - db) ** 2 + aqd
    arg[24] = arg[23] + qa2 * qa2
    arg[25] = (rij + db) ** 2 + aqd
    arg[26] = arg[25] + qa2 * qa2
    arg[27] = (rij + da - qb2) ** 2 + adq
    arg[28] = (rij - da - qb2) ** 2 + adq
    arg[29] = (rij + da + qb2) ** 2 + adq
    arg[30] = (rij - da + qb2) ** 2 + adq
    arg[31] = (rij + qa2 - db) ** 2 + aqd
    arg[32] = (rij + qa2 + db) ** 2 + aqd
    arg[33] = (rij - qa2 - db) ** 2 + aqd
    arg[34] = (rij - qa2 + db) ** 2 + aqd
    arg[35] = rsq + aqq
    arg[36] = arg[35] + (qa2 - qb2) ** 2
    arg[37] = arg[35] + (qa2 + qb2) ** 2
    arg[38] = arg[35] + qa2 * qa2
    arg[39] = arg[35] + qb2 * qb2
    arg[40] = arg[38] + qb2 * qb2
    arg[41] = (rij - qb2) ** 2 + aqq
    arg[42] = arg[41] + qa2 * qa2
    arg[43] = (rij + qb2) ** 2 + aqq
    arg[44] = arg[43] + qa2 * qa2
    arg[45] = (rij + qa2) ** 2 + aqq
    arg[46] = arg[45] + qb2 * qb2
    arg[47] = (rij - qa2) ** 2 + aqq
    arg[48] = arg[47] + qb2 * qb2
    arg[49] = (rij + qa2 - qb2) ** 2 + aqq
    arg[50] = (rij + qa2 + qb2) ** 2 + aqq
    arg[51] = (rij - qa2 - qb2) ** 2 + aqq
    arg[52] = (rij - qa2 + qb2) ** 2 + aqq
    qa = MOPAC_QQ[ni]
    qb = MOPAC_QQ[nj]
    arg[53] = (da - qb) ** 2 + (rij - qb) ** 2 + adq
    arg[54] = (da - qb) ** 2 + (rij + qb) ** 2 + adq
    arg[55] = (da + qb) ** 2 + (rij - qb) ** 2 + adq
    arg[56] = (da + qb) ** 2 + (rij + qb) ** 2 + adq
    arg[57] = (rij + qa) ** 2 + (qa - db) ** 2 + aqd
    arg[58] = (rij - qa) ** 2 + (qa - db) ** 2 + aqd
    arg[59] = (rij + qa) ** 2 + (qa + db) ** 2 + aqd
    arg[60] = (rij - qa) ** 2 + (qa + db) ** 2 + aqd
    arg[61] = arg[35] + 2.0 * (qa - qb) ** 2
    arg[62] = arg[35] + 2.0 * (qa + qb) ** 2
    arg[63] = arg[35] + 2.0 * (qa * qa + qb * qb)
    arg[64] = (rij + qa - qb) ** 2 + (qa - qb) ** 2 + aqq
    arg[65] = (rij + qa - qb) ** 2 + (qa + qb) ** 2 + aqq
    arg[66] = (rij + qa + qb) ** 2 + (qa - qb) ** 2 + aqq
    arg[67] = (rij + qa + qb) ** 2 + (qa + qb) ** 2 + aqq
    arg[68] = (rij - qa - qb) ** 2 + (qa - qb) ** 2 + aqq
    arg[69] = (rij - qa - qb) ** 2 + (qa + qb) ** 2 + aqq
    arg[70] = (rij - qa + qb) ** 2 + (qa - qb) ** 2 + aqq
    arg[71] = (rij - qa + qb) ** 2 + (qa + qb) ** 2 + aqq
    sqr = numpy.sqrt(arg)
    ee = ev / sqr[0]
    dze = -ev1 / sqr[1] + ev1 / sqr[2]
    qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5]
    qxxe = ev1 / sqr[6] - ev1 / sqr[5]
    edz = -ev1 / sqr[7] + ev1 / sqr[8]
    eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11]
    eqxx = ev1 / sqr[12] - ev1 / sqr[11]
    dxdx = ev1 / sqr[13] - ev1 / sqr[14]
    dzdz = ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18]
    dzqxx = ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22]
    qxxdz = ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26]
    dzqzz = (-ev3 / sqr[27] + ev3 / sqr[28] - ev3 / sqr[29] +
             ev3 / sqr[30] - ev2 / sqr[21] + ev2 / sqr[19])
    qzzdz = (-ev3 / sqr[31] + ev3 / sqr[32] - ev3 / sqr[33] +
             ev3 / sqr[34] + ev2 / sqr[23] - ev2 / sqr[25])
    qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35]
    qxxqyy = ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35]
    qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41] - ev3 / sqr[43] - ev2 / sqr[38] + ev2 / sqr[35]
    qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45] - ev3 / sqr[47] - ev2 / sqr[39] + ev2 / sqr[35]
    qzzqzz = (ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52] -
              ev3 / sqr[47] - ev3 / sqr[45] - ev3 / sqr[41] - ev3 / sqr[43] + ev2 / sqr[35])
    dxqxz = -ev2 / sqr[53] + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56]
    qxzdx = -ev2 / sqr[57] + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60]
    qxzqxz = (ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68] + ev3 / sqr[70] -
              ev3 / sqr[65] + ev3 / sqr[67] + ev3 / sqr[69] - ev3 / sqr[71])

    ri[:] = [
        ee, -dze, ee + qzze, ee + qxxe, -edz, dzdz, dxdx,
        -edz - qzzdz, -edz - qxxdz, -qxzdx, ee + eqzz,
        ee + eqxx, -dze - dzqzz, -dze - dzqxx, -dxqxz,
        ee + eqzz + qzze + qzzqzz,
        ee + eqzz + qxxe + qxxqzz,
        ee + eqxx + qzze + qzzqxx,
        ee + eqxx + qxxe + qxxqxx,
        qxzqxz,
        ee + eqxx + qxxe + qxxqyy,
        0.5 * (qxxqxx - qxxqyy),
    ]
    core[:4] = CORE[nj] * ri[[0, 1, 2, 3]]
    core[4:] = CORE[ni] * ri[[0, 4, 10, 11]]
    return ri, core


def _local_two_center_packed(ri, ni, nj):
    m = numpy.zeros((10, 10), dtype=float)
    # Local pair order is ss, xs, xx, ys, yx, yy, zs, zx, zy, zz.
    m[0, [0, 1, 2, 5, 9]] = [ri[0], ri[4], ri[10], ri[11], ri[11]]
    m[1, 0] = ri[1]
    if NATORB[nj] > 1:
        m[[2, 5, 9], 0] = [ri[2], ri[3], ri[3]]
    if NATORB[ni] > 1 and NATORB[nj] <= 1:
        m[0, [1, 2, 5, 9]] = [ri[1], ri[2], ri[3], ri[3]]
    m[1, [1, 2, 5, 9]] = [ri[5], ri[12], ri[13], ri[13]]
    m[2, [1, 2, 5, 9]] = [ri[7], ri[15], ri[17], ri[17]]
    m[5, [1, 2, 5, 9]] = [ri[8], ri[16], ri[18], ri[20]]
    m[9, [1, 2, 5, 9]] = [ri[8], ri[16], ri[20], ri[18]]
    m[3, 3] = ri[6]
    m[6, 6] = ri[6]
    m[3, 4] = ri[14]
    m[6, 7] = ri[14]
    m[4, 3] = ri[9]
    m[7, 6] = ri[9]
    m[4, 4] = ri[19]
    m[7, 7] = ri[19]
    m[8, 8] = ri[21]
    return m


def _packed_to_tensor(packed):
    tensor = numpy.zeros((4, 4, 4, 4), dtype=float)
    for ip, (i, j) in enumerate(_PACKED_PAIRS_4):
        for kp, (k, l) in enumerate(_PACKED_PAIRS_4):
            value = packed[ip, kp]
            tensor[i, j, k, l] = value
            tensor[j, i, k, l] = value
            tensor[i, j, l, k] = value
            tensor[j, i, l, k] = value
    return tensor


def _tensor_to_packed(tensor):
    packed = numpy.empty((10, 10), dtype=float)
    for ip, (i, j) in enumerate(_PACKED_PAIRS_4):
        for kp, (k, l) in enumerate(_PACKED_PAIRS_4):
            packed[ip, kp] = tensor[i, j, k, l]
    return packed


def _rotation_matrix_from_bond(delta):
    rij = numpy.linalg.norm(delta)
    x = delta / rij
    y = numpy.zeros(3, dtype=float)
    z = numpy.zeros(3, dtype=float)
    if abs(x[2]) > .99999999:
        y[1] = 1.0
        z[0] = 1.0
    else:
        z[2] = numpy.sqrt(1.0 - x[2] * x[2])
        inv = 1.0 / z[2]
        y[0] = -inv * x[1] if x[0] > 0 else inv * x[1]
        y[1] = abs(inv * x[0])
        z[0] = -inv * x[0] * x[2]
        z[1] = -inv * x[1] * x[2]
    transform = numpy.zeros((4, 4), dtype=float)
    transform[0, 0] = 1.0
    transform[1:, 1:] = numpy.array((x, y, z), dtype=float).T
    return transform


def _rotate_core_block(css, csp, cpps, cppp, transform):
    local = numpy.zeros((4, 4), dtype=float)
    local[0, 0] = css
    local[0, 1] = local[1, 0] = csp
    local[1, 1] = cpps
    local[2, 2] = cppp
    local[3, 3] = cppp
    return transform @ local @ transform.T


def _native_mopac_rotate(ni, nj, xi, xj):
    delta = numpy.asarray(xi, dtype=float) - numpy.asarray(xj, dtype=float)
    r2 = numpy.dot(delta, delta)
    w = numpy.zeros((10, 10), dtype=float)
    e1b = numpy.zeros(10, dtype=float)
    e2a = numpy.zeros(10, dtype=float)
    if r2 < 2e-5:
        return w, e1b, e2a, 0.0

    rij = numpy.sqrt(r2)
    ri, ccore = _mopac_repp(ni, nj, rij, ev=1.0)
    transform = _rotation_matrix_from_bond(delta)
    local_tensor = _packed_to_tensor(_local_two_center_packed(ri, ni, nj))
    tensor = numpy.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        transform,
        transform,
        transform,
        transform,
        local_tensor,
        optimize=True,
    )
    w = _tensor_to_packed(tensor)
    e1b_mat = -_rotate_core_block(ccore[0], ccore[1], ccore[2], ccore[3], transform)
    e2a_mat = -_rotate_core_block(ccore[4], ccore[5], ccore[6], ccore[7], transform)
    for ip, (i, j) in enumerate(_PACKED_PAIRS_4):
        e1b[ip] = e1b_mat[i, j]
        e2a[ip] = e2a_mat[i, j]

    gam = ri[0]
    scale = 1.0 + numpy.exp(-MOPAC_ALP[ni] * rij) + numpy.exp(-MOPAC_ALP[nj] * rij)
    nt = ni + nj
    if nt == 8 or nt == 9:
        if ni == 7 or ni == 8:
            scale += (rij - 1.0) * numpy.exp(-MOPAC_ALP[ni] * rij)
        if nj == 7 or nj == 8:
            scale += (rij - 1.0) * numpy.exp(-MOPAC_ALP[nj] * rij)
    enuc = CORE[ni] * CORE[nj] * gam * scale
    for atom in (ni, nj):
        d = rij - MOPAC_IDEA_FN3[atom]
        ax = MOPAC_IDEA_FN2[atom] * d * d
        mask = (numpy.abs(MOPAC_IDEA_FN1[atom]) > 0.0) & (ax <= 25.0)
        enuc += CORE[ni] * CORE[nj] / rij * numpy.einsum(
            "i,i->", MOPAC_IDEA_FN1[atom, mask], numpy.exp(-ax[mask])
        )
    return w, e1b, e2a, enuc


def _get_jk_2c_ints(mol, ia, ja):
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    if CORE[zi] > 1 and CORE[zj] <= 1:
        w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ja, ia)
        return w.transpose(2, 3, 0, 1), e2a, e1b, enuc

    ri = mol.atom_coord(ia)
    rj = mol.atom_coord(ja)
    w, e1b, e2a, enuc = _native_mopac_rotate(zi, zj, ri, rj)

    tril2sq = _square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if CORE[zj] <= 1:
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if CORE[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    # enuc from repp integrals is wrong due to the unit of MOPAC_IDEA_FN2 and
    # MOPAC_ALP
    return w, e1b, e2a, enuc


def get_jk(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = numpy.zeros_like(dm)
    vk = numpy.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints(z) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = numpy.arange(p0, p1)
        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = numpy.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = numpy.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, ia, ia)[0]
        vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, ia, ja)[0]
            vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += numpy.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += numpy.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk


def get_eri(mol):
    """Return AM1 AO two-electron integrals in chemist notation."""
    nao = mol.nao
    eri = numpy.zeros((nao, nao, nao, nao), dtype=float)
    atom_charges = mol.atom_charges()
    aoslices = mol.aoslice_by_atom()

    for ia, (p0, p1) in enumerate(aoslices[:, 2:]):
        z = atom_charges[ia]
        j_ints, k_ints = _get_jk_1c_ints(z)
        idx = numpy.arange(p0, p1)
        for a, p in enumerate(idx):
            for b, q in enumerate(idx):
                eri[p, p, q, q] = j_ints[a, b]
                if p != q:
                    eri[p, q, q, p] = k_ints[a, b]
                    eri[p, q, p, q] = k_ints[a, b]
                    eri[q, p, p, q] = k_ints[a, b]
                    eri[q, p, q, p] = k_ints[a, b]

    for ia, (i0, i1) in enumerate(aoslices[:, 2:]):
        for ja, (j0, j1) in enumerate(aoslices[:ia, 2:]):
            w = _get_jk_2c_ints(mol, ia, ja)[0]
            eri[i0:i1, i0:i1, j0:j1, j0:j1] = w
            eri[j0:j1, j0:j1, i0:i1, i0:i1] = w.transpose(2, 3, 0, 1)
    return eri


@njit(nogil=True, cache=True, fastmath=True)
def _fill_dense_ci_from_connectivity(
    h,
    binary,
    h1,
    eri_same,
    eri_cross,
    i_a,
    j_a,
    p_a,
    q_a,
    phase_a,
    i_b,
    j_b,
    p_b,
    q_b,
    phase_b,
    i_aa,
    j_aa,
    p_aa,
    q_aa,
    r_aa,
    s_aa,
    phase_aa,
    i_bb,
    j_bb,
    p_bb,
    q_bb,
    r_bb,
    s_bb,
    phase_bb,
    i_ab,
    j_ab,
    p_ab,
    q_ab,
    r_ab,
    s_ab,
    phase_ab,
):
    n_mo = h1.shape[0]

    for k in range(i_a.shape[0]):
        p = p_a[k]
        q = q_a[k]
        sign = phase_a[k]
        ket = j_a[k]
        val = -sign * h1[p, q]
        for r in range(n_mo):
            if binary[ket, 0, r] and r != q:
                val -= sign * eri_same[p, q, r, r]
            if binary[ket, 1, r]:
                val -= sign * eri_cross[p, q, r, r]
        h[i_a[k], ket] = val

    for k in range(i_b.shape[0]):
        p = p_b[k]
        q = q_b[k]
        sign = phase_b[k]
        ket = j_b[k]
        val = -sign * h1[p, q]
        for r in range(n_mo):
            if binary[ket, 1, r] and r != q:
                val -= sign * eri_same[p, q, r, r]
            if binary[ket, 0, r]:
                val -= sign * eri_cross[p, q, r, r]
        h[i_b[k], ket] = val

    for k in range(i_aa.shape[0]):
        h[i_aa[k], j_aa[k]] = phase_aa[k] * eri_same[p_aa[k], q_aa[k], r_aa[k], s_aa[k]]

    for k in range(i_bb.shape[0]):
        h[i_bb[k], j_bb[k]] = phase_bb[k] * eri_same[p_bb[k], q_bb[k], r_bb[k], s_bb[k]]

    for k in range(i_ab.shape[0]):
        h[i_ab[k], j_ab[k]] = phase_ab[k] * eri_cross[p_ab[k], q_ab[k], r_ab[k], s_ab[k]]


def _dense_ci_hamiltonian_compact(binary, h1, eri_same, eri_cross):
    conn = build_direct_connectivity(binary)
    h = numpy.diag(_compute_diag_compact(h1, eri_same, eri_cross, binary))
    _fill_dense_ci_from_connectivity(
        h,
        binary,
        h1,
        eri_same,
        eri_cross,
        conn.I_A,
        conn.J_A,
        conn.p_A,
        conn.q_A,
        conn.phase_A,
        conn.I_B,
        conn.J_B,
        conn.p_B,
        conn.q_B,
        conn.phase_B,
        conn.I_AA,
        conn.J_AA,
        conn.p_AA,
        conn.q_AA,
        conn.r_AA,
        conn.s_AA,
        conn.phase_AA,
        conn.I_BB,
        conn.J_BB,
        conn.p_BB,
        conn.q_BB,
        conn.r_BB,
        conn.s_BB,
        conn.phase_BB,
        conn.I_AB,
        conn.J_AB,
        conn.p_AB,
        conn.q_AB,
        conn.r_AB,
        conn.s_AB,
        conn.phase_AB,
    )
    return numpy.real_if_close(h)


def energy_nuc(mol):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * BOHR
    enuc = 0
    alp = MOPAC_ALP
    exp = numpy.exp
    gamma = _get_gamma(mol, MOPAC_AM)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            scale = 1. + exp(-alp[ni] * rij) + exp(-alp[nj] * rij)

            nt = ni + nj
            if (nt == 8 or nt == 9):
                if (ni == 7 or ni == 8):
                    scale += (rij - 1.) * exp(-alp[ni] * rij)
                if (nj == 7 or nj == 8):
                    scale += (rij - 1.) * exp(-alp[nj] * rij)
            enuc += CORE[ni] * CORE[nj] * gamma[ia,ja] * scale

            fac1 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[ni], exp(-MOPAC_IDEA_FN2[ni] * (rij - MOPAC_IDEA_FN3[ni])**2))
            fac2 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[nj], exp(-MOPAC_IDEA_FN2[nj] * (rij - MOPAC_IDEA_FN3[nj])**2))
            enuc += CORE[ni] * CORE[nj] / rij * (fac1 + fac2)
    return enuc


def _get_gamma(mol, f03):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * BOHR
    rho = numpy.array([E2 / f03[z] for z in atom_charges])
    gamma = E2 / numpy.sqrt(distances_in_AA**2 + (rho[:,None] + rho)**2 * .25)
    gamma[numpy.diag_indices(mol.natm)] = 0
    return gamma


def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = numpy.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = CORE[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return numpy.diag(dm_diag)


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * HARTREE2KCAL + e_ref
    return e_tot.real


class RAM1:
    '''RHF-AM1 for closed-shell systems.'''

    def __init__(self, mol):
        self.mol = mol
        self.conv_tol = 1e-5
        self.conv_tol_grad = None
        self.max_cycle = 50
        self.damping = 0.0
        self.verbose = 0
        self.e_heat_formation = None
        self.e_tot = None
        self.e_nuc = None
        self.e_elec = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.hcore = None
        self.vhf = None
        self.dm = None
        self.converged = False
        self.cycles = 0
        self._mindo_mol = _make_mindo_mol(mol)
        self._hcore_cache = None
        self._hcore_mo_cache = None
        self._eri_ao_cache = None
        self._eri_mo_cache = None
        self._ci_hamiltonian_cache = {}

    @property
    def nao(self):
        return self._mindo_mol.nao

    @property
    def nelec(self):
        return self._mindo_mol.nelectron

    def build(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._mindo_mol = _make_mindo_mol(self.mol)
        self.clear_integral_cache()
        return self

    def clear_integral_cache(self):
        self._hcore_cache = None
        self._hcore_mo_cache = None
        self._eri_ao_cache = None
        self._eri_mo_cache = None
        self._ci_hamiltonian_cache = {}
        return self

    def clear_mo_integral_cache(self):
        self._hcore_mo_cache = None
        self._eri_mo_cache = None
        self._ci_hamiltonian_cache = {}
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_ao_cross_overlap(self, other, orthogonalized=True):
        """Return AM1 valence-AO overlap between two geometries.

        AM1/NDDO works in an orthogonal valence AO representation.  We use the
        native STO-6G shapes only to define transport between geometries, then
        Löwdin-normalize both endpoint AO bases so the same-geometry transport
        is exactly the identity in the representation used by the SCF/MECI
        coefficients.
        """
        if not isinstance(other, RAM1):
            raise TypeError("AM1 cross-overlap requires another RAM1 reference.")
        sab = _overlap_matrix_between(self._mindo_mol._basis, other._mindo_mol._basis)
        if not orthogonalized:
            return sab
        xa = _symmetric_inverse_sqrt(self._mindo_mol.overlap)
        xb = _symmetric_inverse_sqrt(other._mindo_mol.overlap)
        return xa @ sab @ xb

    def get_mo_cross_overlap(self, other):
        """Return MO overlap C_a^T S_ab C_b for AM1/MECI transport."""
        if self.mo_coeff is None:
            self.run()
        if not isinstance(other, RAM1):
            raise TypeError("AM1 MO cross-overlap requires another RAM1 reference.")
        if other.mo_coeff is None:
            other.run()
        sao = self.get_ao_cross_overlap(other, orthogonalized=True)
        return self.mo_coeff.T @ sao @ other.mo_coeff

    def get_hcore(self, mol=None):
        if self._hcore_cache is None:
            self._hcore_cache = get_hcore(self._mindo_mol)
        return self._hcore_cache

    def get_hcore_mo(self):
        if self.mo_coeff is None:
            raise ValueError("Run RAM1 before requesting MO hcore.")
        if self._hcore_mo_cache is None:
            self._hcore_mo_cache = self.mo_coeff.T @ self.get_hcore() @ self.mo_coeff
        return self._hcore_mo_cache

    def get_eri_ao(self):
        if self._eri_ao_cache is None:
            self._eri_ao_cache = get_eri(self._mindo_mol)
        return self._eri_ao_cache

    def get_eri_mo(self, notation="chem"):
        if notation not in {"chem", "chemist"}:
            raise ValueError("Only chemist notation is implemented for AM1 MO ERIs.")
        if self.mo_coeff is None:
            raise ValueError("Run RAM1 before requesting MO ERIs.")
        if self._eri_mo_cache is None:
            c = self.mo_coeff
            eri_ao = self.get_eri_ao()
            self._eri_mo_cache = numpy.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", eri_ao, c, c, c, c, optimize=True
            )
        return self._eri_mo_cache

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None:
            dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_veff(self, dm=None):
        vj, vk = self.get_jk(dm=dm)
        return vj - 0.5 * vk

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.nelec % 2:
            raise NotImplementedError("RAM1 supports closed-shell systems. Use UAM1 for open shells.")
        nocc = self.nelec // 2
        occ = numpy.zeros(self._mindo_mol.nao)
        occ[:nocc] = 2.0
        return occ

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        return get_init_guess(self._mindo_mol)

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_coeff is None or mo_occ is None:
            raise ValueError("MO coefficients and occupations are not available.")
        mocc = mo_coeff[:, mo_occ > 0]
        return (mocc * mo_occ[mo_occ > 0]) @ mocc.T

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None:
            vhf = self.get_veff(dm)
        e1 = numpy.einsum("ij,ji->", h1e, dm)
        e2 = 0.5 * numpy.einsum("ij,ji->", vhf, dm)
        return e1 + e2, e2

    energy_tot = energy_tot

    def build_mrci_hamiltonian(self, driver):
        if self.mo_coeff is None:
            self.run()
        config = driver.build_configuration_data(self)
        cache_key = (
            tuple(config.active_orbitals),
            bool(driver.full),
            bool(driver.singles),
            bool(driver.doubles),
            driver.spin,
            driver.nref,
            float(driver.selection_threshold),
        )
        cached = self._ci_hamiltonian_cache.get(cache_key)
        if cached is not None:
            driver.determinants = config.binary
            driver.active_determinants = config.active_binary
            driver.determinant_labels = _am1_determinant_labels(config.binary)
            return cached

        h1_spatial = self.get_hcore_mo()
        eri_spatial = self.get_eri_mo()
        eri_aa = eri_spatial - eri_spatial.swapaxes(1, 3)
        h1 = numpy.asarray([h1_spatial, h1_spatial])
        h2 = numpy.stack(
            (
                numpy.stack((eri_aa, eri_spatial)),
                numpy.stack((eri_spatial, eri_aa)),
            )
        )
        active = numpy.asarray(config.active_orbitals, dtype=int)
        if len(active) < h1_spatial.shape[0]:
            frozen = config.frozen_occ.astype(float)
            h1_active = numpy.take(numpy.take(h1, active, axis=1), active, axis=2)
            h2_active_frozen = numpy.take(numpy.take(h2, active, axis=2), active, axis=3)
            h1_active = h1_active + numpy.einsum(
                "STpqrr,Tr->Spq", h2_active_frozen, frozen, optimize=True
            )
            h2_active = numpy.take(numpy.take(h2_active_frozen, active, axis=4), active, axis=5)
            e_frozen = numpy.einsum("Spp,Sp->", h1, frozen, optimize=True)
            e_frozen += 0.5 * numpy.einsum(
                "STppqq,Sp,Tq->", h2, frozen, frozen, optimize=True
            )
            binary_for_ci = config.active_binary
            if driver.full:
                h_ci = _dense_ci_hamiltonian_compact(
                    binary_for_ci,
                    h1_active[0],
                    h2_active[0, 0],
                    h2_active[0, 1],
                )
            else:
                sc1, sc2 = SlaterCondon(binary_for_ci)
                h_ci = CI_H(binary_for_ci, h1_active, h2_active, sc1, sc2)
            h_ci = h_ci + numpy.eye(h_ci.shape[0]) * e_frozen
        else:
            binary_for_ci = config.binary
            if driver.full:
                h_ci = _dense_ci_hamiltonian_compact(
                    binary_for_ci,
                    h1_spatial,
                    eri_aa,
                    eri_spatial,
                )
            else:
                sc1, sc2 = SlaterCondon(binary_for_ci)
                h_ci = CI_H(binary_for_ci, h1, h2, sc1, sc2)
        driver.determinants = config.binary
        driver.active_determinants = config.active_binary
        driver.determinant_labels = _am1_determinant_labels(config.binary)
        self._ci_hamiltonian_cache[cache_key] = h_ci
        return h_ci

    def MRCI(self, **kwargs):
        from pyqed.qchem.semiempirical import MRCI

        return MRCI(self, **kwargs)

    def MECI(self, **kwargs):
        from pyqed.qchem.semiempirical import MECI

        return MECI(self, **kwargs)

    def run(self, conv_tol=None, max_cycle=None, verbose=None, dm0=None, **kwargs):
        if conv_tol is None:
            conv_tol = kwargs.pop("tol", self.conv_tol)
        if max_cycle is None:
            max_cycle = kwargs.pop("max_cycle", self.max_cycle)
        if verbose is None:
            verbose = kwargs.pop("verbose", self.verbose)
        damping = float(kwargs.pop("damping", self.damping))
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.verbose = int(verbose)
        self.damping = damping

        hcore = self.get_hcore()
        if dm0 is None:
            dm = self.get_init_guess()
        else:
            dm = numpy.asarray(dm0, dtype=float)

        e_last = None
        mo_energy = None
        mo_coeff = None
        mo_occ = None
        vhf = None
        for cycle in range(1, self.max_cycle + 1):
            vhf = self.get_veff(dm)
            fock = hcore + vhf
            mo_energy, mo_coeff = eigh(fock)
            mo_occ = self.get_occ(mo_energy, mo_coeff)
            dm_new = self._make_rdm1_from_mo(mo_coeff, mo_occ)
            if damping:
                dm_new = (1.0 - damping) * dm_new + damping * dm
            vhf_new = self.get_veff(dm_new)
            e_elec = self.energy_elec(dm_new, hcore, vhf_new)[0]
            e_tot = e_elec + self.energy_nuc()
            ddm = numpy.linalg.norm(dm_new - dm)
            de = numpy.inf if e_last is None else abs(e_tot - e_last)
            if self.verbose:
                print(f"cycle= {cycle} E= {e_tot:.14f} delta_E= {0.0 if e_last is None else e_tot - e_last:.3g} |ddm|= {ddm:.3g}")
            dm = dm_new
            vhf = vhf_new
            e_last = e_tot
            self.cycles = cycle
            if de < self.conv_tol and ddm < numpy.sqrt(self.conv_tol):
                self.converged = True
                break
        else:
            self.converged = False
            if damping == 0.0:
                if self.verbose:
                    print("SCF did not converge; retrying with damped fixed-point iterations")
                return self.run(
                    conv_tol=conv_tol,
                    max_cycle=max(self.max_cycle, 200),
                    verbose=verbose,
                    dm0=dm,
                    damping=0.7,
                )

        self.e_tot = float(e_last)
        self.e_nuc = float(self.energy_nuc())
        self.e_elec = float(self.e_tot - self.e_nuc)
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.clear_mo_integral_cache()
        self.hcore = hcore
        self.vhf = vhf
        self.dm = dm
        self.energy_tot(dm, hcore, vhf)
        if self.verbose:
            print(f"converged SCF energy = {self.e_tot:.14f}")
            print(f"Heat of formation = {self.e_heat_formation:.14f} kcal/mol")
        return self

    kernel = run

    @staticmethod
    def _make_rdm1_from_mo(mo_coeff, mo_occ):
        mocc = mo_coeff[:, mo_occ > 0]
        return (mocc * mo_occ[mo_occ > 0]) @ mocc.T

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        raise NotImplementedError


class UAM1:
    '''UHF-AM1 placeholder.'''

    def __init__(self, mol):
        self.mol = mol

    def run(self, *args, **kwargs):
        raise NotImplementedError("Native UAM1 is not implemented yet.")

    kernel = run


def _am1_determinant_labels(binary):
    labels = []
    for det in binary:
        alpha = tuple(numpy.flatnonzero(det[0]).astype(int))
        beta = tuple(numpy.flatnonzero(det[1]).astype(int))
        labels.append((alpha, beta))
    return tuple(labels)


class _NativeMINDOMol:
    def __init__(self, symbols, coords, charges, charge=0, spin=0):
        self._symbols = tuple(symbols)
        self._coords = numpy.asarray(coords, dtype=float)
        self._charges = numpy.asarray(charges, dtype=int)
        self.charge = int(charge)
        self.spin = int(spin)
        self.natm = len(self._symbols)
        self.natom = self.natm
        self._build_am1_basis()
        self.nelectron = int(CORE[self._charges].sum() - self.charge)

    def _build_am1_basis(self):
        basis = []
        ao_atom_indices = []
        ao_l = []
        aoslices = []
        for ia, (coord, charge) in enumerate(zip(self._coords, self._charges)):
            p0 = len(basis)
            basis.append(_make_sto_6g_function(charge, 0, coord))
            ao_atom_indices.append(ia)
            ao_l.append(0)
            if charge > 2:
                for shell in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    basis.append(_make_sto_6g_function(charge, 1, coord, shell=shell))
                    ao_atom_indices.append(ia)
                    ao_l.append(1)
            p1 = len(basis)
            aoslices.append((0, 0, p0, p1))

        self._basis = tuple(basis)
        self.ao_atom_indices = numpy.asarray(ao_atom_indices, dtype=int)
        self.ao_l = numpy.asarray(ao_l, dtype=int)
        self._aoslices = numpy.asarray(aoslices, dtype=int)
        self.nao = len(basis)
        self.overlap = _overlap_matrix(basis)

    def atom_charge(self, atm_id):
        return int(self._charges[atm_id])

    def atom_charges(self):
        return self._charges.copy()

    def atom_coord(self, atm_id):
        return self._coords[atm_id].copy()

    def atom_coords(self):
        return self._coords.copy()

    def atom_symbol(self, atm_id):
        return self._symbols[atm_id]

    def atom_symbols(self):
        return list(self._symbols)

    def aoslice_by_atom(self):
        return self._aoslices.copy()

    def ao_labels(self):
        labels = []
        for i, (atom_idx, l) in enumerate(zip(self.ao_atom_indices, self.ao_l)):
            if l == 0:
                orb = "s"
            else:
                shell = self._basis[i].shell
                orb = {(1, 0, 0): "px", (0, 1, 0): "py", (0, 0, 1): "pz"}[tuple(shell)]
            labels.append(f"{atom_idx} {self._symbols[atom_idx]} {orb}")
        return labels


def _make_sto_6g_function(charge, l, coord, shell=None):
    n = _principle_quantum_number(charge)
    zeta = MOPAC_ZS[charge] if l == 0 else MOPAC_ZP[charge]
    if zeta == 0:
        raise ValueError(f"AM1 zeta parameter is not available for Z={charge}, l={l}.")
    if shell is None:
        shell = (0, 0, 0) if l == 0 else (1, 0, 0)
    es = numpy.asarray(gexps[(n, l)], dtype=float) * zeta**2
    cs = numpy.asarray(gcoefs[(n, l)], dtype=float)
    return ContractedGaussian(origin=coord, shell=shell, exps=es, coefs=cs)


def _principle_quantum_number(charge):
    if charge < 3:
        return 1
    if charge < 10:
        return 2
    if charge < 18:
        return 3
    return 4


def _overlap_matrix(basis):
    nao = len(basis)
    s = numpy.empty((nao, nao), dtype=float)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis[: i + 1]):
            value = 0.0
            for ia, wa in enumerate(bi.prim_weights):
                for ib, wb in enumerate(bj.prim_weights):
                    value += wa * wb * gaussian_overlap(
                        float(bi.exps[ia]),
                        tuple(bi.shell),
                        bi.origin,
                        float(bj.exps[ib]),
                        tuple(bj.shell),
                        bj.origin,
                    )
            s[i, j] = s[j, i] = value
    return s


def _overlap_matrix_between(basis_a, basis_b):
    s = numpy.empty((len(basis_a), len(basis_b)), dtype=float)
    for i, bi in enumerate(basis_a):
        for j, bj in enumerate(basis_b):
            value = 0.0
            for ia, wa in enumerate(bi.prim_weights):
                for ib, wb in enumerate(bj.prim_weights):
                    value += wa * wb * gaussian_overlap(
                        float(bi.exps[ia]),
                        tuple(bi.shell),
                        bi.origin,
                        float(bj.exps[ib]),
                        tuple(bj.shell),
                        bj.origin,
                    )
            s[i, j] = value
    return s


def _symmetric_inverse_sqrt(s, threshold=1.0e-12):
    eig, vec = numpy.linalg.eigh(0.5 * (numpy.asarray(s) + numpy.asarray(s).T))
    if numpy.any(eig <= threshold):
        raise numpy.linalg.LinAlgError("AO overlap matrix is singular.")
    return (vec * (eig ** -0.5)) @ vec.T


def _symbols_from_charges(charges):
    return [elements[int(z)].symbol for z in charges]


def _make_mindo_mol(mol):
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=int)
    if hasattr(mol, "atom_symbols"):
        symbols = list(mol.atom_symbols())
    else:
        symbols = _symbols_from_charges(atom_charges)
    coords = numpy.asarray(mol.atom_coords(), dtype=float)
    charge = int(getattr(mol, "charge", 0))
    spin = int(getattr(mol, "spin", 0))
    return _NativeMINDOMol(symbols, coords, atom_charges, charge=charge, spin=spin)


def _get_jk_1c_ints(z):
    if z < 3:  # H, He
        j_ints = numpy.zeros((1,1))
        k_ints = numpy.zeros((1,1))
        j_ints[0,0] = MOPAC_GSS[z] / mopac_param.HARTREE2EV
    else:
        j_ints = numpy.zeros((4,4))
        k_ints = numpy.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = MOPAC_GSS[z] / mopac_param.HARTREE2EV
        j_ints[0,1:] = j_ints[1:,0] = MOPAC_GSP[z] / mopac_param.HARTREE2EV
        j_ints[p_off_idx] = MOPAC_GP2[z] / mopac_param.HARTREE2EV
        j_ints[p_diag_idx] = MOPAC_GPP[z] / mopac_param.HARTREE2EV

        k_ints[0,1:] = k_ints[1:,0] = MOPAC_HSP[z] / mopac_param.HARTREE2EV
        k_ints[p_off_idx] = (MOPAC_GPP[z] - MOPAC_GP2[z]) / (2 * mopac_param.HARTREE2EV)
    return j_ints, k_ints


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf = MOPAC_EHEAT[atom_charges].sum()
    Eat = MOPAC_EISOL[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL


if __name__ == '__main__':
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom='''O  0  0  0
                H  0 -0.757  .587
                H  0  0.757  .587''',
        unit="Angstrom",
    )

    mf = RAM1(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation)
