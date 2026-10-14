"""Paired eight-thread BSE timing with fixed inputs and a saved baseline implementation."""
import importlib.util,importlib,time,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from benchmarks.gw_bse_matched_scaling import restore
from pyqed.gw.gw import GW
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening
from scipy.linalg import cho_factor
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--qp', type=Path, required=True)
parser.add_argument('--before-source', type=Path, required=True)
parser.add_argument('--before-kernel', type=Path, required=True)
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--output-name', default='paired_bse.json')
args=parser.parse_args()
b=importlib.import_module('pyqed.gw.bse')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old=load('_bse_contractions',str(args.before_kernel))
new=b._bse_contractions
base=load('bse_baseline',str(args.before_source));base._bse_contractions=old
out=args.input
with threadpool_limits(limits=1):
 mf,inp=restore(out);gw=GW(mf,freq_int='ac',threads=8);gw.e_qp=np.load(args.qp)['qp']
 response=ContourDeformation(gw,mf.mo_energy,range(len(gw.e_qp)),nw=8,prepare=False)
 screen=StaticScreening(gw._pair_factors,cho_factor(response.dielectric(0.,imaginary=True),lower=True),gw.max_memory)
 m=b.BSE(gw,threads=8);m._static_screening=screen
 no=m.nocc;f=m._pair_factors;o=screen.static_occupied(no);v=screen.static_ov(no)
 x=np.random.default_rng(421).normal(size=(2*no*(m.nso-no),5))
 results=[]
 for repeat in range(3):
  for label,module in ([('before',old),('after',new)] if repeat%2==0 else [('after',new),('before',old)]):
   t=time.perf_counter();r=module.action(f,o,v,x,no,'full',8)[0];elapsed=time.perf_counter()-t
   if label=='before':expected=r
   else:np.testing.assert_allclose(r,expected,rtol=1e-11,atol=1e-11)
   results.append(dict(stage='action',repeat=repeat,label=label,seconds=elapsed));print(results[-1],flush=True)
 for repeat in range(2):
  for label,module in ([('before',base),('after',b)] if repeat==0 else [('after',b),('before',base)]):
   m=module.BSE(gw,threads=8);m._static_screening=screen
   t=time.perf_counter();m.run(nroots=5,tol=args.tol,max_cycle=200,max_space=160);elapsed=time.perf_counter()-t
   results.append(dict(stage='solve',repeat=repeat,label=label,seconds=elapsed,energies=m.e.tolist(),residuals=m.info['residual_norms'].tolist()));print(results[-1],flush=True)
 (out/args.output_name).write_text(json.dumps(results,indent=2)+'\n')
