"""Fresh-process timing and workspace/RSS comparison on saved GW response inputs."""
import argparse
import json
import os
import resource
import sys
import threading
import tracemalloc
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
import psutil
from pyqed.gw.shifted_response import ShiftedGW
from benchmarks.benchmark_gw_shifted_response import dense_reference


def dense_many(e,f,no,orbitals,frequencies,de,df):
    """Explicit dense control; one spectrum/response shared by all requests."""
    gap=(e[no:][None,:]-e[:no,None]).ravel()
    dg=(de[no:][None,:]-de[:no,None]).ravel()
    root=np.sqrt(gap);dr=.5*dg/root
    pairs=f[:,:no,no:].reshape(len(f),-1)
    dp=df[:,:no,no:].reshape(pairs.shape)
    k=pairs.T@pairs;dk=dp.T@pairs+pairs.T@dp
    c=np.diag(gap**2)+4*root[:,None]*k*root[None,:]
    dc=np.diag(2*gap*dg)+4*(dr[:,None]*k*root[None,:]+root[:,None]*dk*root[None,:]+root[:,None]*k*dr[None,:])
    del k,dk
    eigen,u=np.linalg.eigh(c);w=np.sqrt(eigen)
    dc=u.T@dc@u
    droot=dc/(w[:,None]+w[None,:])
    del c
    sign=np.where(np.arange(len(e))<no,1.,-1.)
    result=[]
    for orbital,ws in zip(orbitals,frequencies):
        projected=pairs.T@f[:,orbital,:]
        b=u.T@(root[:,None]*projected)
        db=u.T@(dr[:,None]*projected+root[:,None]*(dp.T@f[:,orbital,:]+pairs.T@df[:,orbital,:]))
        block=[]
        for frequency in ws:
            delta=frequency-e
            den=delta[None,:]+w[:,None]*sign
            z=b/(w[:,None]*den)
            value=2*np.sum(b*z)
            slope_by_orbital=-2*np.sum(b*b/(w[:,None]*den**2),axis=0)
            derivative=(4*np.sum(db*z)-2*np.sum(z*(dc@z)*sign)
                        -2*np.sum(z*(droot@z)*delta)-np.sum(slope_by_orbital*de))
            block.append([value,derivative,np.sum(slope_by_orbital)])
        result.append(block)
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace-workspace',action='store_true',help='Allocation audit only; distorts timing.')
    p.add_argument('--method',choices=['independent','shared','dense'],required=True)
    p.add_argument('--reference',type=Path,help='Precomputed dense report; avoids dense work in shared process.')
    a=p.parse_args()
    with np.load(a.inputs) as data:
        e,f,de,df=[data[k] for k in ['energy','factors','de','df']]
        no=int(data['nocc'])
    import hashlib
    with a.inputs.open('rb') as stream:input_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    reference=json.loads(a.reference.read_text()) if a.reference else None
    if reference is not None:assert reference['input_sha256']==input_hash
    r=None
    frequencies=[e[o]+np.array([-.07,.03]) for o in [no-1,no]]
    outputs=[];diagnostics=[]
    stop=threading.Event();rss=[psutil.Process().memory_info().rss]*2
    progress=dict(orbital=None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def monitor():
        last=perf_counter()
        while not stop.wait(.005):
            rss[1]=max(rss[1],psutil.Process().memory_info().rss)
            now=perf_counter()
            if now-last>10:
                data=dict(progress,pid=os.getpid(),elapsed=now-tick,peak_rss_bytes=max(rss))
                if r is not None:
                    data.update(solves=r.info.get('solves',0),iterations=r.info.get('iterations',0),
                                completed_orders=[x['order'] for x in r.info.get('history',[])],
                                history=r.info.get('history',[]),
                                max_relative_residual=r.info.get('max_relative_residual',0.))
                a.output.with_suffix('.progress.json').write_text(json.dumps(data))
                last=now
    thread=threading.Thread(target=monitor);thread.start()
    if a.trace_workspace:tracemalloc.start()
    tick=perf_counter();cpu=process_time()
    try:
        if a.method=='dense':
            outputs=dense_many(e,f,no,[no-1,no],frequencies,de,df)
        else:
            r=ShiftedGW(e,f,no,solve_tol=1e-12,tolerance=1e-10)
            for orbital,ws in zip([no-1,no],frequencies):
                progress['orbital']=orbital
                print(f'{a.method}: orbital {orbital}',flush=True)
                if a.method=='shared':
                    outputs.append(r.evaluate_many(orbital,ws,energy_tangent=de,factor_tangent=df))
                    diagnostics.append(dict(r.info))
                else:
                    block=[]
                    for w in ws:
                        value,derivative=r.evaluate(orbital,w,energy_tangent=de,factor_tangent=df,reuse=False)
                        diagnostics.append(dict(r.info))
                        _,slope=r.evaluate(orbital,w,frequency_tangent=1.,reuse=False)
                        diagnostics.append(dict(r.info))
                        block.append([value,derivative,slope])
                    outputs.append(block)
    except Exception as error:
        stop.set();thread.join()
        report=dict(method=a.method,success=False,error=repr(error),input_sha256=input_hash,
                    seconds=perf_counter()-tick,cpu_seconds=process_time()-cpu,
                    sampled_peak_rss_bytes=max(rss),baseline_rss_bytes=rss[0],
                    progress=progress,diagnostics=diagnostics,current_info=None if r is None else r.info)
        a.output.write_text(json.dumps(report,indent=2))
        raise
    seconds=perf_counter()-tick;cpu=process_time()-cpu
    workspace_peak=None
    if a.trace_workspace:
        _,workspace_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    stop.set();thread.join()
    outputs=np.asarray(outputs)
    if reference is not None:
        expected=np.asarray(reference['outputs'])
    elif a.method=='dense':
        expected=outputs
    else:
        expected=dense_many(e,f,no,[no-1,no],frequencies,de,df)
    errors=abs(outputs-expected).reshape(-1,3)
    report=dict(method=a.method,success=bool(np.allclose(outputs,expected,atol=2e-9,rtol=0)),input_sha256=input_hash,solve_tol=1e-12,quadrature_tol=1e-10,
        orbitals=[no-1,no],frequencies=[x.tolist() for x in frequencies],nocc=no,nao=len(e),
        rank=len(f),seconds=seconds,cpu_seconds=cpu,process_peak_rss_bytes=peak,
        baseline_rss_bytes=rss[0],sampled_peak_rss_bytes=max(rss),traced_peak_bytes=workspace_peak,
        outputs=outputs.tolist(),max_errors=np.max(errors,axis=0).tolist(),diagnostics=diagnostics,
        solves=sum(d['solves'] for d in diagnostics),iterations=sum(d['iterations'] for d in diagnostics))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['outputs','diagnostics']},indent=2))
    np.testing.assert_allclose(outputs,expected,atol=2e-9,rtol=0)


if __name__=='__main__':main()
