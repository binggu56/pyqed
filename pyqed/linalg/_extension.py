"""Load small algorithm extensions without importing tensor-network modules."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def load_extension(name, source, dependencies, link_flags=()):
    """Return (module, error); cache by source, ABI and compiler configuration."""
    try:
        import pybind11
        source=Path(source)
        compiler=shlex.split(os.environ.get('CXX') or 'c++')
        flags=['-O3','-std=c++17','-shared','-fPIC']
        if sys.platform=='darwin':
            sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
            flags+=['-mcpu=native','-isysroot',sdk,'-isystem',str(Path(sdk)/'usr/include/c++/v1'),
                    '-undefined','dynamic_lookup','-framework','Accelerate']
        elif sys.platform=='win32':
            raise RuntimeError('Runtime extension builds are not supported on Windows')
        flags+=['-I'+sysconfig.get_path('include'),'-I'+pybind11.get_include()]
        digest=hashlib.sha256(repr((sys.version,sysconfig.get_platform(),sysconfig.get_config_var('SOABI'),
                                  compiler,flags,tuple(link_flags),pybind11.__version__)).encode())
        for path in (source,*map(Path,dependencies)):
            digest.update(path.read_bytes())
        cache=Path(tempfile.gettempdir())/'pyqed-algorithms'/digest.hexdigest()
        suffix=sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        target=cache/(name.rsplit('.',1)[-1]+suffix)
        if not target.exists():
            if os.environ.get('PYQED_AUTO_BUILD','1').lower() in ('0','false','no'):
                raise RuntimeError('Algorithm extension build disabled by PYQED_AUTO_BUILD')
            cache.mkdir(parents=True,exist_ok=True)
            # Atomic publication avoids loading an incomplete concurrent build.
            with tempfile.TemporaryDirectory(dir=cache) as staging:
                output=Path(staging)/target.name
                subprocess.run([*compiler,*flags,str(source),'-o',str(output),*link_flags],
                               check=True,capture_output=True,text=True)
                os.replace(output,target)
        spec=importlib.util.spec_from_file_location(name,target)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name]=module
        return module,None
    except Exception as exc:
        return None,str(exc)+'\n'+(getattr(exc,'stderr',None) or '')
