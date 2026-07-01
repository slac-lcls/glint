# back-compat shim: glint_index now lives in the fftindex package. Alias the module object so
# research scripts (incl. their module-global mutations, e.g. gf.QPOW=...) keep working.
import sys, importlib
sys.modules[__name__] = importlib.import_module('glint.glint_index')
