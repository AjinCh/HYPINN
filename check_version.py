import sys
def v(mod):
    try:
        m = __import__(mod)
        return getattr(m, '__version__', 'n/a')
    except Exception:
        return 'not installed'

print("Python:", sys.version.split()[0])
print("torch:", v("torch"))
print("torchvision:", v("torchvision"))
print("numpy:", v("numpy"))
print("pandas:", v("pandas"))
print("matplotlib:", v("matplotlib"))
print("scikit-learn:", v("sklearn"))
print("json5:", v("json5"))
print("yaml (PyYAML):", v("yaml"))
