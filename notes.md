## Firsr Error

I encountered first error while running `pip install -r requirements.txt`:

```
    import distutils.core
ModuleNotFoundError: No module named 'distutils'
```

> The error happens because `numpy==1.25.2` relies on `setuptools/pkg_resources` that call a feature **removed** in Python 3.12 Either upgrade Numpy (≥1.26) for Python 3.12, or use Python 3.11 for Numpy 1.25.2.

Changed `numpy` version from `numpy==1.25.2` to `numpy>=1.26`. **IT WORKED!**

Encounterd a new error after running `pip install -r requirements.txt`

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
umap-learn 0.5.9.post2 requires scikit-learn>=1.6, but you have scikit-learn 1.5.1 which is incompatible.
```

I hope it doesn't break anything.

Then installed other dependencies `pip install matplotlib performer_pytorch umap-learn`


---


## Second Error

```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```
So I replaced the following lines:
```py
...
device = torch.device("cuda")
...
model_dict = model.load_state_dict(torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION))
```
With:
```py
...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...
model_dict = model.load_state_dict(
    torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION, map_location=device)
)
```
This issue is about the cpu/gpu thingies

- [ ] _TODO: Use your GPU to increase the iteration speed_

Now the project is _ready_\* to run

```bash
python test.py # add "--model_arch tiny" to speed things up
```

Output:
```
pr_auc : 0.8935741368889635
roc_auc : 0.8886666666666667
```

---

