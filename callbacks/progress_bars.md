# How to add a progress bar for LightGBM?

### 1. create a class 
```python
import tqdm

class ProgressBarCallback:
  def __init__(self, total_iterations):
    self.pbar = tqdm.tqdm(total=total_iterations, desc="Training LightGBM")
  def __call__(self, env):
    if env.iteration == env.begin_iteration:
      self.pbar.total = env.end_iteration
    self.pbar.update(1)
    if env.iteration == env.end_iteration - 1:
      self.pbar.close() 
```
### 2. initialize an instance 
```python
progress_callback = ProgressBarCallback(total_iterations=num_rounds)
```

### 3. use it in the LightGBM training callbacks 
```python
import lightgbm as lgb

clf = lgb.train(
  ...,
  callbacks=[
    progress_callback
  ]
)
```
