# How to add a progress bar for LightGBM?

### 1. first create a class 
```python
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
