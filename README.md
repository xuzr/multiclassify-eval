# multiclassify-eval
a simple python module to calculate precision, recall,  accuracy and f-measure for multi-classify evaluation

You can init it by:

```python
from evaluation import 
evals = Evaluations(pred,gt,classes)
```

Which `pred`and `gt`  can be a 1-dimension `numpy.ndarray` or `list` range from $[0, N-1]$, $N$ is the num of classes. And `classes` is a 1-dimension list contains class labels and `len(classes)=N` must be hold.

After that you can call `evals.average` or `evals.classes[*]` with `.precision()`, `.recall()`, `.accuracy()`  or `.f1_score()`  to access average or any single class evaluation.

And you can call `print(evals)`  to see all classes.







