# GPU-RunTime-analysis---TensorFlow
Regression analysis via ANN with Tensorflow '2.9.1'.

A kaggle [dataset](https://www.kaggle.com/code/miquel0/gpu-runtime-analysis) is utilised to predict the GPU run time based on 14 variables. The number of instances is 241,600.

```
Attribute Information:

  - Independent variables:

    1-2. MWG, NWG: per-matrix 2D tiling at workgroup level: {16, 32, 64, 128} (integer)
    3. KWG: inner dimension of 2D tiling at workgroup level: {16, 32} (integer)
    4-5. MDIMC, NDIMC: local workgroup size: {8, 16, 32} (integer)
    6-7. MDIMA, NDIMB: local memory shape: {8, 16, 32} (integer)
    8. KWI: kernel loop unrolling factor: {2, 8} (integer)
    9-10. VWM, VWN: per-matrix vector widths for loading and storing: {1, 2, 4, 8} (integer)
    11-12. STRM, STRN: enable stride for accessing off-chip memory within a single thread: {0, 1} (categorical)
    13-14. SA, SB: per-matrix manual caching of the 2D workgroup tile: {0, 1} (categorical)
    
  - Output:
  
    15-18. Run1, Run2, Run3, Run4: performance times in milliseconds for 4 independent runs using the same parameters. They range between 13.25 and 3397.08.
```

- The employed inferential model is a Deep Artificial Neural Network with 5 layers. 
- The python script can be found under the *gpu_runtime_analysis.py*, where the user can easily tune the hyperparameters (e.g., number of layers, neurons, learning rate, epochs etc.) to achieve better performance. 
- The 'training/test loss function per epoch' plot for the current model/parametrisation, as well as the __predicted vs true__ GPU runtime plot for the test set, are provided into the images folder.
