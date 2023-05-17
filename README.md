# Pruebas de TensorRT en una red custom usando pytorch

correr red normal: `python3 evalCustom.py`

correr red optimizada con TRT: `python3 evalCustomNetRT.py`

# Resultados


|  Model      | Name            |size MB | Time CPU (ms)  |Time CUDA (us)|Time CPU %           | Time CUDA % |# of calls | accuracy % | 
|-------------|-----------------|--------|----------------|--------------|---------------------|-------------|-----------|------------|
| Vanilla     |                 |1.7     | 87.000         |351.000       |100%                 |100%         |           |100%        |
|             | model inference |        |                |              |48.00%               |100%         |           ||
|             | cudaLaunchKernel|        |                |              |26.81%               |0%           |15         ||
|             | cudaFree        |        |                |              |11.68%               |0%           |4          ||
| TRT fp16    |                 |1.1     | 5.425          |72.00         |100%                 |             |           |            |
|             | model inference |        |                |              |0%                   |100%         |           ||
|             | cudaLaunchKernel|        |                |              |63.14%               |0%           |3          ||
|             | cudaFree        |        |                |              |36.18%               |0%           |2          ||


---

obs: * resultados obtenidos corriendo las redes en un solo batch de 64 imagenes.
    * El tamaño del modelo Vanilla hace referencia al modelo pasado a onnx.

# Referencias

* profiler pytorch: `https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html`
* profiler pytorch blog: `https://levelup.gitconnected.com/pytorch-official-blog-detailed-pytorch-profiler-v1-9-7a5ca991a97b`