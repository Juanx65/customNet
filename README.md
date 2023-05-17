# Pruebas de TensorRT en una red custom usando pytorch

correr red normal: `python3 evalCustom.py`

correr red optimizada con TRT: `python3 evalCustomNetRT.py`

# Resultados


|  Model      | Name            |size MB | Time CPU (ms)  |Time CUDA (us)|Time CPU %           | Time CUDA % |accuracy % | 
|-------------|-----------------|--------|----------------|--------------|---------------------|-------------|-----------|
| Vanilla     |                 |1.7     | 5.425          |5.00          |5.39                 |             |100        |
|             | model inference |        | 5.425          |5.00          |5.39                 |             |100        |
| TRT fp16    |                 |1.1     | 5.425          |72.00         |42.62                |             |100        |


Self CPU time total: 5.425ms

Self CUDA time total: 72.000us

---

obs: * resultados obtenidos corriendo las redes en un solo batch de 64 imagenes.
    * El tamaño del modelo Vanilla hace referencia al modelo pasado a onnx.

# Referencias

* profiler pytorch: `https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html`
* profiler pytorch blog: `https://levelup.gitconnected.com/pytorch-official-blog-detailed-pytorch-profiler-v1-9-7a5ca991a97b`