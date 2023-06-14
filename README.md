# Entrenar la red 

Usar codigo en `evalCustom.py`, obs: es necesario descomentar la parte que dice "Entrenamiento del modelo" -- ya hare un archivo aparte.

# Construir ONNX

Correr `onnx_transform.py`, recorad cambiar el BATCH_SIZE segun sea necesario.

# Construir ENGINEE

Correr `build_trt.py`, tmb recuerde cambiar BATCH_SIZE, asi como especificar si desa fp32, fp16 o INT8

```
build_trt.py --int8 --input_shape=[128,1, 28, 28] --weights='weights/best128.onnx'
``` 
obs: 128 corresponde al batch size

---

obs: para construir con INT8 fue necesario crear una funcion de preprocesamiento de imagenes para la calibracion del engine en int8, esta funcion de pre procesamiento se encuentra en `processing.py`y depende de como fue entrenado el dataset, ahi aplique la misma normalizacion aplicada al entrenar el modelo dando buenos resultados.

obs: el codigo para la calibracion de int8 por ahora se encuentra en `models/engine.py` quizas dsp lo ordene mejor.

# Pruebas de TensorRT en una red custom usando pytorch

correr red normal: `python evalCustom.py`
 
correr red optimizada con TRT: `python evalCustomNetRT.py`

para poder ver los logs generados en los profilers: `tensorboard --logdir=./log`, luego en el navegador de chrome, buscar `http://localhost:6006/`

# Resultados
## Pythorch Profiler
### Plataforma: Jetson Xavier
* CPU: ARMv8 Processor rev 0 (v8l) × 4
* GPU: NVIDIA Tegra Xavier (nvgpu)/integrated
* RAM: 16 Gb
* SO: Ubuntu 20.04
Estos resultados son obtenidos al correr todo el dataset de purebas (~ 10000 imagenes).
### summary table batch size 2048
|  Model      | Stage           |Time duration (s)     | size (MB)| accuracy (%)|
|-------------|-----------------|----------------------|-----------|-------------|
| Vanilla     |                 |                      |1,7        |99.02        |     
|             | CPU Total       |  17.975              |||
|             | CUDA Total      |  17.982              |||
| TRT fp32    |                 |                      |1.8        |98.97        |
|             | CPU Total       |  13.952              |||        
|             | CUDA Total      |  13.951              |||   
| TRT fp16    |                 |                      |0.907        |98.97      |
|             | CPU Total       |  14.071              |||        
|             | CUDA Total      |  14.070              |||     
| TRT int8    |                 |                      |0.499       |97.70       |
|             | CPU Total       |  14.122              ||        
|             | CUDA Total      |  14.121              |||     


---

## TensorBoard Profiler
### Plataforma: PC Escritorio
* CPU: i3 1200F
* GPU: Nvidia RTX 3060 
* RAM: 32 Gb 3200 MHz
* SO: Ubuntu 22.04
Estos resultados son obtenidos al correr todo el dataset de purebas (~ 10000 imagenes).
### summary table batch size 2048

|  Model      | Stage           |Time duration (us)    | Percentage (%) |  size (MB)| accuracy (%)|
|-------------|-----------------|----------------------|----------------|-----------|-------------|
| Vanilla     |                 |  1.150.770           |100             |1,7        |99.2         |     
|             | Kernel          |  38.751              | 3.37           |||
|             | Memcpy          |  9.722               | 0.83           |||
|             | CPU Exec        |  471.347             | 40,96          |||
|             | Other           |  630.946             | 54.83          |||
| TRT fp32    |                 |  2.155.355           |100             |2,2        |98.97         |
|             | Kernel          |  19,211              |0.89            |||        
|             | Memcpy          |  9,747               |0.45            |||     
|             | CPU Exec        |  1.505.493           |69.85           |||
|             | Other           |  620.904             |28.81           |||
| TRT fp16    |                 |  2,170,024           |100             |1,1        |98.97        |
|             | Kernel          |  8,378	             |0.39            |||        
|             | Memcpy          | 9,706                |0.45            |||     
|             | CPU Exec        |  1,521,359           |70.11           |||
|             | Other           |  630,581             |29.06           |||
| TRT int8    |                 |  3,118,408           |100             |0.894      |97.86        |
|             | Kernel          |  4,722               |0.15            |||        
|             | Memcpy          | 9,690                |0.31            |||     
|             | CPU Exec        |  2,466,560           |79.1            |||
|             | Other           |  637,436             |20.44           |||


* obs: Kernel: kernel execution time on GPU device;
       Memcpy: GPU involved memory compy time;
       Memset: GPU involved memory set time;
       Runtime: CUDA runtime execution time on host side (such as cudaLaunchKernel);
       DataLoader: The data loading time spent in PyTorch DataLoader object;
       CPU Exec: Host compute time, including every Pytorch operator running time;
       Other: time not incuded in the above
* obs: los que no aparecen en la tabla son porque son 0%

--- 

### vanilla trace batch size 2048

<div align="center">
      <a href="">
     <img
      src="img_readme/trace_vanilla.png"
      alt="Trace Vanilla"
      style="width:100%;">
      </a>
</div>

### trt trace batch size 2048

<div align="center">
      <a href="">
     <img
      src="img_readme/trace_trt.png"
      alt="Trace trt"
      style="width:100%;">
      </a>
</div>

---

### vanilla mem batch size 2048

<div align="center">
      <a href="">
     <img
      src="img_readme/memory_vanilla.png"
      alt="Trace Vanilla"
      style="width:100%;">
      </a>
</div>

### trt mem batch size 2048

<div align="center">
      <a href="">
     <img
      src="img_readme/memory_trt.png"
      alt="Trace trt"
      style="width:100%;">
      </a>
</div>

---
## tablas antiguas
### CustomNet batch size 64
|  Model      | Stage           |size MB |Time CPU %           | Time CUDA % |# of calls | accuracy % | 
|-------------|-----------------|--------|---------------------|-------------|-----------|------------|
| Vanilla     |                 |1.7     |100% (87 ms)         |100% (351 us)|           |100%        |
|             | inference       |        |~48.00%              |100%         |           ||
|             | cudaLaunchKernel|        |26.81%               |0%           |15         ||
|             | cudaFree        |        |11.68%               |0%           |4          ||
| TRT fp16    |                 |1.1     |100% (5.425 ms)      |100% (72 us) |           |100%        |
|             | inference       |        |0%                   |100%         |           ||
|             | cudaLaunchKernel|        |63.14%               |0%           |3          ||
|             | cudaFree        |        |36.18%               |0%           |2          ||


### CustomNet batch size 128
|  Model      | Stage           |size MB |Time CPU %           | Time CUDA % |# of calls | accuracy % | 
|-------------|-----------------|--------|---------------------|-------------|-----------|------------|
| Vanilla     |                 |1.7     |100% (81.809 ms)     |100% (613 us)|           |99.218%        |
|             | inference       |        |~50.00%              |100%         |           ||
|             | cudaLaunchKernel|        |26.68%               |0%           |13         ||
|             | cudaFree        |        |11.83%               |0%           |4          ||
| TRT fp16    |                 |1.1     |100% (3.480 ms)      |100% (131 us) |           |99.218%        |
|             | inference       |        |0%                   |100%         |           ||
|             | cudaLaunchKernel|        |64.97%               |0%           |3          ||
|             | cudaFree        |        |34.45%               |0%           |2          ||

---

obs: * resultados obtenidos corriendo las redes en un solo batch de 64/128 imagenes.
    * El tamaño del modelo Vanilla hace referencia al modelo pasado a onnx.
    * Stage inference hace referencia al tiempo de procesamiento usado en las capas de la red para la inferenica en las imagenes
    
# Instalacion de SO en Jetson Xavier AGX

## instalar mediante JetPack SDK Manager:

* instalar SDK Manager mediante este link: `https://developer.nvidia.com/sdk-manager`

obs: es necesario tener SO Ubuntu 20.04, ya que no acepta una version posterior.

* Conectar la tarjeta al PC donde se instalo el SDK Manager mediante el puerto C frontal usando el cable usb-c a usb que viene en la caja del producto.
* Presionar los botones de power y reset de la tarjeta por unos segundos y luego soltarlos para resetear la tarjeta lo que permitira al SDK manager reconocer la tarjetam una vez la reconoce sera posible seguir los pasos del SDK hasta que la instalacion este terminada.

# Instalacion de CustomNet en Jetson Xavier AGX

Una vez terminada la instalacion mediante SDK manager, deberias ser capaz de hacer `sudo apt-get update` y `sudo apt-get upgrade`, luego de instalar python3 en los siguientes pasos, deberias poder hacer `python3` e importar tensorrt  `import tensorrt as trt` sin problemas, lo que signidica que tensorrt esta instalado (estte deberia venir en el sistema instalado por SDK Manager).

* install pip: en la Terminal de la tarjeta usar el siguiente comando
```
sudo apt install python3-pip
```

* instalar python3-dev para evitar errores en otras instalaciones segun `https://stackoverflow.com/questions/26053982/setup-script-exited-with-error-command-x86-64-linux-gnu-gcc-failed-with-exit`
```
sudo apt-get install python3-dev
```
obs: este ya deberia estar instalado x los pasos anteriores.

* Necesario para algunos errores como `ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory`
```
sudo apt-get install libopenblas-dev
```

* Instalar pytorch descarga el wheel deseado de esta pagina: `https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048` y en la terminal, dentro de la carpeta donde se descargo el archivo ejecutar:
```
sudo pip install archivo.whl
```
obs: es necesario usar `sudo`para que se instale dentro de los archivos de `usr/local/python3..` en vez de los archivos de `home`.

para probar la instalacion entra desde la terminal usando `python3`dentro de la terminal de python3:
```
import torch
torch.cuda.is_available()
```
a lo cual deberia decir "True"

* Instala nano `sudo apt-get install nano`
con `nano ~/.bashrc` revisa que al final de este archivo se exporte las versiones de cuda correspondientes:
```
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-11.4/targets/aarch64-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:$LIBRARY_PATH
```
luego, para que los cambios sean efectivos:
```
source ~/.bashrc
```
(revisar `https://forums.developer.nvidia.com/t/pycuda-installation-failure-on-jetson-nano/77152/11`)
para verificar que este paso se hizo correctamente al correr `nvcc --version` en cosola, deberia decir la version de cuda instalada.

* Instalar `torchvision` segun `https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048/13`
```
git clone https://github.com/pytorch/vision
cd vision
sudo python3 setup.py install
```
obs: para la tarjta jetson TX2, es necesario instalar una version anterior, ya que esta solo trabaja con torch 1.10.0, segun `https://medium.com/hackers-terminal/installing-pytorch-torchvision-on-nvidias-jetson-tx2-81591d03ce32`
```
git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
```

* Install `pycuda` 

```
pip install pycuda --user
```
obs: no pudiemos hacer la instalacion usando sudo, por algun error, para poder usar pycuda y que lo reconozca nvida nsight (para hacer profiling) es necesario linkear este paquete al resto de paquetes instalados con sudo con el siguiente comando:
```
sudo ln -s /home/<user_name>/.local/lib/python3.8/site-packages/pycuda /usr/local/lib/python3.8/dist-packages
```
* instalar onnx:
```
sudo pip install onnx
```

obs: para installar onnx en jetson TX2 es necesario:
```
sudo apt update
sudo apt-get install python3-pip
sudo apt-get install cmake libprotobuf-dev protobuf-compiler
sudo pip3 install Cython
sudo pip3 install onnx==1.4.1
```
segun `https://forums.developer.nvidia.com/t/can-not-install-onnx-1-4-1-on-jetson-tx2/173354/5`

* Descargar el repo con git clone ...
* Dejare un link con el dataset para calibrar int8 y un link con el peso best.pth para no tener que entrenar la red en la tarjeta... luego de descargar esos se puede proceder con lo siguientes pasos
* Transformar los pesos a onnx con `python3 onnx_transform.py`
* Crear el engine con `python3 build_trt --int8` (si no esta descargado el dataset de prueba, añadir download=True en la linea donde se carga el dataset)
* Probar el engine con `python3 evalCustomNetRT`
---
# Profiling en la Jetson Xavier
* De los pasos anteriores ya se deberia de generar el profile, para poder ver los resultados instalar tensorboard con `pip install torch_tb_profiler`
* Para poder visaulizarlos, se necesita chrome, por lo que debes instalarlo: `sudo apt install chromium-browser`
* correr en la terminal `tensorboard --logdir=./log` y buscar en el navegador de chrome `http://localhost:6006/`

--- 
# VNC SERVER 
* Nos conectaremos a la jetson Xavier mediante VNC siguiento este tutorial: `https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup`:
* Enable the VNC server to start each time you log in
```
cd /usr/lib/systemd/user/graphical-session.target.wants
sudo ln -s ../vino-server.service ./.
```
* Configure the VNC server
```
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false
```
* Set a password to access the VNC server
```
# Replace thepassword with your desired password
gsettings set org.gnome.Vino authentication-methods "['vnc']"
gsettings set org.gnome.Vino vnc-password $(echo -n 'thepassword'|base64)
```
* Reboot the system so that the settings take effect
```
sudo reboot
```
* Luego para conectarse del lado del PC de escritorio, usando ubuntu:
```
sudo apt update
sudo apt install gvncviewer
gvncviewer <IP de la Jetson>
```
--- 

obs:
* Es necesario usar python3 para ambas tarjetas, en lugar de solo python
* Es necesario usar pip3 en lugar de pip para la jetson TX2

---

# Referencias

* pytorch profiler stable: `https://pytorch.org/docs/stable/profiler.html`
* profiler pytorch: `https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html`
* profiler pytorch blog: `https://levelup.gitconnected.com/pytorch-official-blog-detailed-pytorch-profiler-v1-9-7a5ca991a97b`
* profiler tensor board: `https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html`
* INT8 calibration examples: `https://forums.developer.nvidia.com/t/tensorrt-5-int8-calibration-example/71828/9`
* INT8 calibration github example: `https://github.com/rmccorm4/tensorrt-utils/blob/master/int8/calibration/ImagenetCalibrator.py`
