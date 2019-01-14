# stat_car_object_detection
基于深度学习框架的高速路上车流量的实时统计
# windows平台代码运行
  
1、进入lib文件夹下，修改setup.py文件，因为setup.py文件同时编译了又cpu和GPU的情况。本台电脑没有GPU，所以不需要编译GPU，
   如果直接编译，会报错：OSError: The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDA_PATH
  修改：要注释的代码
  1：#CUDA = locate_cuda()  57行
  2：#customize_compiler_for_nvcc(self.compiler) 110行
    
2、执行Makefile文件中的 python setup.py build_ext --inplace
  
3、注释掉nms_wrapper.py中的  17行 # from nms.gpu_nms import gpu_nms
     #if cfg.USE_GPU_NMS and not force_cpu:   25
     #return gpu_nms(dets, thresh, device_id=0)  26
  
4、执行官网的demo  直接运行tools/demo.py文件
  报错处理：
  报错：ValueError: Buffer dtype mismatch, expected 'int_t' but got 'long long'
  解决办法：打开lib/nns/nms.pyx，将第25行的np.int_t修改为np.intp_t。然后重新执行setup.py

# ubuntu平台代码运行
  
1、先检查pip3是否安装,命令：pip3 –version, 如果不存在就要安装pip3，执行apt-get update，然后执行：sudo apt-get install python3-dev（为后面make操作，注意必须是pyhton3-dev，不能是python-dev），Python2 和 pip2 存在， Python3 也存在，但是 pip3 不存在的解决办法：
    sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    sudo python3  get-pip.py
      
2、先安装环境，执行以上pip3命令：  pip3 install cython
                                  pip3 install opencv-python
                                  pip3 install matplotlib
                                  pip3 install easydict
                                  pip3 install Pillow
                                  pip3 install scipy
                                  pip3 install tensorflow-gpu ==1.3.0    
                                  pip3 install tensorflow-tensorboard == 0.1.8
  
3、可以安装anaconda，在anaconda里面创建各种隔离环境
  
4、执行git clone https://github.com/endernewton/tf-faster-rcnn.git  下载源码或自己直接下载好传到环境中
  
5、cd stat_car_object_detection/lib 进入lib包执行make操作，这里注意因为是用python3，所以需要修改Makefile文件内容，将文件中的python都修改问python3，分别执行命令：make clean   make
  
6、安装 Python COCO API，这是为了使用COCO数据库，分别执行命令：cd data ->git clone https://github.com/pdollar/coco.git ->cd coco/PythonAPI 下面执行make操作前，注意和4步一样修改Makefile文件，将python全部修改为python3，然后执行 make命令
  
7、下载预训练模型Resnet101 for voc pre-trained on 07+12 set，执行命令：./data/scripts/fetch_faster_rcnn_models.sh，或自己下载voc_0712_80k-110k.tgz包放到data目录下，解压：tar xvf voc_0712_80k-110k.tgz
  
8、回到stat_car_object_detection根目录下，建立预训练模型的软连接：
      NET=res101
      TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
      mkdir -p output/${NET}/${TRAIN_IMDB}
      cd output/${NET}/${TRAIN_IMDB}
      ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
	
9、回到根目录：执行命令：python3 tools/demo.py ，这里会报错RuntimeError: Invalid DISPLAY variable，这是因为matplotlib 输出的图像没有输出来，解决办法是修改demo.py中的代码，使得不输出图像：



