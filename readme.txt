python目录下：
sot-base.py：对应sot.cpp
idtd-base.py：对应idtd.cpp
sot-faster.py：速度大于500fps
idtd-faster.py：速度大于500fps
sot目录下：
封装process_image 函数，返回 （x,y）坐标元组
idtd目录下：
封装process_image 函数，返回 （x,y）坐标元组

distance目录下:
      idtd_distance.py：可在图像上画出真实值和预测值，并保存距离大于10的图像名称

      idtd》10.txt:保存了idtd预测值和真实标签距离大于10个像素的图像名称
      其中15997-16191.bmp连续195张识别不了

sot_distance.py:可在图像上画出真实值和预测值，并保存距离大于20的图像名称

      sot》20.txt:保存了idtd预测值和真实标签距离大于20个像素的图像名称
       其中09079-10244.tiff连续1000张基本识别不了


