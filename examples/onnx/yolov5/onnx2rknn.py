import os
import sys
import numpy as np
from rknn.api import RKNN
 
ONNX_MODEL = 'best_car_202208260827.onnx'
RKNN_MODEL = 'self.rknn'
 
if __name__ == '__main__':
 

    rknn = RKNN(verbose=True)
 
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], 
                std_values=[[255, 255, 255]], 
                reorder_channel='0 1 2', 
                target_platform='rk1808',
                quantized_dtype='asymmetric_affine-u8', 
                optimization_level=3,   
                output_optimize=1
        )
    print('done')
 
    '''
    param quantized_dtype: quantize data type, currently support: asymmetric_affine-u8, 
    dynamic_fixed_point-i8,dynamic_fixed_point-i16
    '''
 
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model  failed!')
        exit(ret)
    print('done')
 

    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build yolov5s failed!')
        exit(ret)
    print('done')
 
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5s.rknn failed!')
        exit(ret)
    print('done')
 
    rknn.release()
 