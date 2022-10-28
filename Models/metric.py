import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class Repr():
    def __init__(self):
        pass
    def __repr__(self):
        arg1 = '__repr__는 객체를 출력할 때 보여지는 문자열입니다.'
        arg2 = '한 번 해보겠습니다.'
        return f'{arg1} \n {arg2}'

# print(sys.path())
def stratified_kFold(df, n_splits=5, random_state=42):
    pass

if __name__ == '__main__':
    print(os.getcwd())
    print('k-fold')
    repr = Repr()
    print(repr)