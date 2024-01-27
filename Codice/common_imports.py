# Librerie comuni del progetto
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
from Data_Set_Cancro import split_dataset
import matplotlib.pyplot as plt
from Data_Set_Cancro import import_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import warnings