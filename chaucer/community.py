#!/usr/bin/env python3

"""
Performs community detection on preprocessed orthographic tabular data.
"""

import argparse
import logging
from logging import DEBUG, INFO, WARNING, ERROR
from typing import Any, Dict, Optional, Union

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap plotting




