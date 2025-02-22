import os
from transformers import logging as transformers_logging
import torch

# Temporary solution: Look for some other alternative
torch.classes.__path__ = []

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
transformers_logging.set_verbosity_error()

import streamlit as st

pg = st.navigation(
    {
        "Dashboard":[
            st.Page("input_text.py", title = "Single Text Analysis"), 
            st.Page("batch_analysis.py", title = "Batch Analyzer")
        ]
    }
)
pg.run()