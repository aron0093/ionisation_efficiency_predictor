## threadsafe.py

Utility script to make generators thread safe for multi-processing. This is neccessary to avoid supplying duplicate batches for training while using the fit_generator function of keras with multi-processing set to True.

**Source:** [https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/](https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/)



