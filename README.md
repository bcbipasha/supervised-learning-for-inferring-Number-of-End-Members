# supervised-learning-for-inferring-Number-of-End-Members

This project introduced a supervised learning-based approach for inferring the endmember number in HSI data. The work presented a new strategy for NoE estimation and rigorously
evaluated it by conducting extensive experimentation on sev- eral data. The proposed method demonstrates superior per- formance over existing heuristic-based methods for inferring
the endmember count. This learning-based approach identifies the eigenvalue pattern and comprehends the subtle variations. In this work, we proposed a novel supervised learning framework that infers NoE corresponding to HSI data accord- ing to the eigenvalue pattern. As far as we know, rarely any work has inferred NoE using a supervised learning paradigm. This work constructed a suitable dataset by cropping and introducing noise into widely used hyperspectral unmixing data. The dataset contains eigenvalues and the corresponding Number of Endmembers (NoE). As per HSI unmixing models, the HSI data is represented as a linear or non-linear model of the endmembers. Since the endmembers are statistically independent components, the HSI data is low-rank. Here, the rank equals the number of endmembers. As per random matrix theory, the rank of a matrix can be inferred from the eigenvalues of the covariance matrix. Therefore, eigenvalues of the covariance matrix of the data play a vital role in determining NoE.
For Constructing Dataset we have used very popular HSI Dataset like JadperRidge,Samson,urban etc. For Running the code please find the following link to access tha dataset used for training the models

https://drive.google.com/drive/u/2/folders/1vRdKwHiTVAggYXNGObR7-AWmcGd_yA5T


