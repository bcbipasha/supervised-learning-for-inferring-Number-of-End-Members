load('Urban_R162.mat');
X=Y'
X3D=reshape(X,307,307,162);
[p m n]=size(X3D);
data=reshape(X3D,n,p*m)
X3D=reshape(X,307,307,162);
[p m n]=size(X3D);
data=reshape(X3D,n,p*m);
SNR=20;
J_NOISY=awgn(data,SNR);
[NoEm,gap]=gapvd(J_NOISY