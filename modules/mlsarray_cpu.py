import numpy as np
from scipy.fft import rfft2,irfft2,fft,ifft

class slicelist:
    def __init__(self,Nx,Ny):
        shp=(Nx,Ny)
        insl=[np.s_[0:1,1:int(Ny/2)],np.s_[1:int(Nx/2),:int(Ny/2)],np.s_[-int(Nx/2)+1:,1:int(Ny/2)]]
        shps=[[len(range(*(l[j].indices(shp[j])))) for j in range(len(l))] for l in insl]
        Ns=[np.prod(l) for l in shps]
        outsl=[np.s_[sum(Ns[:l]):sum(Ns[:l])+Ns[l]] for l in range(len(Ns))]
        self.insl,self.shape,self.shps,self.Ns,self.outsl=insl,shp,shps,Ns,outsl

class mlsarray(np.ndarray):
    def __new__(cls,Nx,Ny):
        v=np.zeros((Nx,int(Ny/2)+1),dtype=complex).view(cls)
        return v
    def __getitem__(self,key):
        if(isinstance(key,slicelist)):
            return [np.ndarray.__getitem__(self,l).ravel() for l in key.insl]
        else:
            return np.ndarray.__getitem__(self,key)
    def __setitem__(self,key,value):
        if(isinstance(key,slicelist)):
            for l,j,shp in zip(key.insl,key.outsl,key.shps):
                self[l]=value.ravel()[j].reshape(shp)
        else:
            np.ndarray.__setitem__(self,key,value)
    def irfft2(self):
        self.view(dtype=float)[:,:-2]=irfft2(self,norm='forward',overwrite_x=True)
    def rfft2(self):
        self[:,:]=rfft2(self.view(dtype=float)[:,:-2],norm='forward',overwrite_x=True)
    def ifftx(self):
        self[:,:]=ifft(self,norm='forward',overwrite_x=True,axis=0)
    def fftx(self):
        self[:,:]=fft(self,norm='forward',overwrite_x=True,axis=0)
        
def init_kspace_grid(sl):
    Nx,Ny=sl.shape
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]
    kyl=np.r_[0:int(Ny/2+1)]
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    kx=np.hstack([kx[l].ravel() for l in sl.insl])
    ky=np.hstack([ky[l].ravel() for l in sl.insl])
    return kx,ky

def irft2(uk,Npx,Npy,sl):
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u.irfft2()
    return u.view(dtype=float)[:,:-2]

def rft2(u,sl):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return np.hstack(uk[sl])

def irft(vk,Npx,Nx):
    v = np.zeros(int(Npx/2)+1, dtype='complex128')
    v[1:int(Nx/2)] = vk[:]
    return np.fft.irfft(v, norm='forward')

def rft(v,Nx):
    return  np.fft.rfft(v, norm='forward')[1:int(Nx/2)]

