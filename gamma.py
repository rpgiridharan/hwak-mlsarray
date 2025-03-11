import cupy as cp

# def gam_p(ky,kap,C,D,nu):
#     om_p, om_m = 0.+0.j, 0.+0.j
#     kysq = ky**2
#     common = - 1j*C*(1+kysq)/(2*kysq) - 1j*kysq*(nu+D)/2
#     extra =  cp.sqrt(2*C*(nu-D)*(kysq-1) - C**2*(1+1/kysq)**2 - kysq**2*(nu-D)**2 + 4j*C*ky*kap/kysq)/2
#     om_p = common + extra
#     # om_m = common - extra
#     return cp.imag(om_p)

def gam_p(ky,kap,C):
    om_p = cp.zeros_like(ky, dtype=complex)
    kysq = ky**2
    common = - 1j*C*(1+kysq)/(2*kysq)
    extra =  cp.sqrt(- C**2*(1+1/kysq)**2 + 4j*C*ky*kap/kysq)/2
    om_p = common + extra
    return cp.imag(om_p)

def gammax(ky,kap,C):
    return cp.max(gam_p(ky,kap,C)).get()

def kymax(ky,kap,C):
    return ky[cp.argmax(gam_p(ky,kap,C))].get()   