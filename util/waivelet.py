import pywt

def WT(data):
    x = data.reshape(data.shape[0],-1)
    (ca, cd) = pywt.dwt(x, "haar")
    cat = pywt.threshold(ca, 0.04, mode="soft")
    cdt = pywt.threshold(cd, 0.04, mode="soft")
    tx = pywt.idwt(cat, cdt, "haar")
    return tx
