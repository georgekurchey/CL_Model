class PinballLinear:
    def __init__(self, tau=0.5, lr=0.05, epochs=5, l2=1e-3, batch=256, seed=42):
        self.tau=tau; self.lr=lr; self.epochs=epochs; self.l2=l2; self.batch=batch; self.seed=seed
        self.w=None; self.b=0.0
    def _prep(self, X):
        out=[]
        for row in X:
            out.append([0.0 if (v is None or (isinstance(v,float) and v!=v)) else float(v) for v in row])
        return out
    def fit(self, X, y):
        X=self._prep(X); y=list(y)
        n=len(X); d=len(X[0]) if n>0 else 0
        self.w=[0.0]*d; self.b=0.0
        rng=_rng(self.seed); idx=list(range(n))
        for _ in range(self.epochs):
            _shuffle(idx, rng)
            for i0 in range(0,n,self.batch):
                sel=idx[i0:i0+self.batch]
                if not sel: continue
                gw=[0.0]*d; gb=0.0
                for i in sel:
                    pred=_dot(self.w,X[i])+self.b
                    u=y[i]-pred
                    g=-(self.tau - (1.0 if u<0 else 0.0))
                    for j in range(d): gw[j]+=g*X[i][j]
                    gb+=g
                m=float(len(sel))
                for j in range(d): gw[j]=gw[j]/m + self.l2*self.w[j]
                gb/=m
                for j in range(d): self.w[j]-=self.lr*gw[j]
                self.b-=self.lr*gb
        return self
    def predict(self, X):
        X=self._prep(X)
        return [_dot(self.w,xi)+self.b for xi in X]

class QuantileStack:
    def __init__(self, taus, lr=0.05, epochs=5, l2=1e-3, batch=256, seed=42):
        self.taus=sorted(taus)
        self.models=[PinballLinear(tau=t, lr=lr, epochs=epochs, l2=l2, batch=batch, seed=seed) for t in self.taus]
    def fit(self, X, Y):
        for m in self.models: m.fit(X,Y)
        return self
    def predict(self, X):
        if not X: return []
        # return raw (unsorted); walkforward enforces monotonicity at the end
        return [[m.predict([x])[0] for m in self.models] for x in X]

def _dot(w,x): return sum((wi*xi for wi,xi in zip(w,x)), 0.0)

def _rng(seed):
    class R:
        def __init__(self,s): self.s=s
        def rand(self):
            self.s=(1664525*self.s+1013904223)%(2**32)
            return self.s/(2**32)
    return R(seed)

def _shuffle(a, rng):
    for i in range(len(a)-1,0,-1):
        j=int(rng.rand()*(i+1)); a[i],a[j]=a[j],a[i]
