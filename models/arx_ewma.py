
class ARX_EWMA:
    def __init__(self, lmbda=0.94, ridge=1e-2):
        self.lmbda=lmbda; self.ridge=ridge
        self.intercept_=0.0; self.coef_=None; self.sigma2_=1e-4
    def _prep(self, X):
        out=[]
        for row in X:
            out.append([0.0 if (v is None or (isinstance(v,float) and v!=v)) else float(v) for v in row])
        return out
    def fit(self, X, y):
        X=self._prep(X); y=list(y)
        n=len(X); d=len(X[0]) if n>0 else 0
        # Gram + ridge
        G=[[0.0]*(d+1) for _ in range(d+1)]
        b=[0.0]*(d+1)
        for i in range(n):
            xi=[1.0]+X[i]; yi=y[i]
            for a in range(d+1):
                b[a]+=xi[a]*yi
                for c in range(d+1):
                    G[a][c]+=xi[a]*xi[c]
        for j in range(d+1): G[j][j]+=self.ridge
        w=solve(G,b)
        self.intercept_, self.coef_=w[0], w[1:]
        # residuals
        resid=[]; 
        for i in range(n):
            mu=self.intercept_+sum(self.coef_[k]*X[i][k] for k in range(d))
            resid.append(y[i]-mu)
        ew=(sum(r*r for r in resid)/max(1,len(resid))) if resid else 1e-4
        lam=self.lmbda
        for r in resid: ew=lam*ew + (1-lam)*(r*r)
        self.sigma2_=max(ew,1e-8)
        return self
    def predict_params(self, X):
        X=self._prep(X)
        mu=[self.intercept_+sum(self.coef_[k]*xi[k] for k in range(len(xi))) for xi in X]
        s=(self.sigma2_**0.5)
        return mu, [s]*len(mu)

def solve(A, b):
    n=len(A)
    # augment
    for i in range(n):
        A[i]=A[i][:]+[b[i]]
    # Gauss-Jordan
    for col in range(n):
        piv=max(range(col,n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col])<1e-12: continue
        if piv!=col: A[col],A[piv]=A[piv],A[col]
        pivval=A[col][col]
        for j in range(col, n+1): A[col][j]/=pivval
        for r in range(n):
            if r==col: continue
            factor=A[r][col]
            for j in range(col, n+1):
                A[r][j]-=factor*A[col][j]
    return [A[i][n] for i in range(n)]
