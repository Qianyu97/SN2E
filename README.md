# SN2E
## 项目描述
### Union algorithm

```math
\begin{aligned}
\mu_{\cup} &= \frac{1}{n} \sum{\mu_i} \\
\Sigma_{\cup} &= \frac{1}{n} \sum{(\Sigma_i + (\mu_i - \mu_{\cup})^2)}
\end{aligned}
```

### Intersection algorithm

```math
\begin{aligned}
\mu_{\cap} &= \Sigma_{\cup} (\sum{\Sigma^{-}_i\mu_i} )\\
\Sigma_{\cap} &= (\sum{\Sigma_i^{-}})^{-}
\end{aligned}
```

### step 1:  Go up from leaf concept to attributes

### step 2:  Go down from attributes to leaf concept

### step 3:  Go up from leaf concept to trunk concept(keep gradient)

### step 4: iterate exclusion algorithm