# GPD-CM-for-SPP

## Saddle-point problem with a nonlinear coupling operator
$$
\min_{\mathbf{v}\in \mathbb{R}^{n}}\max_{\mathbf{w}\in \mathbb{R}^{m}_{+}}\mathcal{L}(\mathbf{v},\mathbf{w})=\Vert\mathbf{A}\mathbf{v}-\mathbf{a}\Vert^{2}+\langle\mathbf{w},e^{\mathbf{B}\mathbf{v}}-\mathbf{b}\rangle-\Vert\mathbf{C}\mathbf{w}-\mathbf{c}\Vert^{2},
$$

where the dimensions of constants are listed as $\mathbf{A}\in \mathbb{R}^{n\times n}$, $\mathbf{B}\in \mathbb{R}^{m\times n}$, $\mathbf{C}\in \mathbb{R}^{m\times m}$, $\mathbf{a}\in \mathbb{R}^{n}$, $\mathbf{b}\in \mathbb{R}^{m}$ and $\mathbf{c}\in \mathbb{R}^{m}$. 

In the experiment, the dimension of $(n,m)$ is set to $10\times10$, and all matrices and vectors are randomly generated in Matlab by using the command $\mathsf{rand(m,n)}$ while ensuring that $\mathcal{L}(\mathbf{v},\mathbf{w})$ is an SPP. The initial values of $(\mathbf{v},\mathbf{w})$ are set to $\mathbf{v}^{0}=2\cdot\mathsf{ones(n,1)}$ and $\mathbf{w}^{0}=\mathsf{ones(m,1)}$, respectively. The regularization parameters are set to $\mu=1, \sigma=0.95,$ and $\alpha=0.5$. In addition, all methods are implemented serially. 

## Proposed GPD_CM 

main.py is the proposed method.

main_AH.py is the Arrow-Hurwicz method.

GY_PDHGM is the variant PDHGM.

## Attention

1. run parameter_generate.py to create new parameters

2. run different methods 
