# COMP3710 Report

**Benjamin Bateman**
48819101

---

## Section 3 — Feasability Check

### a)

The primal formulation of Support Vector Regression (SVR) has loss function $\max\{0, |y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)|-\varepsilon\}$. As $\varepsilon\to0$, the loss function approaches $\max\{0, |y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)|\}$. As this is never below 0, we have a loss function of $|y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)|$. This is the same as the absolute error ($\ell_1$) loss function. Therefore, the SVR converges to the optimisation problem:

$$\min_{\boldsymbol{\theta}} \frac1n\sum^n_{i=1}|y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)| +\lambda\|\boldsymbol{\theta}\|_2^2$$

The $\varepsilon$ insensitive loss function used for Support Vector Regression is the absolute loss function, but with $\ell(y, \hat{y}) = 0 \text{ when } |y-\hat{y}|\le\varepsilon$. As $\varepsilon \to 0$, the range of this '0' region shrinks, concentrating around $x=0$. At this point, it becomes $\ell_1$, the absolute loss function.

### b)

$\varepsilon$ insensitive loss ignores errors smaller than a certain threshold $\varepsilon$. For the primal SVR problem, this is $\ell_\varepsilon = \max\{0, |y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)|-\varepsilon\}$. To find what contributes to the gradient (and subgradient), the differentials and subdifferentials will be calculated. There are 3 cases. For simplicity's sake, $|y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)|$ will be referred to as $d$.

**Case 1: $|d| < \varepsilon$**

Here, $\ell_\varepsilon(d)=0$, so $\nabla\ell_\varepsilon(d)=0$.
Therefore, this case does not contribute to the gradient.

**Case 2: $|d| > \varepsilon$**

Here, $\ell_\varepsilon(d)$ is differentiable, with subdifferential

$$\partial\ell_\varepsilon(d) = \{1\} \text{ if } d>\varepsilon \text{ and } \{-1\} \text{ if } d < \varepsilon$$

**Case 3: $|d| = \varepsilon$**

This case is non-differentiable. We find all $g$ such that

$$|d|\ge gd$$

If $d > 0$:

$$d\ge gd \Longrightarrow g\le 1$$

If $d < 0$: write $d = -|d|$, so

$$|d|\ge g(-|d|) \Longrightarrow 1\ge -g \Longrightarrow g \ge-1$$

Thus, $-1\le g\le 1$, and

$$\partial|d| = [-1, 1]$$

This means that the point where $|y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)| = \varepsilon$ affects the gradient.
Therefore only points with $|y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)| \ge \varepsilon$ contribute to the gradient.

### c)

#### (i)

$\psi(r)$ can be calculated by deriving with respect to $r$.

**Kernel Ridge Regression (KRR):**

$$\ell(r)=r^2$$

$$\frac{d}{dr}(\ell(r))=\psi(r)=2r$$

**Kernelised Least Absolute Deviations (KLAD):**

$$\ell(r)=|r|$$

$$\frac{d}{dr}(|r|)=\psi(r)=\begin{cases}
-1 & r< 0 \\
[-1, 1] & r = 0\\
1& r>0
\end{cases}$$

**Support Vector Regression (SVR):**

$$\ell_\varepsilon(r)=\max\{0, |r|-\varepsilon\}$$

$$\frac{d}{dr}(\ell_\varepsilon(r))=\psi(r)=\begin{cases}
-1 & r< -\varepsilon \\
0& -\varepsilon< r < \varepsilon\\
1 & r > \varepsilon\\
[-1, 1] & |r|=\varepsilon
\end{cases}$$

#### (ii)

**KRR:** $\psi(r) = 2r$, so a large residual influences proportional to $|r|$. Therefore, a large outlier will influence the gradient heavily, pulling the update away from the optimal solution.

**KLAD:** As $\psi(r)=1$ for all $r\ne0$, every point has the same weight as any other point. This way, an outlier will not dominate the update.

**SVR:** Points that lie within $-\varepsilon$ and $\varepsilon$ are 0 and contribute nothing to the gradient. Points outside the '0' zone act in the same way as KLAD, where every point contributes the same amount to an update.

#### (iii)

KRR is not robust to outliers, as a large outlier with a large residual will dominate an update. The squared loss grows quadratically, and $\psi(r) = 2r$ is unbounded.

KLAD is robust to outliers as its absolute loss function grows linearly, and its $\psi(r)$ is bound between -1 and 1. No point has more of an influence than any other point.

SVR is robust as the $\varepsilon$ insensitive loss function grows linearly when $|r|>\varepsilon$, so its $\psi(r)$ is bound between -1 and 1.

### d)

For all code, see the attached Jupyter notebook.

#### (i)

scikit-learn does not have a KLAD function, but it was proved in part a) that KLAD is just SVR with epsilon 0. This was used to train the KLAD model.

![Question 1 d)(i)](image.png)

#### (ii)

![Fitted kernel models](kernel_models.png)

![MSEs](mses.png)

The fitted models' plots and MSEs can be found above. On the clean dataset, there was very little difference between the models, with all having around a 0.034 train MSE and a 0.006 test MSE. Visually, there is no obvious difference too.

On the corrupted dataset, all still performed relatively well, with both KLAD and SVR maintaining almost identical test MSEs. KRR performed the worst, visually deviating from the other models in the corrupted dataset. Its test MSE jumped from 0.0068 to 0.0176, a big difference. All train MSE values significantly increased, which is to be expected with so many outliers.

All functions maintained smoothness, with no irregularities or jumpiness that would be seen in a highly overfit model.

#### (iii)

![Epsilon sensitivity](diii.png)

As epsilon increases, the robustness increases, but so does the inaccuracy of the data. As epsilon increases past 1, the model heavily underfits the data, creating an overly-smoothed model that is not at all helpful. This is seen in both the number of support vectors and MSE score. Near 0, like epsilon=0.01, there are 92 support vectors and there exists a test MSE of 0.0052. At epsilon = 1 however, there are just 15 support vectors and there exists a test MSE of 0.42.

In at least this data, it seems that the lower the epsilon the better. It should be noted that at epsilon=0, it becomes the KLAD model, which was the best performing in part (ii). The lower the epsilon, the less robust the model is to outliers, so in other cases with perhaps more severe outliers, the optimal epsilon is larger than it is here.

---

## Question 2 — Multi-Output Gaussian Processes (MOGP)

### a)

#### (i)

$$\mathbf{K}_x \otimes\mathbf{B} =
\begin{bmatrix}
[\mathbf{K}_x]_{11}\mathbf{B}  & \dots & [\mathbf{K}_x]_{1n}\mathbf{B} \\
\vdots & \ddots & \vdots\\
[\mathbf{K}_x]_{n1}\mathbf{B}  & \dots & [\mathbf{K}_x]_{nn}\mathbf{B}
\end{bmatrix}$$

As $[\mathbf{K}_x]_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$, each $(i, j)$ block is $k(\mathbf{x}_i, \mathbf{x}_j)\mathbf{B}$, which is equal to $\mathbf{K}(\mathbf{x}_i, \mathbf{x}_j)$. As this is true for all $(i, j)$, $\tilde{\mathbf{K}} = \mathbf{K}_x \otimes\mathbf{B}$.

#### (ii)

As $k$ models input similarity, and $\mathbf{B}$ models output correlation, this structure shows that they do not interact at all. The output correlation is identical throughout all $(i, j)$ in the input space, meaning that the correlation between two outputs is the same everywhere.

#### (iii)

The diagonal values of $\mathbf{B}$, $\mathbf{B}_{pp}$, where $p \in \{1, 2, \dots, m\}$ represent the variance of output $p$. The off diagonals, $\mathbf{B}_{pq}$, represent the covariance between outputs $p$ and $q$. A value greater than 0 means $p$ and $q$ tend to be positively correlated (move together), a value less than 0 means $p$ and $q$ tend to be negatively correlated (move in opposite directions), and a value of 0 means they have no correlation at all.

### b)

#### (i)

$$\tilde{\mathbf{y}} = \tilde{\mathbf{f}} + \tilde{\boldsymbol{\varepsilon}}$$

Here, $\tilde{\mathbf{f}} \sim \mathcal{N}(0, \tilde{\mathbf{K}})$ and $\tilde{\boldsymbol{\varepsilon}} \sim \mathcal{N}(0, \sigma^2\mathbf{I}_{nm})$. $\tilde{\mathbf{y}}$ is then a sum of two independent Gaussian distributions:

$$\tilde{\mathbf{y}} \sim \mathcal{N}(0+0, \tilde{\mathbf{K}}+\sigma^2\mathbf{I}_{nm})$$

$$=\mathcal{N}(0, \tilde{\mathbf{K}}+\sigma^2\mathbf{I}_{nm})$$

Therefore, $\tilde{\mathbf{y}}$ is Gaussian with covariance matrix $\tilde{\mathbf{K}}+\sigma^2\mathbf{I}_{nm}$.

#### (ii)

A $\sigma^2$ close to 0 will lead to a model that is neither smooth nor robust, as outliers will have strong negative impacts. As $\sigma^2$ increases, noise is added to the model, making the distribution less reliant on the input data. This reduces the impact of outliers, increasing robustness. It also smooths out the model. At a very large $\sigma^2$, the model will be too smooth and robust, not following the training data at all. Therefore, a balance must be reached when choosing the value of $\sigma^2$.

### c)

#### (i)

We have:

$$f_q(\mathbf{x}) = \sum^R_{r=1}w_{qr}u_r(\mathbf{x}), \quad q=1,\dots,m$$

where $u_r(\mathbf{x})\sim GP(0, k_r(\mathbf{x}, \mathbf{x}'))$, and:

$$[\mathbf{K}(\mathbf{x}, \mathbf{x}')]_{pq}=\operatorname{Cov}(f_p(\mathbf{x}), f_q(\mathbf{x}'))$$

Substitute $f_p$ and $f_q$ into the covariance formula. The variable $s$ will be introduced to differentiate $f_p$ from $f_q$.

$$\operatorname{Cov}(f_p(\mathbf{x}), f_q(\mathbf{x}')) = \operatorname{Cov}\left(\sum^S_{s=1}w_{pr}u_r(\mathbf{x}), \sum^R_{r=1}w_{qr}u_r(\mathbf{x}')\right)$$

We can take the sums and constants out, leaving

$$\operatorname{Cov}(f_p(\mathbf{x}), f_q(\mathbf{x}')) = \sum^R_{s=1}\sum^R_{r=1}w_{ps}w_{qr}\operatorname{Cov}(u_s(\mathbf{x}), u_r(\mathbf{x}'))$$

As $u_s(\mathbf{x})$ and $u_r(\mathbf{x}')$ are independent Gaussian processes for all $r, s \in R$,

$$\operatorname{Cov}(u_s(\mathbf{x}), u_r(\mathbf{x}')) = \begin{cases}
k_r(\mathbf{x}, \mathbf{x}') & r=s\\
0& r\ne s
\end{cases}$$

As the 0 cases do not contribute to the sums, we are summing only values when $r=s$. Therefore:

$$[\mathbf{K}(\mathbf{x}, \mathbf{x}')]_{pq} = \sum^R_{r=1}w_{pr}w_{qr}k_r(\mathbf{x}, \mathbf{x}')$$

#### (ii)

With $R=1$, the LMC becomes:

$$[\mathbf{K}(\mathbf{x}, \mathbf{x}')]_{pq} = w_{p1}w_{q1}k(\mathbf{x}, \mathbf{x}')$$

$w_pw_q$ is the $(p, q)$ position of the $m\times m$ matrix $\mathbf{w}_1\mathbf{w}_1^\top$. Therefore, we can rewrite this as:

$$\mathbf{K}(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}')\mathbf{w}_1\mathbf{w}_1^\top$$

Suppose $\mathbf{B} = \mathbf{w}_1\mathbf{w}_1^\top$. For $R=1$ to be the special separable space, $\mathbf{B}$ would have to be positive semi-definite. As $\mathbf{B}$ is symmetric, we have to check that for any $\mathbf{x} \in \mathbb{R}^m$, $\mathbf{x}^\top \mathbf{Bx} \ge0$.

$$\mathbf{x}^\top \mathbf{Bx} = \mathbf{x}^\top (\mathbf{w}_1\mathbf{w}_1^\top)\mathbf{x} = (\mathbf{x}^\top\mathbf{w}_1)(\mathbf{w}_1^\top\mathbf{x})=(\mathbf{w}_1^\top\mathbf{x})^2 \ge 0$$

Therefore, $\mathbf{B}$ is a positive semi-definite matrix, and $\mathbf{K}(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}')\mathbf{B}$, which is the formula found in part a) for the separable kernel.

#### (iii)

At $R=1$, the separable kernel, relationships between outputs are determined entirely by one weight vector. There may be more than one underlying structure, however. Multiple latent processes can help map the multiple different underlying processes found in the outputs. Each process has a new kernel $k_r(\mathbf{x}, \mathbf{x}')$ which means different outputs can share different input-space structures. The equation

$$[\mathbf{K}(\mathbf{x}, \mathbf{x}')]_{pq} = \sum^R_{r=1}w_{pr}w_{qr}k_r(\mathbf{x}, \mathbf{x}')$$

shows that for each new input space, you can assign different weights to each pair $(p, q)$, but only by changing the input space. With $R=1$, we only have one input space, and so cannot map the complexities in the correlation between outputs.

### d)

#### (i)

The Multi-Output Gaussian Processes framework naturally handles missing outputs because it is a joint Gaussian distribution over all outputs at all input locations. By having a distribution, we already have a prediction for every output at every input. If a point $\mathbf{x}_i$ only has a subset of the outputs, the model still updates the posterior over missing outputs via the cross-covariance structure $\mathbf{K}$. Knowing the missing value would help, but when we don't have access to it, the structure of MOGP allows us to handle it without imputation by keeping them as unobserved variables whose posterior is informed by the other outputs.

#### (ii)

$\tilde{\mathbf{f}}$ is a multivariate normal with covariance matrix $\tilde{\mathbf{K}}$. This means that $\tilde{\mathbf{f}} \sim \mathcal{N}(0, \tilde{\mathbf{K}})$.

Therefore,

$$\tilde{\mathbf{f}} =
\begin{bmatrix}
\mathbf{f}_{\text{obs}}\\
\mathbf{f}_{\text{miss}}
\end{bmatrix} \sim \mathcal{N}\left(0, \begin{bmatrix}
\mathbf{K}_{\text{obs}\cdot \text{obs}} & \mathbf{K}_{\text{obs}\cdot\text{miss}}\\
\mathbf{K}_{\text{miss}\cdot\text{obs}} & \mathbf{K}_{\text{miss}\cdot\text{miss}}
\end{bmatrix}\right)$$

#### (iii)

$$\mathbf{f}_{\text{miss} | \text{obs}} \sim \mathcal{N}(\mu_{\text{miss} | \text{obs} }, \Sigma_{\text{miss} | \text{obs} })$$

Where the conditional mean is:

$$\mu_{\text{miss} | \text{obs} } = \mu_{\text{miss}} +\mathbf{K}_{\text{miss}\cdot\text{obs}}\mathbf{K}_{\text{obs}\cdot \text{obs}}^{-1}(\mathbf{f}_\text{obs}-\mu_\text{obs})$$

$$=\mathbf{K}_{\text{miss}\cdot\text{obs}}\mathbf{K}_{\text{obs}\cdot \text{obs}}^{-1}\mathbf{f}_\text{obs}$$

and the conditional covariance is:

$$\Sigma_{\text{miss} | \text{obs}} = \mathbf{K}_{\text{miss}\cdot\text{miss}} - \mathbf{K}_{\text{miss}\cdot\text{obs}}\mathbf{K}_{\text{obs}\cdot\text{obs}}^{-1}\mathbf{K}_{\text{obs}\cdot\text{miss}}$$

So,

$$\mathbf{f}_\text{miss} | \mathbf{f}_\text{obs} \sim \mathcal{N}\left(\mathbf{K}_{\text{miss}\cdot\text{obs}}\mathbf{K}_{\text{obs}\cdot \text{obs}}^{-1}\mathbf{f}_\text{obs}, \ \mathbf{K}_{\text{miss}\cdot\text{miss}} - \mathbf{K}_{\text{miss}\cdot\text{obs}}\mathbf{K}_{\text{obs}\cdot\text{obs}}^{-1}\mathbf{K}_{\text{obs}\cdot\text{miss}}\right)$$

### e)

#### (i)

When outputs are treated as independent, we are making the assumption that the outputs are not correlated and have no joint distribution. Mathematically, we are assuming that $\operatorname{Cov}(\mathbf{f}_p(\mathbf{x}), \mathbf{f}_q(\mathbf{x}'))=0, \ \forall p\ne q, \ \forall\mathbf{x}, \mathbf{x}'$.

#### (ii)

When tracking a patient's vital signs, a multi-output GP would far about perform multiple single-output GPs. Statistics like heart rate, blood pressure and body temperature all influence each other. A low heart rate might reduce body temperature, and a high body temperature might increase heart rate. All parts of the body influence each other, so using the MOGP framework would significantly outperform multiple single output GPs.

#### (iii)

The term borrowing strength refers to when one output uses the confidence and prediction of another related output to improve its own predictions. This is most useful when a sparsely recorded output is strongly correlated to another output that is recorded more often. When this happens, the sparsely populated output borrows the strength of the densely populated output's relationships with all other outputs.

---

## Question 3 — Kernel Methods for Regression

### Data Preprocessing

Exploratory Data Analysis was performed on the california_housing dataset. Specific coding details can be found in the 'Assignment_3.ipynb' file. Using pandas' `.describe()` function, it was found that there were 20,640 entries in the table, each corresponding to a different (latitude, longitude) value. Latitude ranged from -124.35 to -114.31 while longitude ranged from 32.54 to 41.95, and both were accurate to two decimal places. On average there were 499.539680 homes per coordinate, meaning there is data from 10,310,499 homes.

Of the 20,640 entries, only 5 had an ocean_proximity value of 'ISLAND', meaning there is not a lot of data relating to this. When taking the sample 2,000, it is important to include at least a few of these entries, or the model will have little basis on island prediction. ocean_proximity was also the only column that was not an integer or float. As there are just 5 distinct categories in ocean_proximity, one-hot encoding was applied.

An outlier test was run on all numeric features and the target. Of the 8 tested features, 4 had over 5% outliers. 5.2% of the target's values were also outliers.

The target, median_house_value, was grafted temporarily onto X to run `.corr()` and see how correlated each feature was with each other. The python package seaborn was used to apply a heatmap to visually see the differences.

![Correlation heatmap](heatmap.png)

Latitude and longitude are correlated heavily to each other and to some of the ocean_proximity features, which makes sense given that they describe the placement of the housing sector on the map. total_rooms, total_bedrooms, population and households are all highly correlated. housing_median_age is negatively correlated with all of these, as well as with living inland. Most ocean_proximity features seem negatively correlated with each other, except for island which has so few features that it has almost no correlation with anything.

median_house_value is quite positively correlated with median_income, slightly correlated with living an hour from the ocean, near the bay, or near the ocean. It is quite negatively correlated with living inland.

Only one feature, total_bedrooms, contained missing values. As this had 0.98 correlation to households, the median of the ratio between bedrooms and households was used to impute the missing values. There was a median of 1.048 bedrooms per household.

```python
ratio = (X['total_bedrooms'] / X['households']).median()
X['total_bedrooms'].fillna(X['households'] * ratio)
```

2,000 rows were sampled to reduce computational complexity (kernel methods require an $n\times n$ matrix). Two island entries were manually sampled to ensure they had some influence within our train-test split.

```python
from sklearn.model_selection import train_test_split
island_rows = X[X["ocean_proximity_ISLAND"]==1].sample(n=2, random_state=123)
others = X[X["ocean_proximity_ISLAND"]!=1].sample(n=1998, random_state=123)
X_sample = pd.concat([island_rows, others])
y_sample = y[X_sample.index]


X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=123
)
```

To scale the data, StandardScaler was used. In order to not scale the categorical variables that had been encoded, they were split, the numerical values were scaled, and then they were rejoined.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = [col for col in X.columns if 'ocean_proximity' not in col]
categorical_cols = [col for col in X.columns if 'ocean_proximity' in col]

X_train_numerical = X_train[numerical_cols].reset_index(drop=True)
X_train_categorical = X_train[categorical_cols].reset_index(drop=True)
X_test_numerical = X_test[numerical_cols].reset_index(drop=True)
X_test_categorical = X_test[categorical_cols].reset_index(drop=True)

X_train_numerical_scaled = scaler.fit_transform(X_train_numerical)
X_test_numerical_scaled = scaler.transform(X_test_numerical)

X_train_numerical_scaled = pd.DataFrame(X_train_numerical_scaled, columns=numerical_cols)
X_test_numerical_scaled  = pd.DataFrame(X_test_numerical_scaled, columns=numerical_cols)


X_train = pd.concat([X_train_numerical_scaled, X_train_categorical], axis=1)
X_test = pd.concat([X_test_numerical_scaled, X_test_categorical], axis=1)
```

### Implementing the Models

All kernel models use a loss function to attempt to learn from the data. This function is defined on a residual $r_i = y_i-\boldsymbol{\theta}^\top\boldsymbol{\phi}(\mathbf{x}_i)$, such that it grows as a function of $r$. We define it $\ell(r)$.

A regularisation term, by default $\ell_2$, with hyperparameter $\lambda$ is added to artificially induce a local maximum. This term will be tuned on a model-by-model and kernel-by-kernel basis to obtain optimal results.

For comparison's sake, all three models will try the same three kernels. The kernels chosen were:

**Radial Basis Function (RBF):** This was chosen because it is a strong default choice for nonlinear problem and many of the california_housing features are non-linear in relation to the target. RBF measures similarity based on Euclidean distance, with close points having strong influence on each other. This is ideal when examining how house placement and size affects cost.

**Polynomial Kernel:** Feature interactions within our dataset are very important, having significant influence over each other. With such cases, a polynomial kernel is ideal. Varying degree $d$ will help identify the degree of nonlinearity present. With only 9 features, I would expect $d$ to be relatively low.

**Laplacian Kernel:** As mentioned in the data preprocessing section, there are a significant number of outliers in the dataset. A Laplacian kernel is more robust to outliers, using $\ell_1$ regularisation. Other than this, Laplacian is similar to RBF and so carries over that kernel's strengths.

#### KRR

Kernel Ridge Regression uses loss function $\ell(r)=r^2$.

For RBF, GridSearchCV was used to find the optimal values for the gamma and alpha hyperparameters. Mean squared error was used to score in GridSearchCV. Gamma values of [0.001, 0.01, 0.05, 0.1, 1] and alpha values of [0.001, 0.01, 0.1] were tested. Optimal values were: `{'alpha': 0.001, 'gamma': 0.01}`.

The polynomial kernel has hyperparameters gamma, degree and coef0, along with the regularisation hyperparameter alpha. The same gamma and alpha values were tested as with RBF, along with [2, 3, 4, 5, 6] for degree and [0, 1] for coef0. Optimal values were: `{'alpha': 0.001, 'coef0': 1, 'degree': 5, 'gamma': 0.001}`.

The laplacian kernel, being similar to RBF, needed only the alpha and gamma hyperparameters optimised. Testing on the same values as above, optimal values of `{'alpha': 0.1, 'gamma': 0.1}` were found.

To evaluate, y_train and y_test were first scaled using standard scaler to prevent overly large MSE and MAE scores.

KRR is quite sensitive to changes in alpha and gamma. Plotting gamma and alpha sensitivities, we get the following.

| ![KRR RBF Sensitivity](krr_rbf_sensitivities.png) | ![KRR Laplacian Sensitivity](krr_lap_sensitivities.png) | ![KRR Polynomial Sensitivity](krr_polynomial.png) |
|:---:|:---:|:---:|
| (a) KRR RBF Sensitivity | (b) KRR Laplacian Sensitivity | (c) KRR Polynomial Sensitivity |

For the above hyperparameter sensitivity curves, the polynomial kernel had by far the highest variance when gamma increased. This is because the degree of the polynomial kernel is 5, which greatly enhances slight changes. All models saw a sharp spike in test MSE as gamma increased, but no obvious change in train MSE.

RBF and laplacian kernels had test MSE dip around alpha = 0.01 or 0.1, before increasing again. This was not reflected in the polynomial kernel, where it increased consistently.

For each kernel, the following MSE and MAE values were found:

![KRR scores](krr_scores.png)

The laplacian kernel performed the best in both test MSE and test MAE, with just 0.1575 MSE and 0.2882 MAE. It also trained the fastest. The polynomial kernel performed the worst in all categories, training the slowest by just 0.01s.

#### KLAD

KLAD will be trained by using sklearn's SVR model with epsilon = 0. Instead of alpha, sklearn's SVR uses C for its regularisation hyperparameter. From 1 d), $C = 1/(2n\lambda)$. $n=1600$, as our training set has $1600$ elements. KLAD uses loss function $\ell(r)=|r|$.

For RBF, the gamma and C hyperparameters were optimised. Using ranges `'gamma': [0.001, 0.01, 0.05, 0.1, 1]` and `'C': [0.001, 0.01, 0.1, 1, 10, 100]`, optimal values of `{'C': 10, 'gamma': 0.05}` were found.

For the polynomial kernel, I had to reduce the ranges of values tested in order to reduce training time. I had to reduce the gamma range to [0.01, 0.001, 0.05, 0.1], as anything near 1 would drastically slow down the training process. The C range was reduced to [0.1, 1, 10], and degree range to just [2, 3, 4, 5]. Optimal values of `{'C': 10, 'degree': 2, 'gamma': 0.1}` were found.

sklearn's SVR method does not support laplacian kernel, so `from sklearn.metrics.pairwise import laplacian_kernel` was used to manually create a laplacian kernel before setting SVR's kernel parameter to `'precomputed'`. Using the same gamma and C values as used for RBF, the optimal values were: `{'gamma': 0.1, 'C': 10}`.

Plotting C and gamma sensitivities gives the following.

| ![KLAD RBF Sensitivity](klad_rbf.png) | ![KLAD Laplacian Sensitivity](klad_lap.png) | ![KLAD Poly Sensitivity](klad_poly.png) |
|:---:|:---:|:---:|
| (a) KLAD RBF Sensitivity | (b) KLAD Laplacian Sensitivity | (c) KLAD Poly Sensitivity |

Both RBF and laplacian kernels followed the same gamma trajectory, with test MSEs starting high, minimising at about 0.1 and then increasing rapidly. Interestingly, a gamma of 0.1 outperformed gamma of 0.05 in the RBF kernel, which is not the optimal value obtained from GridSearchCV.

Test MSE was consistently lower than train MSE when adjusting C. All kernels showed the same trend here, starting at a very high MSE that continually decreased as C increased.

For each kernel, the following training time, MSE and MAE values were found.

![KLAD scores](klad_scores.png)

Again, the laplacian kernel performed the best with a 0.1643 MSE and 0.2931 MAE. Polynomial is again the worst performing, with a 0.2555 MSE and a 0.3709 MAE. The RBF kernel trained the fastest, followed by laplacian and then polynomial. The polynomial kernel took almost twice as long to train as the RBF.

#### SVR

SVR uses loss function $\max\{0, |r|-\varepsilon\}$. All hyperparameters tested were the same as in KLAD, with the addition of testing the 'epsilon' hyperparameter using the value range [0.001, 0.01, 0.1, 1].

For RBF, optimal values of `{'C': 10, 'epsilon': 0.1, 'gamma': 0.1}` were found.

For the polynomial kernel, optimal values of `{'C': 10, 'degree': 3, 'epsilon': 0.001, 'gamma': 0.1}` were found.

For the laplacian kernel, optimal values of `{'gamma': 0.1, 'C': 10, 'epsilon': 0.1}` were found.

Plotting C, gamma and epsilon sensitivities gives the following.

| ![SVR RBF Sensitivity](svr_rbf.png) | ![SVR Laplacian Sensitivity](svr_lap.png) | ![SVR Poly Sensitivity](svr_poly.png) |
|:---:|:---:|:---:|
| (a) SVR RBF Sensitivity | (b) SVR Laplacian Sensitivity | (c) SVR Poly Sensitivity |

All three kernels followed similar trends as epsilon increased, staying flat/slightly decreasing up to 0.1, before increasing quickly towards 1. The laplacian kernel's train MSE started far lower than either RBF's or the polynomial kernel's. Similar to the KLAD-RBF example above, epsilon=0.1 actually outperformed epsilon=0.001 for the polynomial, which is not what GridSearchCV output as optimal.

The trends in the gamma and C sensitivity curves were very similar to those in the KLAD model. For gamma, RBF and laplacian started high, minimised around 0.1, and then increased. For polynomial, MSE continually decreased as gamma increased. It should be noted that, as mentioned previously, although polynomial may perform better with an even higher gamma, doing so took too long to train and so higher gammas were not considered.

For each kernel, the following training time, MSE and MAE values were found. Epsilon values of 0.001 and 0.1 were both tested for the polynomial kernel, and it was found that 0.1 performed the best.

![SVR scores](svr_scores.png)

The SVR model did not break the trend seen in the other two kernels. Laplacian performed the best in both test MSE and test MAE, followed by RBF and then polynomial. Laplacian trained the fastest, and polynomial trained by far the slowest, at around 7x slower than the RBF kernel.

### Model Comparison and Discussion

Overall, the best performing model-kernel combination was KRR with laplacian kernel. In fact, the laplacian kernel was the best performing kernel in all three models. This may be because the laplacian kernel is the most robust to outliers, and the dataset had a very high outlier rate.

There was no obvious difference in robustness to outliers throughout the models. Theoretically, KRR, with loss function $\ell(r)=r^2$ is not robust to outliers at all, but it managed to perform the best in both test MSE and MAE over all kernels. This could be explained by the scaling done to both the features and the target, as well as by the fact that what outliers there were were not so large as to dominate. Most likely it was a problem with scaling, as standardising them likely reduced the weight of the outliers and boosted the performance of KRR. SVR and KLAD both performed very similarly to KRR, and are both theoretically much more robust, so a dataset with more significant outliers or less extreme scaling would have to be done to better see the effects of loss function on robustness.

The trade-off between predictive accuracy and computational cost was most evident when trying to obtain optimal polynomial kernel C values. As C increased, time to train increased exponentially, making it infeasible to test higher and potentially better performing values of C. The best performing of all the models was actually the fastest to train of all the models.

Using different kernels, a dataset with more outliers, or a different scaling system may yield different and more theoretically consistent results.

### References

- Pandas python package - https://pandas.pydata.org
- Scikit learn python package - https://scikit-learn.org/stable/
- Seaborn python package - https://seaborn.pydata.org
- Numpy python package - https://numpy.org