---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell}
:tags: [hide-input]
import numpy as np
from ipywidgets import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skewnorm
import matplotlib as mpl
import pandas as pd
```
# Knowledge Modeling and Likelihood Tests

## Snail Example

We start with an example with a dataset called abalone, which contains information about snails. 
We want to predict the sex of a snail from their number of rings.

We can predict a snail as either a male or a female. This leads to the following possible mistakes:

- Mistake 1: predicting a snail's sex as female when it is a male
- Mistake 2: predicting a snail's sex as male when it is a female

To get the predictions right, we have to know how female and male snails differ from each other. 
Do the males have more rings (visible on their cone), do the females have a different color? 
Answers to this questions help us to classify a snail's sex easier. 
The more knowledge we have about the population of snails, the better we can design our classifier. 

### Initialize Dataset

First we have to load the dataset and store the the number of males with their corresponding number of rings into a new variable.
The same goes for female:

```{code-cell}
:tags: [hide-input]
snails = pd.read_csv('D:/Dokumente/Python Scripts/abalone.csv')
snails.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                  'Shucked weight', 'Viscera weight', 'Shell weight',
                  'Rings']
sex = np.array(snails['Sex'])
number_of_rings = np.array(snails['Rings'])
infant_idx = sex == 'I'
sex = sex[~infant_idx]
number_of_rings = number_of_rings[~infant_idx]
sex[sex == 'M'], sex[sex == 'F'] = -1, 1
sex = np.int8(sex)

males = np.zeros((30))
females = np.zeros((30))
for i in range(30):
    males[i] = sum(sex[number_of_rings == i] == -1)
    females[i] = sum(sex[number_of_rings == i] == 1)

males[3] -= 2
males[4] -= 3
males[5] -= 5
males[6] -= 5
males[7] -= 15
males[8] -= 45
males[9] -= 36
males[10] -= 40
males[11] -= 15
females[6] += 5
females[7] += 10
females[13] += 10
females[14] += 5
females[15] += 12
females[16] += 10
females[26] += 2
print("sum males ", sum(males))
print("sum females ", sum(females))
```

We can now print a plot with the males and females and their number of rings:

```{code-cell}
:tags: [hide-input]
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.bar(np.arange(30) - 0.2, males, width=0.4)
ax.bar(np.arange(30) + 0.2, females, width=0.4)

ax.legend(['Male', 'Female'], frameon=False)
ax.set_xlabel('Number of rings')
ax.set_ylabel('Number of snails');
```

### What is the "best" classifier?

The best classifier or the best predictor, is the one that makes the fewest mistakes. 
This is also called the minimum error rule. Remember the two mistakes we can make in the snail prediction:

```{code-cell}
:tags: [hide-input]
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(-np.array([-1, 1])[np.int8(males > females)[3:]]);
```


### A good but impractical rule

We need a good rule to find the best predictor. To do this, we have to measure the entire population of snails. 
With the measurement, we get additional knowledge about the population / problem. This is what makes prediction possible. 
### Modeling Knowledge

Knowledge about the population makes predictions possible in the first place. 
The more knowledge we have, the more accurate our classifier gets. To work with this knowledge, 
we have to represent them in a suitable way. In Machine Learning we model knowledge as probability
distributions. In the following we have some patterns $X$ and some labels $Y$. The label is in this case 
the sex of the snail (male or female). The pattern has to do with the number of rings, since we want to predict
the label with the number of rings (pattern). 
- we use training and test data which are independent samples from a joint distribution
- in other words: there is some joint distribution over patterns and labels $p_{x,y}(x,y)$ or $p(x,y)$
- for the moment, we assume that $p$ is known
To do predictions, we need a so-called predictor. This is a rule, a formula or an algorithm. 
  

```{code-cell}
:tags: [hide-input]
l1, s1, a1 = 7.40, 4.48, 3.12
l2, s2, a2 = 7.63, 4.67, 4.34

x = np.linspace(skewnorm.ppf(0.001, a1, loc=l1, scale=s1),
                skewnorm.ppf(0.999, a1, loc=l1, scale=s1), 
                100)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
line1, = axs[0].plot(x, skewnorm.pdf(x, a1, loc=l1, scale=s1),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = axs[0].plot(x, skewnorm.pdf(x, a2, loc=l2, scale=s2),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = axs[0].text(15, 0.12, '0.000')

axs[0].set_xlabel('Number of rings')
axs[0].set_ylabel('Probability density')
axs[0].set_ylim(0, 0.154)

thr0 = 15
thrline, = axs[0].plot([thr0, thr0], [0, 0.20])

def update(thr=thr0):
    err1 = skewnorm.cdf(thr, a2, loc=l2, scale=s2)
    err2 = 1 - skewnorm.cdf(thr, a1, loc=l1, scale=s1)
    
    p_error = (err1 + err2) / 2
    
    thrline.set_xdata([thr, thr])
    axs[1].plot(err1, 1 - err2, 'b.')
    text.set_text('$\mathbb{P}_{\mathrm{err}}$ = %0.3f' % (p_error,))
    fig.canvas.draw_idle()
```

This shows the probability densitiy function (pdf) for the males and females regarding the number of rings.

```{code-cell}
:tags: [hide-input]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
line1, = axs[0].plot(x, skewnorm.pdf(x, a1, loc=l1, scale=s1),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = axs[0].plot(x, skewnorm.pdf(x, a2, loc=l2, scale=s2),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = axs[0].text(15, 0.12, '0.000')

axs[0].set_xlabel('Number of rings')
axs[0].set_ylabel('Probability density')
axs[0].set_ylim(0, 0.154)

thrline, = axs[0].plot([thr0, thr0], [0, 0.20])
interact(update, thr=(5.0, 22.5, 0.1));
```

## Prediction
In binary classification (0, 1) we can formalize the minimum error rule / the best predictor using the following definitions:
We can use this again for the snail example. Instead of using male and female for the sex, we are using 0 and 1. Therefore 
Y = 1, is predicting the snail's sex as female and Y=0 is predicting the snails sex as male. 
Y is the label or in our case the sex of a snail, which we want to predict. 

Assume that $Y$ has priori probabilities: 
- $p_0 = \mathbb{P}[Y=0]
- p_1=\mathbb{P}[Y=1]$

Regarding snails: $p_0$ is the probability that a snail's sex is male and $p_1$ is the probability that a snail's sex is female.
Since we have the same count of males and females (see above), the probability of a snail being a male or female
is $\frac{1}{2}$. This means that the classes are balanced. 

### Prediction (continued)

We use here $p$ for conditional probability, $p_0$ and $p_1$ as prior probabilities and $\mathbb{P}$ as probability itself.

$p_0$ and $p_1$ are proportions of two classes in the population. If we draw a large number of $n$ 
of samples from $p$ there will be approximately $p_0n$ labels $0$ and $p_1n$ labels $1$.
Remember snail example where we had $p_0 = p_1 = \frac{1}{2}$. 
The patterns or groups are modeled by a random vector $X$. The distribution of $X$ depends on $Y$. 
Since we have binary classification $Y$ can be either zero or one. This connection between $X$ and $Y$ is 
called joint distribution. The conditional probabilities (probability of $x$ given $Y$) are 

- $p(x \mid Y = 0)$
- $p(x \mid Y = 1)$

If we have $p(x \mid Y = y)$ we have a special case called generative models or likelihood functions. In this model 
we have the joint probability $p(x,y) = p(x \mid Y=y)p(Y=y)$. 

 


### Prediction via optimization

With the help of these definitions above, the optimal predictor can finally be calculated. Since we want the optimal predictor, 
we can use optimization to get the correct result, in other words optimization over algorithms. Other 
possible methods would be prediction via networks (graph attention network, graph neural network) or compressed sensing. 
We are going to focus on prediction via optimization

We already defined the set of algorithms as $A = \left\{ f(x) = \begin{cases} 0 & \text{if}~ x \leq \eta \\ 1 & 
\text{if}~ x > \eta \end{cases} \ \bigg| \ \eta \in \mathbb{R} \right \}
$ with the optimization problem to find $f \in A$ and $f$ has to minimize $\mathbb{P}$(mistake), 
the one with the least errors. Let's take a look at $\mathbb{P}$(mistake): 

- To get the probability of false classified patterns, we have to think about all possibilities: 
- $\mathbb{P}(f(X)=0 \mid Y=1)$: predict zero but actual is 1, calling for now $\mathbb{P}(0\_but\_1)$
- $\mathbb{P}(f(X)=1 \mid Y=0)$: predict one but actual is 0, calling for now $\mathbb{P}(1\_but\_0)$
- other cases are not interesting, since they are either predict zero and actual is zero or predict one and actual is one
Using our knowledge from statistics/probability we have $\mathbb{P}$(mistake)= $\mathbb{P}(0\_but\_1)\cdot 
  \mathbb{P}(Y=1) + \mathbb{P}(1\_but\_0)\cdot \mathbb{P}(Y=0)$
  
The result on the calculation of our optimization is called an estimate or a prediction. Written $\hat{Y} \equiv f(X)$


#### Risk

If we have calculated $\hat{Y}$ we are interested into how good is our prediction. First we introduce a term called Loss. 
Since our prediction will make mistakes, we want to make the best out of it, so we are choosing a price which we are paying
for different kind of mistakes. This is called loss or loss-function. An easy example of a loss-function could be:
$ loss(\hat{y}, y) = \begin{cases} 0, \hat{y} = y \\ 1, \hat{y} \neq y \end{cases}$
More interseting is the so called Risk. The risk defines the expectations how often our model do a mistake. Remember, 
the mistake is connected to the before defined loss-function. Therefore we can define the risk in case of our prediction/estimation
as :
- $R[\hat{Y}]:=\mathbb{E}_{(X,Y)\sim \mathbb{P}}[loss(\hat{Y}(X),Y)]$

Obviously we want as fewer mistakes as possibles. Therefore, we want the smallest possible risk/risk-function. Instead of 
looking for the biggest as in a maximization problem, we have here a minimization problem. We want to minimize our risk. 
To minimize the risk, we have to use the prediction rule which leads to the smallest risk:
- $\hat{Y} = f_{best}(X)$ where $f_{best} = arg\min\limits_{f \in A}R[f(X)]$. 

This leads to the optimal predictor/estimate $\hat{Y}(x) = \mathbb{1}\{\mathbb{P}[Y=1\mid X=x] \geq factor \cdot 
\mathbb{P}[Y=0 \mid X=0]\}$. $factor$ is a function regarding the different possible outcomes/losses:
- $factor = \frac{loss(1,0) - loss(0,0)}{loss(0,1)-loss(1,1)}$

## Likelihood Tests

For likelihood tests, we have to know about the posterior probability as well as probabilities called likelihoods:
- $\mathbb{P}[Y=y \mid X=x]$ are posterior probability
- $p(x \mid Y = y)$ are likelihoods

They are both used in the Bayes Theorem:

- The probability of an event $A$ occurring given that $B$ is true is equals to the probability of event $B$ occurring 
given that $A$ is true multiplied with the probability of $A$ divided by the probability of $B$
  
- As mathematical statement: $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$

Using the Bayes Theorem, posterior probability and the likelihood we get:

- $\mathbb{P}[Y=y \mid X=x] = \frac{p(x\mid Y=y)p_y}{p(x)}$, where $p(x)$ is the density of the marginal distribution of $X$.

Remember our optimal predictor: $\hat{Y}(x) = \mathbb{1}\{\mathbb{P}[Y=1\mid X=x] \geq factor \cdot \mathbb{P}[Y=0\mid X=x]\}$
With the help of Bayes and likelihood our optimal predictor becomes:
- $\hat{Y}(x) = \mathbb{1}\{\frac{p(x\mid Y=1)}{p(x\mid Y=0)} \geq \frac{p_o(loss(1,0)-loss(0,0))}{p_1(loss(0,1)-loss(1,1))}\}$

Using this predictor is called a likelihood ratio test. Generally a likelihood ratio test is a predictor of the form 
$\hat{Y}(x) = \mathbb{1}\{\mathcal{L}(x) \geq \mathcal{n}\}$, where $\mathcal{L}(x) := \frac{p(x \mid Y=1)}{p(x\mid Y=0)}$.
In our likelihood ratio test example we have a specific term for $\mathcal{n}$ which is also called threshold. The threshold
has to be bigger than zero. 

### Signal and Noise Example

Using the knowledge of likelihood ratio tests we can do an example regarding signal and noise. We are still in a linear system,
 where $Y$ can be zero or one. If $Y=0$ we observe $w$, where $w \sim \mathcal{N}(0,1)$ and if $Y=1$ we observe $w + s$ 
for a deterministic scalar $s$. 
- The pdf (probability density function) of a standard normal distribution ($\sim N(0,1)$) is $\phi(z) = \frac{1}{\sqrt{2\pi}}
e^{-\frac{z^2}{2}}$.
  
- In our case this leads to following pdf's: 
- $p(x\mid Y=0)=\mathcal{N}(0,1)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$ and
- $p(x \mid Y=1) = \mathcal{N}(s,1)=\frac{1}{\sqrt{2\pi}}e^{-\frac{(x-s)^2}{2}}$

$s$ is called shift and it determines how hard it is to predict $Y$
  
## Example without Likelihood ratio tests


```{code-cell}
:tags: [hide-input]


l1, s1, a1 = 7.40, 4.48, 3.12
l2, s2, a2 = 7.63, 4.67, 4.34

x = np.linspace(skewnorm.ppf(0.001, a1, loc=l1, scale=s1),
                skewnorm.ppf(0.999, a1, loc=l1, scale=s1), 
                100)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
line1, = ax.plot(x, skewnorm.pdf(x, a1, loc=l1, scale=s1),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = ax.plot(x, skewnorm.pdf(x, a2, loc=l2, scale=s2),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = ax.text(15, 0.12, '0.000')

ax.set_xlabel('Number of rings')
ax.set_ylabel('Probability density')
ax.set_ylim(0, 0.154)

thr0 = 15
thrline, = ax.plot([thr0, thr0], [0, 0.20])

def update(thr=thr0):
    err1 = skewnorm.cdf(thr, a2, loc=l2, scale=s2)
    err2 = 1 - skewnorm.cdf(thr, a1, loc=l1, scale=s1)
    
    p_error = (err1 + err2) / 2
    
    thrline.set_xdata([thr, thr])
    text.set_text('$\mathbb{P}_{\mathrm{err}}$ = %0.3f' % (p_error,))
    fig.canvas.draw_idle()
    
interact(update, thr=(5.0, 22.5, 0.1));
```


## Example with likelihood ratio tests

```{code-cell}
:tags: [hide-input]

sigma = 1
loc1 = 0.0
loc2 = 2

x_min = norm.ppf(0.001, loc=min(loc1, loc2), scale=sigma)
x_max = norm.ppf(0.999, loc=max(loc1, loc2), scale=sigma)
x = np.linspace(x_min, x_max, 200)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
line1, = ax.plot(x, norm.pdf(x, loc=loc1, scale=sigma),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = ax.plot(x, norm.pdf(x, loc=loc2, scale=sigma),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = ax.text(2, 0.12, '$\mathbb{P}_{\mathrm{err}}$ = %0.3f' % (0,))

ax.set_xlabel('Number of rings')
ax.set_ylabel('Probability density')
y_max = 1.1 / sigma / np.sqrt(2 * np.pi)
ax.set_ylim(0, y_max)

thr0 = 0
thrline, = ax.plot([thr0, thr0], [0, y_max], 'k')

def update(thr=thr0):
    err2 = 1 - norm.cdf(thr, loc=loc1, scale=sigma)
    err1 = norm.cdf(thr, loc=loc2, scale=sigma)
    
    p_error = (err1 + err2) / 2
    
    thrline.set_xdata([thr, thr])
    text.set_text('$\mathbb{P}_{\mathrm{err}}$ = %0.3f' % (p_error,))
    fig.canvas.draw_idle()
    
interact(update, thr=(x_min, x_max, (x_max - x_min) / 200));
```

## Gaussian example

Instead of normal distribution, let's now work with Gaussian distribution. For this example we define a prior probability
$p_1 = \mathbb{P}(Y=1)$ which is very small, e.g. $p_1 = 10^{-6}$. There is no cost/loss if we declare $\hat{Y} = 0$. 
On the other side, if we declare $\hat{Y} = 1$ we have a cost of 100 if $Y$ is actual 0 and we gain a reward if we predict correct. 

| loss | $\hat{Y}$ = 0 | $\hat{Y}$ = 1|
|-----|--------:|--------:|
| $Y$ = 0 | 0   |  100  |
| $Y$ = 1 | 0 | —1'000'000  |

We can define the optimal threshold value $\mathcal{n}$ as $log\mathcal{n} = log(\frac{p_0(loss(1,0)-loss(0,0))}{p_1(loss(0,1)-loss(1,1))})
\approx 4.61$. 

To receive the optimal predictor we use the calculation $logp(x\mid Y=1) = logp(x\mid Y=0) = -\frac{1}{2}(x-s)^2 + \frac{1}{2}
x^2 = sx-\frac{1}{2}s^2$. This leads to the following predictor:
- $\hat{Y} = \mathbb{1}\{sX > \frac{1}{2}s^2+log(\mathcal{n})\}$ 

## Types of Errors and successes

We already mentioned that there are different errors which we can do with a classifier. In the snail example we could classifiy
a snail as male if it was actual a female. We now define these types of errors more formally:

|                   | $Y = 0$        | $Y = 1$        |
|-------------------|----------------|----------------|
|   $\hat{Y} = 0$    | true negative  | false negative |
|   $\hat{Y} = 1$    | false positive | true positive  |$

- True positive rate / $TPR = \mathbb{P}[\hat{Y}(X) = 1 \mid Y = 1]$
- False positive rate / $FPR = \mathbb{P}[Y(X) = 1 \mid Y = 0]$
- False negative rate / $FNR = 1 - TPR$
- True negative rate / $TNR = 1 - FPR$


```{code-cell}
:tags: [hide-input]
sig1 = 1
sig2 = 3
loc1 = 0.0
loc2 = 4
p1 = 0.3
p2 = 1 - p1

x_min = norm.ppf(0.001, loc=min(loc1, loc2), scale=max(sig1, sig2))
x_max = norm.ppf(0.999, loc=max(loc1, loc2), scale=max(sig1, sig2))                
x = np.linspace(x_min, x_max, 200)

fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
line1, = ax.plot(x, norm.pdf(x, loc=loc1, scale=sig1),
       'b-', lw=4, alpha=0.6)
line2, = ax.plot(x, norm.pdf(x, loc=loc2, scale=sig2),
       'r-', lw=4, alpha=0.6)
line3, = ax.plot(x, p1*norm.pdf(x, loc=loc1, scale=sig1) + p2*norm.pdf(x, loc=loc2, scale=sig2),
       'k-', lw=1, alpha=0.6)

text_str = '$\mathbb{P}_{\mathrm{err}}$ = %0.3f\n' \
         + 'TPR = %0.3f\n' \
         + 'FNR = %0.3f\n' \
         + 'FPR = %0.3f\n' \
         + 'TNR = %0.3f\n'

text = ax.text(2, 0.25, text_str % (0,0,0,0,0))

ax.set_xlabel('Number of rings')
ax.set_ylabel('Probability density')
ax.legend(['Class 0', 'Class 1'], frameon=False)
y_max = 1.1 / min(sig1, sig2) / np.sqrt(2 * np.pi)
ax.set_ylim(0, y_max)

thr0 = 0
thrline, = ax.plot([thr0, thr0], [0, y_max], 'k')

def update(thr=thr0):

    # FPR = P(\hat{Y} = 1 | Y = 0)
    #     = P(BLUE > thr)
    FPR = 1 - norm.cdf(thr, loc=loc1, scale=sig1)
    # FPR = P(\hat{Y} = 0 | Y = 1)
    #     = P(RED <= thr)
    FNR = norm.cdf(thr, loc=loc2, scale=sig2)
    
    # P(mistake) = P(\hat{Y} = 1 | Y = 0) * p_0 + P(\hat{Y} = 0 | Y = 1) * p_1
    #            = p_0 * FPR + p_1 * FNR
    p_mistake = p1*FPR + p2*FNR
    
    thrline.set_xdata([thr, thr])
    text.set_text(text_str % (p_mistake, 1 - FNR, FNR, FPR, 1 - FPR))
    fig.canvas.draw_idle()

interact(update, thr=(x_min, x_max, (x_max - x_min) / 300));
```