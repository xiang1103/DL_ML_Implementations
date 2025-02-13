### Why can't models achieve perfect accuracy? 
-  Systematic Uncertainty: Because we have a finite set of data, we have finite amount of knowledge, so the model will not achieve the perfect accuracy 
-  Stochastic Uncertainty: Even with inifinite amount of data available, models still can't achieve perfect accuracy because each data set contains only *partial information* about the task. We'll need different kinds of task. This is also known as noise 
-  **Therefore**, to achieve high certainty, we need different kinds of data (like biopsy example + images data) and a lot of these data. 
### Basic Probability 
-  **Frequentist view**: probability in terms of frequency of repeatable events 
-  **Bayesion Probability**: Quantifying uncertainty through a lot of observations/ data
- **Random Variables**: Variable denoting the occurane of some events 
-  **joint probability**: The probability of random variables equaling to certain events 

### Probability Rules: 
-  **Sum Rules**: A rule to calculate the probability of a single random variable. Think of a 2D table. For example, 
$p(X=x_i)= \sum_{j=1}^M {p(X=x_i,Y=y_j)}$ in this case, we compute all the occurance of $y_j$
 and the total probability is the probability of both random variables occuring 
-  **Product Rule**: Calculating joint probabilities by: 
    $$p(X=x_i,Y=y_j)= \frac{n_{ij}}{c_i} *\frac{c_i}{N}=p(Y=y_j|X=x_i)*p(X=x_i)$$
-  **Marginal Probability**: Probability of a single random variable 
- **Prior Probability**: Probability before we observed the result of some test (before we got data)
- **Posterior Probability**: Probability (usually conditional) after we got the data 

### Generalized Formulas 
- *sum rule*: $p(X)=\sum_{Y} p(X,Y)$
- *product rule*: $P(X,Y) = P(Y|X)*P(X) = P(Y,X)$ *---this formula leads to Bayes' Theorem* 

### Bayes' Theorem 
- $P(Y|X)= \frac{P(X|Y) * P(Y)}{P(X)}$
- The denomenator can be seen as the normalization constant to make sure the sum of P(Y|X) =1 
- Bayes' Rule derivation of sum rule: $P(X)=P(X|Y) * P(Y)$

### Conditional and Prior probability 
- When we are calculating conditional probability, especially posterior, after we have gotten the data/probability, Bayes' Theorem considers the prior probability (i.e. $p(Y)$), therefore, **if the prior is low, the final conditional probability will be lower than expected** 


### Probability Density 
-  The probability of a variable x falling into a certain interval 
- Because the variables are now in a continuous plane, the sum rules, product rules change from summation to integral 
- Sum Rule: $p(x)= \int p(x,y)dy$

### Expectations 
- The weighted average of a function $f(x)$ under probability distribution p(x) is the **expectation** of f(x) 
- $\mathbb{E}[f]=\sum_{x}p(x)f(x)$
- The average is weighted by the probability of x occuring 
- In continuous distributions (x is continuous), the formula is written as $\mathbb{E}[f]= \int p(x)f(x)dx$