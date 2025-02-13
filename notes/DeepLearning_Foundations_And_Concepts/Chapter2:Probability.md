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

### Generalized Formulas 
- *sum rule*: $p(X)=\sum_{Y} p(X,Y)$
- *product rule*: $P(X,Y) = P(Y|X)*P(X) = P(Y,X)$ *---this formula leads to Baye's Theorem* 