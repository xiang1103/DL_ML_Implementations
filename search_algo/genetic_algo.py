''' 
## Evolutionary Algorithm 

It's a generation algorithm based on how evolution works. 

### Methods 
- The algorithm is like a tree search where we start with a pool of candidates and evaluate which ones we want to take to make offsprings through a **fitting function**.  
- This process is testing then selecting the fittest.   

- Then we take the **crossover function** to combine the attributes of the best candidates chosen to make offsprings. **mutation** is added to generate something new 
- This process generates new things that are different from the starting random and getting more optimized for the objective function   

- Evaluate the offspring again as the new generation and determine if we continue the evolution. 

- Generation part comes in at this iterative crossover, mutation 

Simple genetic algorithim to generate text based on GeekforGeeks
'''

import random # we will use random to help sampling random initial pool 
import sys 

population_size= 100    # per generation size 

genes = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''    # possible genes to create 
target= "I was here at one point" 

# represent each individual in the population 
class Individual (object): 
    def __init__(self, chromosome): 
        self.chromo = chromosome 
        self.fit_score= self.fit2() 

    @classmethod
    def mutate_genes(self):
        '''  
        mutate the gene of the selected individual to get something new 
        @return: a single gene 
        '''
        mutation= random.choice(genes)  # randomly mutate to a different gene 
        return mutation 

    @classmethod
    def create_random(self):
        '''  
        Create a random individual from the initial stage 
        @return: a list of random genes that has the same length as the target 
        '''
        length = len(target)
        return [self.mutate_genes() for _ in range(length)]    


    def mate(self, indiv_2): 
        '''' 
        crossover function to make an offspring 
        @param indiv_2: the other individual to make offspring 
        We will do this by probability of taking. 
        This is a stochastic process, and not incorporating any information 

        @return: a child Individual with genome 
        '''
        child_genome= [] 
        for p1, p2 in zip(self.chromo, indiv_2.chromo): 
            prob= random.random() 
            if (prob<0.45): # take parent 1 
                child_genome.append(p1)
            elif (prob<0.9): # take parent 2 
                child_genome.append(p2)
            else: # or mutate a random one 
                child_genome.append(self.mutate_genes())
        return Individual(child_genome)


    def fit(self): 
        '''  
        Calculate the fitness by comparing the difference between the input text and target text 
        This is smilar with a loss function 
        '''
        fit=0 
        for p1, gt in zip(self.chromo, target): 
            fit += abs(ord(gt) - ord(p1))
        return fit 


    def fit2(self): 
        '''' 
        A second way of just comparing how many characters are different 
        '''
        fit= 0 
        for p1, gt in zip(self.chromo, target):
            if (p1!=gt): fit+=1 
        return fit 


def main(): 
    generation =1 # current generation 
    population = [] # all the children in the population 
    for _ in range(population_size): # generate first population 
        genome= Individual.create_random() 
        population.append(Individual(genome))

    # start generation process 
    found= False 
    prev_score=sys.maxsize    # the prev generation score so we can check if there are any improvements and prevent infinite loop

    while (not found): 
        # sort the population by their fitness score 
        population = sorted(population,key = lambda x:x.fit_score)
        
        # check if we have found the target 
        if (population[0].fit_score==0): 
            found =True
            break 

        # start to create the next generation 
        next_generation= [] 

        # only take the top 10% to be chosen for mating and pass into the next generation
        s = int (0.1*population_size)
        next_generation.extend(population[:s]) 

        # make offsprings from crossover functions to generate 90% of the population 
        # we will choose from top 20% of the population 
        s2 = int (0.9*population_size) 
        s3 = int (0.2*population_size)
        for _ in range(s2): 
            parent1= random.choice(population[:s3])
            parent2= random.choice (population[:s3])
            child = parent1.mate(parent2)
            next_generation.append(child)
        population= next_generation 

        print(f"Generation: {generation} | Best String:{"".join(population[0].chromo)} | Fitness: {population[0].fit_score} ")
        generation+=1 
        # generated worse 
        if (prev_score < population[0].fit_score): 
            print(f"Current generation is worse than previous generation. Terminating.")
            found= True  

        # set the prev score  
        prev_score = population[0].fit_score 

    # final printing 
    print(f"Generation: {generation} | Final String:{"".join(population[0].chromo)} | Fitness: {population[0].fit_score} ")

if __name__ == "__main__": 
    main() 
