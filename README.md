# age_distortion

age_dist_policy_iter_general.py is the Python code pertaining to the simulations in 'Optimal Policies for Age and Distortion in a Discrete-Time Model' presented in the 2021 Information Theory Workshop (ITW). It outputs the optimal values for each eta value (see the manuscript) in a .csv file.

## Usage

The main function takes the following arguments:

   filename: Name of the output .csv file containing the optimal costs
   
   eta_max: the initial (and maximum) eta value to start the algorithm 
   
   eta_min: the final (and minimum) eta value to compute the average cost
   
   eta_num: number of eta values that the algorithm is run
   
   imp_num: support size of the importance distribution, i.e. |V|.
   
   imp_dist: the importance distribution must be given here. E.g., for V = {1,20} and P(V = 20) = 0.3, one must enter 1 0.7 20 0.3. The number of arguments must be 2*|V|
   
   sp_dist: the speaking distribution. There are 4 possible choices.
   
   1 - geom: geometric distribution. Following geom, one must enter its parameter p. E.g., geom 0.2
        
   2 - poisson: Poisson distribution. Following poisson, one must enter its parameter lambda. E.g., poisson 3
        
   3 - binomial: Binomial distribution. Following binomial, one must enter n and p. E.g., binomial 5 0.2
       
   4 - bernoulli: Bernoulli distribution. Following bernoulli, one must enter the two support values and the success probability p. E.g., bernoulli 3 10 0.3 yields P(Z = 10) = 0.3 and P(Z = 3) = 0.7.
        
Some examples:

```bash
python age_dist_policy_iter_general.py out1 4.5 0.45 30 2 1 0.8 10 0.2 geom 0.5
```

```bash
python age_dist_policy_iter_general.py out2 4.5 0.45 30 2 1 0.8 10 0.2 poisson 1
```

```bash
python age_dist_policy_iter_general.py out3 4.5 0.45 30 2 1 0.8 10 0.2 binomial 4 0.25
```

```bash
python age_dist_policy_iter_general.py out4 4.5 0.3461538462 30 2 1 0.8 10 0.2 bernoulli 1 2 1.0
```

## References

Y. İnan, R. Inovan and E. Telatar, "Optimal Policies for Age and Distortion in a Discrete-Time Model," 2021 IEEE Information Theory Workshop (ITW), 2021, pp. 1-6, doi: 10.1109/ITW48936.2021.9611456.

