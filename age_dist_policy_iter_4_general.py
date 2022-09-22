import numpy as np
import sys
import copy
import csv
import math

# from multiprocessing import Pool, cpu_count


    


def find_idx(state, imp_num):
    """
    Given the string b, returns its index in the tree
    """ 

    l = len(state)
    return int(sum([int(state[i])*(imp_num)**(i) for i in range(0,l)])+(imp_num**l-1)/(imp_num-1))

def find_optimal_policy(imp_dist, imp_num, sp_dist, sp_param, eta = 0.5, K_init = 0, str_tree = [], h_tree = [], temp_h_tree = [], prob_tree = [], pol_tree=[], parent = [], cost2parent = [], avg_cost = 0, st_list = {}, st_list_rev = [], K_cost = []):

    #gives E[V].

    imp_mean = sum([imp_dist[0][i]*imp_dist[1][i] for i in range(0,imp_num)])

    #Depending on the specified distribution, creates the probability vector, and calculates E[Z], K, and E[max(Z-K,0)]

    if sp_dist == 'geom':
        p = sp_param
        q = 1-p
        speak_mean = 1/p

        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)

        prob = [p*(q**float(i)) for i in range(0,K)]
        prob[-1] = prob[-1]/p
    
        speak_mean_trun = (q**K)/p

    elif sp_dist == 'poisson':
        speak_mean = 1 + sp_param
        
        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)
        prob = [np.exp(-sp_param)*(sp_param**i)/math.factorial(i) for i in range(0,K)]
        prob[-1] = 1-sum(prob[0:-1])

        speak_mean_trun = sum([np.exp(-sp_param)*(i-K)*(sp_param**(i-1))/math.factorial(i-1) for i in range(K+1,100)])

    elif sp_dist == 'uniform':
        speak_mean = (1 + sp_param)/2
        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)
        prob = [1/sp_param if i < sp_param else 0 for i in range(0,K)]
        prob[-1] = 1-sum(prob[0:-1])

        speak_mean_trun = sum([(i-K)/sp_param if i <= sp_param else 0 for i in range(K+1,100)])

    elif sp_dist == 'zipf':
        
        normalization = sum([i**(-sp_param) for i in range(1,1000)])

        speak_mean = sum([i**(-sp_param+1)/normalization for i in range(1,1000)])

        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)

        prob = [(i+1)**(-sp_param)/normalization for i in range(0,K)]
        prob[-1] = 1-sum(prob[0:-1])

        speak_mean_trun = sum([i**(-sp_param)*(i-K) for i in range(K+1,100)])

    elif sp_dist == 'binomial':

        sp_param_n = int(sp_param[0])
        p = sp_param[1]
        q = 1-p
        speak_mean = p*sp_param_n + 1

        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)

        prob = [math.comb(sp_param_n,i)*(p**i)*(q**(sp_param_n-i)) if i <= sp_param_n else 0 for i in range(0,K)]
        prob[-1] = 1-sum(prob[0:-1])

        speak_mean_trun = sum([math.comb(sp_param_n,i-1)*(p**(i-1))*(q**(sp_param_n-i+1))*(i-K) if i <= sp_param_n + 1 else 0 for i in range(K+1,100)])

    elif sp_dist == 'bernoulli':

        sp_param_l = int(sp_param[0])
        sp_param_h = int(sp_param[1])

        p = sp_param[2]
        q = 1-p

        speak_mean = p*sp_param_h + q*sp_param_l

        K = min(int(math.ceil((max(imp_dist[0]) - min(imp_dist[0]))/eta/speak_mean)),20)

        prob = [0]*K
        prob[sp_param_l-1] = q

        if K >= sp_param_h:
            prob[sp_param_h-1] = p
            speak_mean_trun = 0
        else:
            prob[-1] = 1-sum(prob[0:-1])
            speak_mean_trun = max(sp_param_l-K,0)*q + max(sp_param_h-K,0)*p




    speak_dist = prob
    print('speak_mean: ', speak_mean)
    print('speak_mean_trun: ', speak_mean_trun)
    print('K:', K)

    #Initializes the tree if K_init == 0, else appends new states if necessary.
    

    len_init = int((imp_num**(K_init+1)-1)/(imp_num-1))
    len_final = int((imp_num**(K+1)-1)/(imp_num-1))

    #########################################################################
    # str_tree: stores the string representations
    # h_tree: stores relative values
    # temp_h_tree: stores the variable temp
    # prob_tree: stores the probabilities of each string
    # pol_tree: stores policies 
    # parent: stores the parent_B1's
    # cost2parent: stores the cost to reach parent_B1's
    # avg_cost: average cost
    # st_list: the dictionary of states in B1
    # st_list_rev: index list of the dictionary st_list
    # K_cost: the variable added for the improved policy iteration
    #########################################################################


    if K_init == 0:
        str_tree = ['']
        h_tree = [0]
        temp_h_tree = [0]
        prob_tree = [1]
        pol_tree = [0]
        parent = [0]
        cost2parent = [0]
        avg_cost = imp_mean*(speak_mean-1)/speak_mean
        st_list = {}
        st_list[1] = 0
        st_list_rev = [1]
        K_cost = [0]
        st_len = 1


    for i in range(len_init,len_final):
        str_tree.append(str((i-1)%imp_num) + str_tree[(i-1)//imp_num])
        prob_tree.append(prob_tree[(i-1)//imp_num]*imp_dist[1][(i-1)%imp_num])
        temp_h_tree.append(0)
        K_cost.append(0)

        if i <= imp_num:
            cost2parent.append(0)
            parent.append(i)

        else:
            cost2parent.append(imp_dist[0][int(str_tree[i][0])]/speak_mean + cost2parent[(i-1)//imp_num])
            parent.append(parent[(i-1)//imp_num])

        h_tree.append(sum([imp_dist[0][int(elm)]/speak_mean for elm in str_tree[i]]))
        h_tree[-1] -= imp_dist[0][int(str_tree[-1][-1])]/speak_mean
        pol_tree.append(len(str_tree[i])-1)

    print('init finished')


    #Below, policy evaluation and update stages are performed repeatedly until convergence.


    for trials in range(0,1000):

        #########################################################################
        # POLICY EVALUATION
        #########################################################################

        #costs: vector of one step costs
        #P: (identity matrix) - (transition matrix of the induced Markov Chain)

        costs = np.zeros([len(st_list),1])
        P = np.eye(len(st_list))

        for i in range(0,len(st_list)):
  
            costs[i] += eta*(len(str_tree[st_list_rev[i]])-1)
            P[i,0] = 1

            #k runs over the possible values of the next interspeaking time Z, truncated with K

            for k in range(1,K+1):

                pp = speak_dist[k-1]

                for k1 in range((imp_num**k-1)//(imp_num-1),(imp_num**(k+1)-1)//(imp_num-1)):
                    pt = prob_tree[k1]
                    
                    new_str = str_tree[st_list_rev[i]][1:]+str_tree[k1]
                    new_len = len(new_str)

                    #If the next state has length greater than K, the skipped elements contribute to the distortion cost

                    if new_len > K:
                        cost_acc = pp*pt*sum([imp_dist[0][int(elm)] for elm in new_str[0:new_len-K]])/speak_mean
                        costs[i] += cost_acc
                        new_str = new_str[new_len-K:]

                    next_idx = int(find_idx(new_str,imp_num))
                    
                    costs[i] += pp*pt*cost2parent[next_idx]

                    if(parent[next_idx] > imp_num):
                        p_idx = st_list[parent[next_idx]]
                        P[i, p_idx] -= pp*pt
                        

            # If Z exceeds K, the expected number of skipped elements contribute to the distortion cost

            costs[i] += speak_mean_trun/speak_mean*imp_mean


        #obtain h for states in B_1 by solving the linear system Ph = c

        h = np.linalg.solve(P, costs)

        avg_cost = float(h[0])

        print('lambda: ', avg_cost)

        #update h values for all states in the tree

        for i in range(imp_num+1,len(str_tree)):
            st = parent[i]
            
            if st > imp_num:
                h_tree[i] = float(h[st_list[st]]) + cost2parent[i]

        print('policy evaluated')
        print('-------------')

        #reset the policy count and B_1

        pol_count = imp_num+1

        st_list = {}
        st_list[1] = 0

        st_list_rev = [1]
        st_len = 1

        #########################################################################
        # POLICY UPDATE
        #########################################################################

        #initialize K_cost

        pp = speak_dist[-1]

        cost = 0

        for k1 in range((imp_num**K-1)//(imp_num-1),(imp_num**(K+1)-1)//(imp_num-1)):
            pt = prob_tree[k1]
            cost += pp * pt * h_tree[k1]

        cost += speak_mean_trun/speak_mean*imp_mean


        for i in range(0,imp_num+1):
            temp_h_tree[i] = avg_cost
            cost2parent[i] = 0
            parent[i] = i
            K_cost[i] = cost

        #policy updates for every state of length greater than 1

        for i in range(imp_num+1,len(str_tree)):

            current_state = str_tree[i]
            current_len = len(current_state)

            temp_pol = 0

            #cost_suffix: optimal cost until reaching the reference state if the first element is skipped

            cost_suffix = imp_dist[0][int(current_state[0])]/speak_mean + temp_h_tree[(i-1)//imp_num]

            #Below, optimal cost until reaching the reference state is calculated when the first element is taken (unless the first element is unimportant)

            if current_state[0] != '0':

                j = int((i-1)/imp_num) 

                current_state = str_tree[j]
                current_len = len(current_state)

                sum_check = 0

                cost = eta*(current_len)

                k_cost = 0

                for k in range(1,K-current_len+1):

                    pp = speak_dist[k-1]

                    for k1 in range((imp_num**k-1)//(imp_num-1),(imp_num**(k+1)-1)//(imp_num-1)):
                        pt = prob_tree[k1]
                        
                        new_str = current_state+str_tree[k1]
                        new_len = len(new_str)

                        if new_len == K:
                            k_cost += pp * pt * h_tree[find_idx(new_str,imp_num)]

                        else:
                            cost += pp * pt * h_tree[find_idx(new_str,imp_num)]


                k_cost += sum(speak_dist[K-current_len:])*imp_dist[0][int(current_state[0])]/speak_mean
                k_cost += K_cost[int((i-1)/imp_num)]


                cost += k_cost


                K_cost[i] = k_cost
                K_cost[i-1] = k_cost

                #if taking the first element incurs less cost, update the policy accordingly and add the state to B_1 

                if cost < cost_suffix:
                    temp_pol = 0
                    temp_h_tree[i] = cost
                    cost2parent[i] = 0
                    parent[i] = i
                    st_list[i] = st_len
                    st_list_rev.append(i)
                    st_len += 1

                else:
                    temp_pol = pol_tree[(i-1)//imp_num]+1
                    temp_h_tree[i] = cost_suffix
                    cost2parent[i] = imp_dist[0][int(str_tree[i][0])]/speak_mean + cost2parent[(i-1)//imp_num]
                    parent[i] = parent[(i-1)//imp_num]

            else:
                temp_pol = pol_tree[(i-1)//imp_num]+1
                temp_h_tree[i] = cost_suffix
                cost2parent[i] = imp_dist[0][int(str_tree[i][0])]/speak_mean + cost2parent[(i-1)//imp_num]
                parent[i] = parent[(i-1)//imp_num]

            if temp_pol == pol_tree[i]:
                pol_count += 1

            pol_tree[i] = temp_pol


        #If the policy vector is unchanged, the algorithm terminates.

        if pol_count == len(str_tree):
            print('policy not changed')
            return(K, str_tree, h_tree, temp_h_tree, prob_tree, pol_tree, parent, cost2parent, avg_cost, st_list, st_list_rev, K_cost)
            

        else:
            print('policy updated')
            print('# of total states: ' + str(len(str_tree)))
            print('# of zero_states: ' + str(st_len))
    

    pass
    


def main(): 

    """
    Computes the optimal policy and the optimal average cost given the problem setup. For each eta value, the optimal cost is written to a csv file. Must be run with the following arguments:

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
    """ 

    try:
        if sys.argv[1]:
            filename = sys.argv[1]
            eta_max = float(sys.argv[2])
            eta_min = float(sys.argv[3])
            eta_num = int(sys.argv[4])
            print('filename: ', filename)
            print('eta_min : ' , eta_min, 'eta_max : ' , eta_max, 'eta_num : ' , eta_num)

    except:
        print('no argument given - default mode')
        filename = 'output'
        eta_min = 9*0.3
        eta_max = 9*0.3/12
        eta_num = 20

    try:
        if sys.argv[5]:
            imp_num = int(sys.argv[5])
            imp_dist = [[float(sys.argv[6+2*i]) for i in range(0,imp_num)],[float(sys.argv[7+2*i]) for i in range(0,imp_num)]]
            print(imp_dist)

    except:
        print('importance distribution error - default mode')
        imp_dist = [[1,20],[0.8, 0.2]]
        imp_num = len(imp_dist[0])

    try:
        if sys.argv[6+2*imp_num] == 'binomial':
            sp_dist = sys.argv[6+2*imp_num]
            sp_param = [float(sys.argv[7+2*imp_num]),float(sys.argv[8+2*imp_num])]
            

        elif sys.argv[6+2*imp_num] == 'bernoulli':
            sp_dist = sys.argv[6+2*imp_num]
            sp_param = [float(sys.argv[7+2*imp_num]),float(sys.argv[8+2*imp_num]),float(sys.argv[9+2*imp_num])]

        else:
            sp_dist = sys.argv[6+2*imp_num]
            sp_param = float(sys.argv[7+2*imp_num])

    except:
        print('speaking distribution error - geometric distribution with parameter 0.5 is set as default')
        sp_dist = 'geom'
        sp_param = 0.5


    eta_range = np.logspace(np.log10(eta_max),np.log10(eta_min),eta_num)

    file = open(filename + '.csv', 'w')

    writer = csv.writer(file)

    writer.writerow(['eta','avg'])

    eta = eta_range[0]
    print('eta: ' + str(eta))
    print(sp_param)
    (K, str_tree, h_tree, temp_h_tree, prob_tree, pol_tree, parent, cost2parent, avg_cost,st_list, st_list_rev, K_cost) = find_optimal_policy(imp_dist, imp_num, sp_dist, sp_param, eta)
    writer.writerow([eta,avg_cost])

    print('***************************************************************************')

    for eta in eta_range[1:]:
        print('eta: ' + str(eta))
        (K, str_tree, h_tree, temp_h_tree, prob_tree, pol_tree, parent, cost2parent, avg_cost, st_list, st_list_rev,K_cost) = find_optimal_policy(imp_dist, imp_num, sp_dist, sp_param, eta, K, str_tree, h_tree, temp_h_tree, prob_tree, pol_tree, parent, cost2parent, avg_cost,st_list, st_list_rev,K_cost)
        writer.writerow([eta,avg_cost])
        print('***************************************************************************')


    file.close()

if __name__ == '__main__':
	main()