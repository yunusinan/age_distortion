import numpy as np
import sys
import time
import copy
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing import Pool, cpu_count



class node():

    def __init__(self, state, idx, table, nodes, prob, limit, p_success, mean, str_rep, eta = 0, truncated=0, policy=0):
        self.state = state
        self.length = len(self.state)
        self.idx = idx
        self.Hvalue = 0
        self.limit = limit
        self.table = table
        self.strategies = []
        self.truncated = truncated
        self.nodes = nodes
        self.prob = prob
        self.mean = mean
        self.eta = eta
        self.p_success = p_success
        self.policy = policy
        self.str_rep = str_rep

        if(truncated):
            self.length = 1
            self.state = [0]

    def getUpdatedHValue(self):
 

        finished = 0
        lower_bnd = 0
        mean = self.mean
        p_success = self.p_success
        eta = self.eta

        

        # if(self.stopped == 0):
        l = self.length
        s = 0

        self.strategies = []
        st_wc = []
        # else:
        #     l = self.policy+2
        #     s = self.policy+1
            # print('stopped')

        # print(self.state)

        for t in range(s, l):

            # next_states = []

            i = l-t-1

            if(t == -1 and self.truncated == 1):
                aaaa = 0
            else:
                aaaa = self.limit - (l - i)

            # if(t == 0):
            exp_max_prob = 0
            max_prob_acc = 0

            sss = 0
            list_range1 = int((3**(aaaa + 1) - 3) / 2)
            states_i = ''.join(map(str, self.state[i + 1:]))
            prob_acc = 0
            list_range2 = int((3**(aaaa) - 3) / 2)

            for j in range(0, list_range1):
                states_j = ''.join(map(str, self.table[j][0]))
                new_name = states_i + states_j
                n = self.nodes[new_name]
                n_next = self.nodes[states_j]

                sss += n.Hvalue * n_next.prob
                # exp_max_prob += n.Hvalue * n_next.prob
                # print(sss)
                prob_acc += n_next.prob
                # print(prob_acc)
                if j >= list_range2:
                    exp_max_prob += n.Hvalue * n_next.prob
                    max_prob_acc += n_next.prob
                    # print(new_name)
                    # print(exp_max_prob,max_prob_acc)
                    # time.sleep(0.1)
                #     # for j in range (list_range2,list_range1):
                #     #states_j = ''.join(map(str,self.table[j][0]))

                #     n = self.nodes[new_name + 't']
                #     n_next = self.nodes[states_j + 't']

                #     # next_states.append([n.state,n_next.prob]);

                #     sss += n.Hvalue * n_next.prob
                # print(sss);
                
            # print(next_states)

            # print(exp_max_prob, max_prob_acc)
            # time.sleep(0.1)

            # if(self.truncated == 1):
            expected_max_buffer = exp_max_prob/max_prob_acc
            st_wc.append(expected_max_buffer)
            # print(expected_max_buffer)
                # time.sleep(1000)

            if(finished == 0):
                # if(i < l-1):
                if(lower_bnd == 0):
                    sss += (1 - prob_acc) * (mean*(aaaa + 1/p_success-1) + sum(self.state[i + 1:]))

                self.strategies.append(sum(self.state[0:i]) + eta * (l - i - 1) / (p_success) + sss)

                # print('here')
                # print(self.state, self.strategies);

            else:
                sss2 = 0
                # print('finished')
                    # if(self.truncated == 0):
                        # print('here0')
                        # print(sss)
                        # sss += (1 - prob_acc) * (expected_max_buffer)
                        # print('last idx and truncated')
                    # print(lambda_hat)

                sss2 += sum(((1-p_success)**np.arange(0,l-i-1))*np.array(self.state[i+1:]))

            

                temp = st_wc

                # print(self.state,st_wc,np.arange(0,l-i))
                sss2 += (p_success)*sum(((1-p_success)**np.arange(0,l-i))*np.array(temp[::-1]))

                # print(st_wc)
                # sss2 += ((1-p_success)**(t))*np.array(st_wc[0])
                # print(st_wc)
                # print(sum(p_success*((1-p_success)**np.arange(0,l-i-1))*np.array(st_wc[-t:])))
                # sss2 += p_success/(1-p_success)


                # if(sss2 < 0):
                #     print('wwww1')
                #     print(sss2)

                # sss2 -= sum(eta*(np.arange(t-1,-0.5,-1)*((1-p_success)**np.arange(0,l-i-1))))
                # print(sss2)
                # print(self.strategies)
                # print(st_wc)

                # if(sss2 < 0):
                #     print('wwww2')
                #     print(sss2)
                    # print(prob_acc)


                    # time.sleep(1000)

                # sss2 -= sum(self.state[0:i+1])
                # print(sss2)
                # if(sss2 < 0):
                #     print('wwww3')
                #     print(sss2)
                #     print(prob_acc)
                #     print(i)


                    # time.sleep(1000)


                sss2 *= (1-prob_acc)



                sss2 += (1-p_success)**(self.limit-1)*(st_wc[0] + mean/p_success)

                # if(i != l-1):
                #     print(self.state, i, sss2, prob_acc)
                # print('aaaa')
                # print(sss, sss2)

            # print(sss,sss2,t)

                # print(self.state,(1 - prob_acc),i,l,sss2)

                # print('------------------------------------------------------\n')
                # print(i,sss)
                # print(self.state, sum(self.state[0:i])*(prob_acc) - (1-prob_acc)*self.state[i] + eta * (l - i - 1) / (p_success) + sss)


                if(l == 1):
                    
                    self.strategies.append(sss+sss2)
                    # st_wc.append(expected_max_buffer)
                else:
                    self.strategies.append(sum(self.state[0:i]) + (eta * (l-i-1) / (p_success)) + sss + sss2)
                    # st_wc.append(expected_max_buffer)
                    # print((eta * (l-i-1) / (p_success)))
            
                    # if(sss + sss2 < 0):
                # print('wwwww')
                
                # print(sss,sss2,prob_acc,eta * (l-i-1) / (p_success) )
                # time.sleep(1)

                # if(self.truncated == 1):
                #     lambda_hat = self.strategies
            # print(self.state, self.strategies)

                
        # print(self.state, self.strategies)


        # print(self.state, self.strategies)
        # print(self.state, np.flipud(self.strategies), st_wc)
        # time.sleep(0.5)
        self.policy = l-1-np.argmin(self.strategies)

        # stop = 0

        # if self.policy != 0 and self.stopped == 0:
        #     for elm in self.state[0:self.policy-1]:
        #         if elm == 3:
        #             self.stopped = 1
        #             stop = 1
        #             # print(self.state, self.policy)
        #             # time.sleep(5)
        #             break
                    
        # if stop == 1:
        #     # print('-----------------------------------------')
        #     aaaa = self.limit-self.length-1
        #     list_range1 = int((3**(aaaa+1) - 3) / 2)
        #     states_i = ''.join(map(str, self.state))

        #     for j in range(0, list_range1):
        #         states_j = ''.join(map(str, self.table[j][0]))
        #         length_j = len(self.table[j][0])
        #         new_name = states_j + states_i
        #         # print(new_name)
        #         n = self.nodes[new_name]
        #         n.stopped = 1;
        #         n.policy = length_j + self.policy





        return [min(self.strategies), self.policy]

    def setHValue(self, val):
        self.Hvalue = val

    def setEtaValue(self, val):
        self.eta = val

    def setPolicy(self, val):
        self.policy = val

    def setLimit(self, val):
        self.limit = val

    def generateRowPolicy(self,policy,support):

        mean = self.mean
        # self.strategies = []
        eta = self.eta
        p_success = self.p_success
        l = self.length
        i = policy
        s = 0
        str_rep = self.str_rep

        list_range1 = int((support**(self.limit) - support) / (support-1))

        row_pol = np.zeros(list_range1+2)

        row_pol[self.idx] += 1
        prob_acc = 0

        aaaa = self.limit - (l - i)

        list_range2 = int((support**(aaaa+1) - support) / (support-1))

        sss = 0
        states_i = str_rep[i + 1:]
        list_range3 = int((support**(self.limit-1) - support) / (support-1))


        for j in range(0, list_range1):
            states_j = self.table[j][2]
            new_name = states_i + states_j

            # if(self.state == [3,1]):
            #     print(new_name)
            jump = (len(new_name)-(self.limit-1))

            new_name = new_name[-(self.limit-1):]

            # if(self.state == [3,1]):
            #     print(new_name)
            
            n = self.nodes[new_name]
            n_next = self.nodes[states_j]

            row_pol[n.idx] += -n_next.prob
            prob_acc += n_next.prob

            if j >= list_range2:
                # print(self.state, new_name ,jump)
                row_pol[list_range1 + 1] += n_next.prob*sum(self.state[i + 1: i + 1 + jump])

            if j >= list_range3:
                row_pol[n.idx] += -(1-p_success)/p_success*n_next.prob

            # if(self.state == [3,1]):
            #     print(row_pol)

            # if(''.join(map(str,self.state)) == '31'):
                # print(new_name)
                # print(row_pol[table_len + 1] )
            # print(sss);
        # print(next_states)

        # print('------------------------------------------------------\n')


        temp = (1-prob_acc)*sum(self.state[i + 1:])

        row_pol[list_range1 + 1] += (1-prob_acc)*(mean/p_success) + temp

        row_pol[list_range1] = 1
        row_pol[list_range1 + 1] += sum(self.state[0:i]) + eta * (l - i - 1) / (p_success)



        # print(prob_acc)

        
        # print((1-prob_acc)*(mean/p_success))

        return row_pol


def generateStates(max_buffer, p_success, prob_arr, d):

    support = len(d)
    idx = [0]
    i = 1
    k = 1
    states = []
    prob_success = p_success

    while k <= (support**(max_buffer + 1) - support) / (support-1):

        idx[0] = i % (support+1)

        disp = 1

        prob = 1

        for j in range(0, len(idx)):
            if idx[j] == 0:
                if j == len(idx) - 1:
                    idx[j] = 1
                    idx.append(1)
                    prob_success = prob_success * (1 - p_success)

                else:
                    idx[j + 1] = (idx[j + 1] + 1) % (support+1)
                    idx[j] = 1

                    # i += 1;
                disp = 0

            prob = prob * prob_arr[idx[j] - 1]

        if(disp):
            k += 1

            state_dist = []

            for elm in idx:
                state_dist.append(d[elm-1])

            states.append([state_dist, prob * prob_success,
                           ''.join(map(str, copy.copy(idx)))])
            # print([prob,prob_success,prob*prob_success])
            # print(''.join(str(idx)))

        i += 1

    return states


# def generatePolicies(states, num_3, num_2, d):
#     policies = []

#     for i in range(0, len(states)):
#         trun2 = 0
#         trun3 = 0

#         s = states[i][0]
#         l = len(s)
#         idx2 = l - 1
#         idx3 = l

#         for j in range(0, l):
#             if s[l - 1 - j] == d[1]:
#                 if trun2 < num_2:
#                     idx2 = l - 1 - j
#                 trun2 += 1
#             elif s[l - 1 - j] == d[2]:
#                 if trun3 < num_3:
#                     idx3 = l - 1 - j
#                 trun3 += 1
#             if(trun2 >= num_2 and trun3 >= num_3):
#                 break

#         if idx3 == l:
#             policies.append(idx2)
#         else:
#             policies.append(idx3)

#     return policies

def getUpdatedHValue(node):
    return node.getUpdatedHValue()


def _main_():

    node_arr = {}
    max_buffer = 7
    p_success = 0.2
    d = [1, 100]
    prob_arr = [0.6,0.4]
    support = len(d)

    mean = sum(i[0] * i[1] for i in zip(d, prob_arr))

    states = generateStates(max_buffer, p_success, prob_arr, d)

    # policies = generatePolicies(states, 1, 2, d)

    st_no = len(states)


    # for i in range(0, st_no):
    # print([states[i][0], policies[i]])

    temp_buffer_size = 2
    temp_st_no = int((support**(temp_buffer_size+1)-support)/(support-1))

    # print(temp_st_no)
    policies = []

    # node_arr['t'] = node([], 0, states, node_arr, (1 - p_success) ** (max_buffer), max_buffer + 1, p_success, mean, 0, truncated=1)

    for i in range(0, temp_st_no):
        node_arr[states[i][2]] = node(
            states[i][0], i, states, node_arr, states[i][1], temp_buffer_size + 1, p_success, mean, states[i][2])
        policies.append(len(states[i][0])-1)
    # print(np.abs(last_val-current_val))

    # time.sleep(1000)
    # try:
    #     workers = cpu_count()
    # except NotImplementedError:
    #     workers = 1
    # pool = Pool(processes=workers)

    f = open("dynprog4_lb.txt", "w")
    f2 = open("dynprog4_table.txt", "w")

    eta_limit = (max(d)-min(d))*p_success

    print(eta_limit)

    eta_list = np.logspace(np.log10(eta_limit),-4,num = 30)
    updates = 0

        # print([states[i][0], Hv[i][1]])
        
        # node_arr[states[i][2]].setHValue(0)
        
        # node_arr[states[i][2]].setPolicy(len(states[i][0])-1)
        # node_arr[states[i][2]].setStopped(0)
        # row.append(node_arr[states[i][2]].generateRowPolicy(len(states[i][0])-1))


    temp = [0]*(temp_st_no+2)
    temp[0] = 1

    row = np.zeros((temp_st_no+1,temp_st_no+2))

    for eta in eta_list:

        for i in range(0, temp_st_no):
            node_arr[states[i][2]].setEtaValue(eta)
            node_arr[states[i][2]].setPolicy(policies[i])
            row[i,:] = node_arr[states[i][2]].generateRowPolicy(policies[i],support)

        row[temp_st_no,:] = np.array(temp)

        # lambda_hat = 0

        changed = 1
        augment = 0

        start = time.time()

        while changed == 1:

            if(augment == 1):

                augment = 0
                temp_buffer_size += 1

                print('here')
                print(temp_buffer_size)
                temp_st_no = int((support**(temp_buffer_size+1)-support)/(support-1))

                for i in range(int((support**(temp_buffer_size)-support)/(support-1)), temp_st_no):
                    node_arr[states[i][2]] = node(states[i][0], i, states, node_arr, states[i][1], temp_buffer_size + 1, p_success, mean,states[i][2])
                    policies.append(len(states[i][0])-1)


                temp = [0]*(temp_st_no+2)
                temp[0] = 1
                row = np.zeros((temp_st_no+1,temp_st_no+2))

                # print((temp_st_no+1,temp_st_no+2))

                for i in range(0, temp_st_no):

                    node_arr[states[i][2]].setLimit(temp_buffer_size + 1)
                    node_arr[states[i][2]].setEtaValue(eta)
                    node_arr[states[i][2]].setPolicy(policies[i])
                    row[i,:] = node_arr[states[i][2]].generateRowPolicy(policies[i],support)

                row[temp_st_no,:] = np.array(temp)

            changed = 0
            A = row[:,range(0,temp_st_no+1)]
            b = row[:,temp_st_no+1]
            # print(A)

            h = np.linalg.solve(A,b)

            row_temp = copy.copy(row)
            # print(h[-1]*p_success)

            for i in range(0,temp_st_no):

                p = policies[i]
                current_h = h[i]

                # print(current_h)

                while p > 0:

                    state = states[i][2]
                    policy_val = state[p]

                    p -= 1

                    if(states[i][0][p] > 1):

                    # print(row[i,:])
                        row_temp[i,:] = node_arr[states[i][2]].generateRowPolicy(p,support)


                        # print(row_temp[i,:])
                        A_temp = row_temp[:,range(0,temp_st_no+1)]
                        b_temp = row_temp[:,temp_st_no+1]

                        h_temp = np.linalg.solve(A_temp,b_temp)
                        next_h = h_temp[i]

                        # print(states[i][2])

                        # print(states[i][2])
                        # print(next_h, current_h, p)
                  

                        if(next_h < current_h):
                            policies[i] = p
                            # print('eere',states[i][2])
                            # print(next_h, current_h, p)
                            row[i,:] = row_temp[i,:]
                            current_h = next_h
                            changed = 1

                            if(p == 0 and len(state) == temp_buffer_size):
                                print(node_arr[states[i][2]].state)

                                if(temp_buffer_size < max_buffer):
                                    augment = 1
                            elif(p == 0):
                                print(node_arr[states[i][2]].state)

                        else:
                            # print('1 seen')
                            row_temp[i,:] = row[i,:]
                    # break

            row = copy.copy(row_temp)


        end = time.time()
        # print(policies)        

        opt = h[-1]*p_success
        print([eta, opt, end-start])
        # print(policies[int((3**(max_buffer) - 3) / 2):])
        f.write(str(eta)+'\t'+str(opt)+'\n')

        

        # print(node_arr['t'].Hvalue)

        line_color = 'r' if eta >= eta_limit/(max_buffer) else 'b' 

        plt.plot([0, (opt-0) / eta], [opt, 0], line_color)

        plt.draw()
        time.sleep(0.001)

    for i in range(0,temp_st_no):
        print(states[i][0], h[i], policies[i])
        f2.write(states[i][2]+'\t'+str(policies[i])+'\n')

    plt.show()
    f.close()
    f2.close()

if __name__ == '__main__':
    _main_()
