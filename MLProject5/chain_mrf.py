import numpy as np

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.n = p.chain_length()
        self.k = p.num_x_values()

    def unary_message(self,index):
        k = self.k+1
        msg = [0]*k
        for i in range(1,k):
            msg[i] = self._potentials.potential(index,i)
        return msg

    def binary_message(self,index,prev_node_msg):
        k = self.k+1

        msg = [0]*k
        
        potential = 0

        #For every msg in previous node
        for i in range(1,k):
            #For every msg in current node
            for j in range(1,k):
                potential  = self._potentials.potential(index,j,i)
                msg[i] += potential*prev_node_msg[j]
        return msg

    
    def forward_message(self):
        
        msg = [[]]

        n = self.n

        for i in range(1,n+1):
            if i == 1:
                #temp = self.unary_message(i)
                temp = [0]
                temp.extend([1]*self.k)
                msg.append(temp)
            else:
                temp = self.binary_message(n+i-1,msg[i-1])
                temp = (np.asarray(temp) * self.unary_message(i)).tolist()
                msg.append(temp)
                
    
        return msg


    def backward_message(self):
        
        msg = [[]]

        n = self.n

    
        for i in range(n,0,-1):
            if i == n:
                #temp = self.unary_message(i)
                temp = [0]
                temp.extend([1]*self.k)
                msg.append(temp)
            else:
                temp = self.binary_message(n+i,msg[-i+n])
                temp = (np.asarray(temp) * self.unary_message(i)).tolist()
                msg.append(temp)
                
        temp = msg[1:]
        temp.reverse()
        reversed_msg = [[]]
        reversed_msg.extend(temp)
        return reversed_msg
       

    def marginal_probability(self, x_i):
        # should return a python list of type float, with its length=k+1, and the first value 0

        #Initialize message before and after x_i. 

        all_forward_messages = []
        all_forward_messages = self.forward_message()
        #print(all_forward_messages)

        all_backward_messages = []
        all_backward_messages = self.backward_message()
        #print(all_backward_messages)

        idx = x_i

        res = (np.asarray(all_forward_messages[idx]) * np.asarray(all_backward_messages[idx])).tolist()
        
        newres = []

        #Normalize
        if sum(res) != 1:
            ressum = sum(res)
            for each_term in res:
                each_term /= ressum
                newres.append(each_term)

        return newres

        


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self.n = p.chain_length()
        self.k = p.num_x_values()
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        return 0.0
