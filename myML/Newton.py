class myNewton():
    def __init__(self, x=8, error=0.001):
        self.x = x
        self.errlist = []
        self.xlist = []
        self.error = error
    
    def formula(x):
         return x - (3*x**2-1)/(6*x)

    def fit(self):
        while True:
            x2 = self.x - formula(self.x)
            self.errlist.append(abs(x2 - self.x))
            self.xlist.append(self.formula())
            if abs(x2 - self.x) < self.error:
                break
            self.x = x2
    
    def print_ans(self):
        print("x = {0:.5f}".format(self.x))

    def print_err(self):
        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(self.errlist)
    
    def print_graph(self):
        pass
        

if __name__ == '__main__':
    N = myNewton(8, 0.0001)
    N.fit()
    N.print_ans()
    N.print_err()

        
        
