import numpy 

with open('gridsearch/results_2.txt', 'r') as file:
    for line in file:
        l = line.strip().split()
        results = l[5:]
        total = 0
        for res in results:
            total += float(res.replace('[','').replace(']','').replace(',',''))
        print('Run id {} \t Average {:.2f}'.format(l[2], 100 * total / 5))



