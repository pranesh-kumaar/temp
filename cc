#Creating First Generation of Trade Signals

population = 10
signal_population = []
fill = [0]*population
layer_weights = [[fill],[fill],[fill]]

for x in range(0,population):
    signal =[]
    for z in neuron_cycle: 
        n1 = z
        random_weights()
        layer_2()
        layer_3()
        output = layer_4()
        signal.append(output)
        
        weights = [w1,w2,w3]
    
    for y in range(0,len(layer_weights)):
        layer_weights[y][x]=(weights[y])
        
    signal_population.append(signal)
