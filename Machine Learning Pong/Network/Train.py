from Network import Network
from operator import itemgetter
import numpy as np
from Evaluate import evaluate


def evalGen(gen, max_steps):
    scores = []
    for nn in gen:
        evaled = evaluate(nn, max_steps=max_steps, draw=False)
        scores.append((evaled[0], nn, evaled[1]))
    return scores


class Train:
    def __init__(self, generations, size=100, shape=None):
        if shape is None:
            shape = [8, 8, 8]
        self.generations = generations
        self.size = size
        shape.append(1)
        self.shape = (3, shape)

    def train(self, slice_size=10, draw_frequency=10, mutation_rate=0.5, change_value=5, max_steps=100000):
        gen = []
        best = []

        for i in range(self.size):
            gen.append(Network(self.shape[0], self.shape[1]))

        for i in range(self.generations):

            scores = evalGen(gen, max_steps=max_steps)
            scores = sorted(scores, key=itemgetter(0), reverse=True)
            best = scores[0:self.size // slice_size]

            newGen = []

            for nn in best:
                network = nn[1]
                weights, biases = network.getWeights()

                for j in range(len(weights)):
                    weights[j] = np.copy(weights[j])

                for j in range(slice_size-1):
                    net = Network(self.shape[0], self.shape[1], weights=weights, bias=np.copy(biases))
                    net.update(mutation_rate=mutation_rate, change_value=change_value)
                    newGen.append(net)

                newGen.append(network)

            gen = newGen

            # Run some inference type things

            unique = []

            for network in scores:
                if network[0] not in unique:
                    unique.append(network[0])
            print(best)
            print(unique)

            if best[0][2]:
                break

            if i % draw_frequency == 0 and i > 0:
                evaluate(best[0][1], max_steps=max_steps, draw=True)

        w, b = best[0][1].getWeights()
        with open('best.txt', 'w') as f:
            f.write('weights='+str(w)+'\n'+'bias='+str(b)+'\n')
        print("weights=" + str(w), "bias=" + str(b))
        evaluate(best[0][1], max_steps=max_steps, draw=True)

# t = Train(500, size=100)
# t.train()
