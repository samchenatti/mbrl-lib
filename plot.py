import matplotlib.pyplot as plt

if __name__ == '__main__':
    # f = open('/home/figo/Develop/IC/mbrl-lib/exp/mbpo/default/gym___WalkingNao-v0/2023.06.25/193512/results.csv')
    # f = open('/home/figo/Develop/IC/mbrl-lib/exp/mbpo/default/gym___WalkingNao-v0/2023.06.24/212033/results.csv')
    f = open('/tmp/2023.07.24/183346/results.csv')
    r = [
        line.split(',')[1]
        for line in f
    ]

    r = [float(v) for v in r[1:]]

    plt.plot(list(range(len(r))), r)
    plt.show()
