from matplotlib import pyplot as plt
from clifford_builder.clifford import sample_clifford_group

if __name__ == '__main__':
    qcirc = sample_clifford_group(4, True)

    qcirc.draw("mpl")
    plt.show()
