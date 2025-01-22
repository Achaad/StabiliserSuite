import time

from matplotlib import pyplot as plt
from clifford_builder.clifford import sample_clifford_group

if __name__ == '__main__':
    start = time.process_time_ns()
    qcirc = sample_clifford_group(40, True)
    end = time.process_time_ns()
    qcirc.draw("mpl")
    plt.show()
    draw = time.process_time_ns()

    print(f"Elapsed time: {(end - start) / 1000000} ms")
    print(f"Draw time: {(draw - end) / 1000000000} s")