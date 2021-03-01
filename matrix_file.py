import random
def write(Theta):
    """ write to the file a new set of random initial values
    """
    with open("theta.txt", "w") as file:
        file.truncate(0)
        for i in range(len(Theta)):
            for line in Theta[i]:
                for num in line:
                    file.write(str(round(random.uniform(-0.01, 0.01), 4)) + " ")
                file.write("\n")
    return

def read(Theta):
    """ read data stored in theta.txt
    """
    with open("theta.txt", "r") as file:
        idx, i, j = 0, 0, 0
        for line in file:
            for num in line.split():
                Theta[idx][i][j] = float(num)
                j += 1
            i, j = i + 1, 0
            if i == Theta[idx].shape[0]:
                idx += 1
                i = 0
    return Theta
        