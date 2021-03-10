import random
def write(Theta, trained = False):
    """ write to the file a new set of random initial values
    """
    with open("theta.txt", "w") as file:
        file.truncate(0)
        if not trained:
            for i in range(len(Theta)):
                for line in Theta[i]:
                    for num in line:
                        file.write(str(round(random.uniform(-1, 1), 4)) + " ")
                    file.write("\n")
        else:
            for i in range(len(Theta)):
                for line in Theta[i]:
                    for num in line:
                        file.write(str(num) + " ")
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
        