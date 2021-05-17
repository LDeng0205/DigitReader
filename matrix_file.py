import random
def write(Theta, trained = False, filename = 'theta.txt'):
    """ write to the file a new set of random initial values
    """
    with open(filename, "w") as file:
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

def read(Theta, filename = 'theta.txt'):
    """ read data stored in theta.txt
    """
    with open(filename, "r") as file:
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
        