import numpy as np


def number_of_certain_probability(sequence, probability):
    x = np.random.random()
    print("x",x)
    cumulative_probability = 0.0
    print("sequence",sequence)
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
       # print(item)
    return item


if __name__ == '__main__':
    sum1 = 0
    p1 = 0
    alpha = 0.8
    probability = []
    for n in range(1, 21):
        sum1 += 1/(n**alpha)
    #print("sum1",sum1)
    for n in range(1, 21):
        p = (1/(n**alpha))/sum1
        probability.append(p)
        p1 += p
        #print(n, p)
    print("p",probability)
    with open("probability.txt","w") as f:
        for i in probability:
            f.write(str(i)+"\n")
    #print("sum(probality)",sum(probability))

    # value_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    value_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    a = number_of_certain_probability(value_list,probability)
    print("a",a)
    f_list = [20 - i for i in range(20)]
    print(f_list)