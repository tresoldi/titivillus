"""
Temporary file for scaffolding the development.
"""

import random

import titivillus


def main1():
    for i in range(3):
        c = titivillus.Codex(
            chars=(1, 2, 3, 4, 5),
            origins=tuple([titivillus.OriginCopy(1) for _ in range(5)]),
            age=random.randint(1, 10)
        )
        print(c)

    chars1 = (1, 2, 3, 4, 5)
    chars3 = (1, 2, 4, 5, 6, 7)
    origins1 = tuple([
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1)])
    origins3 = tuple([
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1),
        titivillus.OriginCopy(1)])
    age = 1.0

    codex1 = titivillus.Codex(chars1, origins1, age)
    codex2 = titivillus.Codex(chars1, origins1, age)
    codex3 = titivillus.Codex(chars3, origins3, age)

    for method in ["edit", "jaccard", "mmcwpa"]:
        print(method,
              titivillus.codex_distance(codex1, codex2, method=method),
              titivillus.codex_distance(codex1, codex3, method=method),
              )

    print("KITTEN")
    print("edit", titivillus.distance.edit_distance("kitten", "sitting"))
    print("jaccard", titivillus.distance.jaccard_distance("kitten", "sitting"))
    print("mmcwpa", titivillus.distance.mmcwpa_distance("kitten", "sitting"))
    print("mmcwpa", titivillus.distance.subseq_jaccard_distance("kitten", "sitting"))


def main2():
    x = titivillus.collect_subseqs("abcde")
    print(x)

    x = titivillus.collect_subseqs([c for c in "abcde"])
    print(x)

    for n in titivillus.ngrams_iter("abcde", 2):
        print(n)


def main3():
    s1 = titivillus.random_stemma()
    s2 = titivillus.random_stemma(num_roots=2)

    print(s1)
    print(s2)

    dg1 = s1.as_graph()
    dg2 = s2.as_graph()

    import matplotlib.pyplot as plt
    import networkx as nx
    nx.draw(dg1)
    plt.show()
    nx.draw(dg2)
    plt.show()


main3()
