"""
Temporary file for scaffolding the development.
"""

import random

import titivillus

def main():
    s1 = titivillus.random_stemma(num_roots=2)
    # s2 = titivillus.random_stemma("tiago",num_roots=2)

    s1.print()

    # s1.to_dot()

    for i in range(3):
        c = titivillus.Codex(
            chars=(1, 2, 3, 4, 5),
            origins=(("copy", 1), ("copy", 1), ("copy", 1),
                     ("copy", 1), ("copy", 1),),
            age=random.randint(1, 10)
        )
        print(c)

    chars1 = (1, 2, 3, 4, 5)
    chars3 = (1, 2, 4, 5, 6, 7)
    origins1 = ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1)
    origins3 = ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1)
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
    print("edit", titivillus.codex.edit_distance("kitten", "sitting"))
    print("jaccard", titivillus.codex.jaccard_distance("kitten", "sitting"))
    print("mmcwpa", titivillus.codex.mmcwpa_distance("kitten", "sitting"))

main()