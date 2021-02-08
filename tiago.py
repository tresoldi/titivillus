"""
Temporary file for scaffolding the development.
"""

import random

import titivillus

s1 = titivillus.random_stemma(num_roots=2)
# s2 = titivillus.random_stemma("tiago",num_roots=2)

s1.print()

#s1.to_dot()

for i in range(3):
    c = titivillus.Codex(
        chars=(1, 2, 3, 4, 5),
        origins=(("copy", 1), ("copy", 1), ("copy", 1),
                 ("copy", 1), ("copy", 1),),
        age=random.randint(1, 10)
    )
    print(c)

chars1 = (1, 2, 3, 4, 5)
chars2 = (1, 2, 3, 4, 5)
origins = ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1), ("copy", 1)
age = 1.0

codex1 = titivillus.Codex(chars1, origins, age)
codex2 = titivillus.Codex(chars2, origins, age)
print("--", titivillus.codex_distance(codex1, codex2))

print("kitten")
d = titivillus.codex.edit_distance("kitten", "sitting")
print(d)