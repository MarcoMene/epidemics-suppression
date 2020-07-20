cs = [0, 1]

l = [lambda x, c=c: x + c for c in cs]
print(l[0](0))
