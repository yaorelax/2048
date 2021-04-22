
playground = [0, 2, 4, 0, 0, 0]
print(playground)

tmp = []
for a in playground:
    if a != 0:
        tmp.append(a)

print(tmp)

for _ in range(len(tmp) - 1):
    tip = 0
    while True:
        if tip < len(tmp) - 1:
            if tmp[tip] == tmp[tip + 1]:
                tmp[tip] *= 2
                del tmp[tip + 1]
            tip += 1
        else:
            break

print(tmp)