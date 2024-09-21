list1 = []
for i in range(101):
    if i % 2 == 0:
        list1.append(i)
        
print(list1)

print([i for i in range(101) if i % 2 == 0])

list2 = []
for j in range(101):
    list2.append(j)
print(list2[::2])