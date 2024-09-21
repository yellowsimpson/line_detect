from pathlib import Path

path = Path('my_favorite.txt')

contents = Path.read_text(path)

contents_list = contents.splitlines()
contents_list1 = list(set(contents_list))

for i, word in enumerate(contents_list1):
    contents_list1[i] = word.lstrip()
    
for j, word in enumerate(contents_list1):
    contents_list1[j] = word.rstrip()
    
print(contents_list1)