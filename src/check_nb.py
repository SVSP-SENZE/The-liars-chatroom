import json
nb = json.load(open('colab_liars_chatroom_solution.ipynb'))
print('Cells in notebook:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    ln = len(c['source'])
    print(f'  Cell {i}: {ct} ({ln} lines)')
