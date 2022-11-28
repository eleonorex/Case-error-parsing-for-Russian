import pickle

with open('comb_verb_noun.pickle', 'rb') as f:
    data = pickle.load(f)

print(data)

dependencies = data['ходить']
print(dependencies)

next_deps = dependencies['_']
print(next_deps)
