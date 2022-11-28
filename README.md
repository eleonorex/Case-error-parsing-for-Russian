# Case error parsing for Russian

---

## NLP project in autocorrection for Russian language.

**Structure**

Pickle files contain correct case dependencies in following format:

'verb': {'preposition(optional)': {'case': {'complement': frequency}

for collocations of type _verb + noun_ and _noun + noun_.

---

When the program is given a text:

1. some case errors are generated by randomly replacing one case in a sentence (the word form is also changed in order to match new, incorrect case)
2. new sentences are then parsed with _spacy dependency matcher_ to find all collocations of a certain syntactic dependency type
3. by comparing parsed dependencies with corresponding data in pickle files, the number of identified and missed errors is counted, writing the results in a new file.
