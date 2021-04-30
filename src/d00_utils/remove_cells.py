import re #python regular expression matching module
script = re.sub(r'# In\[.*\]:\n','',open('modelling.py').read())
with open('feat_eng_select_2.py','w') as fh:
    fh.write(script)

