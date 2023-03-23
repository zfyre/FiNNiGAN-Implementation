import os

Datadir = os.path.join('data','output')
directory = os.fsencode(Datadir)
print(len(os.listdir(directory)))
for DIR in os.listdir(directory):
    dirname = os.fsdecode(DIR)
    sampledir = os.fsencode(os.path.join(Datadir,dirname))
    print(dirname,len(os.listdir(sampledir)))

    assert len(os.listdir(sampledir))==3