
import os
import re
import torch

# reads from torch files
class CaptureLoader:

    def __init__(self, rootDir, onlyReadThose=None):
        self.rootDir = rootDir
        self.onlyReadThose = onlyReadThose
        
        self.prepare()
        
    def prepare(self):
        self.batchData = {}
        for p,sd,f in sorted(os.walk(self.rootDir)):
            for name in sorted(f):
                #print(p,sd,f,name)
                capturedThing = '_'.join(name.split('_')[:-1])
                if self.onlyReadThose and capturedThing not in self.onlyReadThose:
                    continue
                batchDescr = name.split('_')[-1].split('.')[0]
                batchNums = list(map(int, re.findall(r'\d+', batchDescr)))
                batchBegin, batchEnd = batchNums[0], batchNums[1]
                if (batchBegin, batchEnd) in self.batchData:
                    self.batchData[(batchBegin, batchEnd)][capturedThing] = os.path.join(p, name)
                else:
                    self.batchData[(batchBegin, batchEnd)] = {capturedThing: os.path.join(p, name)}
                #tensor = torch.load(os.path.join(p, name))
        self.batchesNamesInOrder = sorted(self.batchData.keys())

    def __len__(self):
        return len(self.batchesNamesInOrder)
        
    def __getitem__(self, idx):
        paths = self.batchData[self.batchesNamesInOrder[idx]]
        return {whatCaptured: torch.load(tensorPath) for whatCaptured, tensorPath in paths.items()}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    
if __name__ == '__main__':

    cl = CaptureLoader("/pio/scratch/1/i283340/MGR/zs/capture/try20/8")

    for data in cl:
        print(data.keys(), [t.shape for t in data.values()])

    cl2 = CaptureLoader("/pio/scratch/1/i283340/MGR/zs/capture/try20/8", ('ctx', 'cpcctc_align', 'phone_align'))

    for data in cl2:
        print(data.keys(), [t.shape for t in data.values()])