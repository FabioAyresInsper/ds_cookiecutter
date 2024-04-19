from git import Repo
import os 

def sanity_check(model, tokenizer, dataset, preprocessing):
    device = next(model.parameters()).device
    print(preprocessing(['this is a test', 'this is another test']))
    texts = [dataset[0]]
    print(texts)
    tokens, _ = preprocessing(texts)
    print(tokens.shape)
    x = tokens.to(device)
    print(x.shape)
    try:
        y = model(x)
        print("Sanity check passed! Device: " + str(device))
        return True
    
    except:
        print("Sanity check failed! Tried device: " + str(device))
        return False
    
def git_check_if_commited():
    repo = Repo(os.getcwd())
    
    # Check for untracked files
    if len(repo.untracked_files) > 0:
        print("There are untracked files in your repository! You must track or delete them before going forth!")
        #print(repo.untracked_files)
        for d in repo.untracked_files:
            print(d)
        return False        
    
    # Check for unstaged modifications
    diffs = repo.index.diff(None)
    if len(diffs) > 0:
        print("Halt! Unstaged modifications lurk behind these codes!\nThou shall 'git add' thy modifications to these files before venturing forth:")
        for d in diffs:
            print(d.a_path)
        return False
    
    # Check for committed modifications
    diffs = repo.index.diff(repo.head.commit)
    if len(diffs) > 0:
        print("Halt! You are ahead of HEAD!\nThou shall 'git commit' thy modifications to these files before venturing forth:")
        for d in diffs:
            print(d.a_path)
        return False

    return True

class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = 999999999.9
        self.waiting = 0
    
    def __call__(self, x):
        if x < self.best_loss:
            self.best_loss = x
            self.waiting = 0
        else:
            self.waiting += 1
        
        if self.waiting >= self.patience:
            return True
        else:
            return False
        