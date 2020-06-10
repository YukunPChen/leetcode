#method  1

def isSubsequence(self, s, t):
    remainder_of_t = iter(t)
    for letter in s:
        if letter not in remainder_of_t:
            return False
    return True
    
    
#method 2
def isSubsequence(self, s, t):
    t = iter(t)
    return all(c in t for c in s)
