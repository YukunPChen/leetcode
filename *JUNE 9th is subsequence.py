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


# the main idea in this solution is by using iterator 
# in this way, if you find one character exist in a iterator, the iteraotr will only test the remaining character
# need more practice to understand iterator in python
