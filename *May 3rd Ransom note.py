def canConstruct(self, ransomNote, magazine):
    return not collections.Counter(ransomNote) - collections.Counter(magazine)
    
 #this solution is fast because it use the counter function inside collection 
 
 if the counters are same, result is 0 and the return is not 0 which is 1
