class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.set = set([])
        
    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.set:
            self.set.add(val)
        else:
            return False
        return True
    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.set:
            self.set.discard(val)
        else:
            return False

        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        self.len= len(self.set)
       
        
        return list(self.set)[random.randrange(0,self.len)]
