
# coding: utf-8

# In[55]:


#playground for leet code


# In[ ]:


#1>. given a non-empty array of integers, every element appears twice except for one, find the single one


# In[23]:



class Solution:
    def __init__(self,list1):
        self.list=[]
        for i in range(len(list1)):
            if list1[i] in ['[',']']:
                1+1
            else:
                self.list.append(list1[i])
        
    def singleNumber(self):
        self.queue=[]
        for i in range(len(self.list)):

            if self.list[i] not in self.queue:
                self.queue.append(self.list[i])
            else:                    
                self.queue.remove(self.list[i])
        print(self.queue[0])
        


ss=Solution(input())
ss.singleNumber()


# In[130]:


#2>. write a algorithm to determine if a number is happy 


# In[9]:


class Solution:
    def isHappy(self,n:int) -> bool:
        n=Solution.multiadd(n)
        pool=[]
        while (n!=1):
            n= Solution.multiadd(n)
            if n in pool:
                break
            else:
                pool.append(n)
            #print("after calculation is : ",n)
            
        return(n==1)
    def multiadd(n):
        #print(n)
        mi=0
        while n/10 >0:  
            sin = n%10
            #print(sin)
            mis = sin*sin
            mi += mis
            n = n//10
            #print(n,"left, and the sum now is",mi)
        return mi


# In[13]:


x =  Solution()
x.isHappy(100)


# May 3rd
# Ransom Note
# 
# Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.
# 
# Each letter in the magazine string can only be used once in your ransom note.
# 
#  
# 
# Example 1:
# 
# Input: ransomNote = "a", magazine = "b"
# Output: false
# 
# Example 2:
# 
# Input: ransomNote = "aa", magazine = "ab"
# Output: false
# 
# Example 3:
# 
# Input: ransomNote = "aa", magazine = "aab"
# Output: true
# 
#  
# 
# Constraints:
# 
#     You may assume that both strings contain only lowercase letters.
# 
# 

# In[326]:


from collections import Counter
def canConstruct(ransomNote: str, magazine: str):
    For i in Counter(ransomNote):
        if i 
    
canConstruct('baa','aab')


# In[331]:


Counter('baa').values()


# #3>. given an inteeger array nums, find the contiguous subarray (containing at least one number)
# #which has the largest sum and return its sum 
# #**********important
# Example:
# 
# Input: [-2,1,-3,4,-1,2,1,-5,4],
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.
# 

# May 4th  Number Complement
# 
# Given a positive integer num, output its complement number. The complement strategy is to flip the bits of its binary representation.
# 
#  
# 
# Example 1:
# 
# Input: num = 5
# Output: 2
# Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
# 
# Example 2:
# 
# Input: num = 1
# Output: 0
# Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.
# 
#  
# 
# Constraints:
# 
#     The given integer num is guaranteed to fit within the range of a 32-bit signed integer.
#     num >= 1
#     You could assume no leading zero bit in the integer’s binary representation.
#     This question is the same as 1009: https://leetcode.com/problems/complement-of-base-10-integer/
# 
# 

# In[ ]:


class Solution:
    def findComplement(self, num: int) -> int:
        bina = bin(num)[2:]
        new = ''
        print(bina)
        for i in range(len(bina)):
            if int(bina[i]) == 1:
                new+='0'
            else:
                new+='1'
        new = '0b'+new
        
                
        return int(new,2)


# May 5th
#   First Unique Character in a String
# 
# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
# 
# Examples:
# 
# s = "leetcode"
# return 0.
# 
# s = "loveleetcode",
# return 2.
# 
#  
# 
# Note: You may assume the string contain only lowercase English letters.
# 

# In[ ]:


class Solution:
    def firstUniqChar(self, s: str) -> int:
        out= -1
        for i in range(len(s)):
            n = s[i]            
            if n in s[:i]:
                pass
            elif n in s[i+1:]:
                pass
            else:
                print('found')
                out = i
                break
        return out


# May 6th  Majority Element
# 
# Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
# 
# You may assume that the array is non-empty and the majority element always exist in the array.
# 
# Example 1:
# 
# Input: [3,2,3]
# Output: 3
# 
# Example 2:
# 
# Input: [2,2,1,1,1,2,2]
# Output: 2
# 
# 

# In[ ]:


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
    


# May 7th  Cousins in Binary Tree
# 
# In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.
# 
# Two nodes of a binary tree are cousins if they have the same depth, but have different parents.
# 
# We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.
# 
# Return true if and only if the nodes corresponding to the values x and y are cousins.
# 
#  
# 
# Example 1:
# 
# Input: root = [1,2,3,4], x = 4, y = 3
# Output: false
# 
# Example 2:
# 
# Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
# Output: true
# 
# Example 3:
# 
# Input: root = [1,2,3,null,4], x = 2, y = 3
# Output: false
# 
#  
# 
# Constraints:
# 
#     The number of nodes in the tree will be between 2 and 100.
#     Each node has a unique integer value from 1 to 100.
# 
# 

# In[ ]:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        self.dictionary ={}
        value =1
        depth =0
        
        if root != None:
            value = root.val
            depth += 1
            self.dictionary[value] = (depth,0)
            self.checkchild(root,depth)
        
        if (self.dictionary[x][0]==self.dictionary[y][0]):
            if (self.dictionary[x][1]!=self.dictionary[y][1]):
                return True 
            else:
                return False
        else:
            return False

            
    def checkchild(self,root,depth):
        leftdepth = depth
        rightdepth = depth
        if root.left != None:
            leftvalue = root.left.val
            leftdepth += 1
            parent = root.val
            self.dictionary[leftvalue] =  (leftdepth,parent)
            self.checkchild(root.left,leftdepth)
        else:
            pass
        if root.right != None:
            rightvalue = root.right.val
            rightdepth += 1
            parent = root.val
            self.dictionary [rightvalue] = (rightdepth,parent)
            self.checkchild(root.right,rightdepth)
        else:
            pass
        
    


# May 8th
#   Check If It Is a Straight Line
# 
# You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.
# 
#  
# 
#  
# 
# Example 1:
# 
# Input: coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
# Output: true
# 
# Example 2:
# 
# Input: coordinates = [[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]
# Output: false
# 
#  
# 
# Constraints:
# 
#     2 <= coordinates.length <= 1000
#     coordinates[i].length == 2
#     -10^4 <= coordinates[i][0], coordinates[i][1] <= 10^4
#     coordinates contains no duplicate point.
# 
# 

# In[ ]:


class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        
        slope =self.liner(coordinates[0],coordinates[1])
        newslope = slope
        print(slope)
        
        
        for i in range(len(coordinates)-1):
            newslope = self.liner(coordinates[i],coordinates[i+1])
            if newslope != slope:
                return False
        return True
            
        
    def liner(self,point1,point2):
        (x1,y1,x2,y2) = point1[0],point1[1],point2[0],point2[1]
        if x2-x1 != 0: 
            slope = (y2-y1)/(x2-x1)
        else:
            slope = 99
        return slope

                


# May 9th  Valid Perfect Square
# 
# Given a positive integer num, write a function which returns True if num is a perfect square else False.
# 
# Follow up: Do not use any built-in library function such as sqrt.
# 
#  
# 
# Example 1:
# 
# Input: num = 16
# Output: true
# 
# Example 2:
# 
# Input: num = 14
# Output: false
# 
#  
# 
# Constraints:
# 
#     1 <= num <= 2^31 - 1
# 
# 

# In[ ]:


class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        i = 1
        
        while (i*i)  <= num:
            if (i*i) == num:
                return True
            i +=1
        
        return False


# May 10th  Find the Town Judge
# 
# In a town, there are N people labelled from 1 to N.  There is a rumor that one of these people is secretly the town judge.
# 
# If the town judge exists, then:
# 
#     The town judge trusts nobody.
#     Everybody (except for the town judge) trusts the town judge.
#     There is exactly one person that satisfies properties 1 and 2.
# 
# You are given trust, an array of pairs trust[i] = [a, b] representing that the person labelled a trusts the person labelled b.
# 
# If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return -1.
# 
#  
# 
# Example 1:
# 
# Input: N = 2, trust = [[1,2]]
# Output: 2
# 
# Example 2:
# 
# Input: N = 3, trust = [[1,3],[2,3]]
# Output: 3
# 
# Example 3:
# 
# Input: N = 3, trust = [[1,3],[2,3],[3,1]]
# Output: -1
# 
# Example 4:
# 
# Input: N = 3, trust = [[1,2],[2,3]]
# Output: -1
# 
# Example 5:
# 
# Input: N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
# Output: 3
# 
#  
# 
# Constraints:
# 
#     1 <= N <= 1000
#     0 <= trust.length <= 10^4
#     trust[i].length == 2
#     trust[i] are all different
#     trust[i][0] != trust[i][1]
#     1 <= trust[i][0], trust[i][1] <= N
# 
# 

# In[ ]:



class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
    	count = [0] * (N+1)
    	for i,j in trust:
    		count[i] -= 1
    		count[j] += 1
    	for i in range(1,N+1):
    		if count[i] == N-1:
    			return i
    	return -1


#   Flood Fill
# 
# An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).
# 
# Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.
# 
# To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.
# 
# At the end, return the modified image.
# 
# Example 1:
# 
# Input: 
# image = [[1,1,1],[1,1,0],[1,0,1]]
# sr = 1, sc = 1, newColor = 2
# Output: [[2,2,2],[2,2,0],[2,0,1]]
# Explanation: 
# From the center of the image (with position (sr, sc) = (1, 1)), all pixels connected 
# by a path of the same color as the starting pixel are colored with the new color.
# Note the bottom corner is not colored 2, because it is not 4-directionally connected
# to the starting pixel.
# 
# Note:
# The length of image and image[0] will be in the range [1, 50].
# The given starting pixel will satisfy 0 <= sr < image.length and 0 <= sc < image[0].length.
# The value of each color in image[i][j] and newColor will be an integer in [0, 65535].
# 
#    Hide Hint #1  
# Write a recursive function that paints the pixel if it's the correct color, then recurses on neighboring pixels.

# In[ ]:


class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        R, C = len(image), len(image[0])
        color = image[sr][sc]
        if color == newColor: return image
        def dfs(r, c):
            if image[r][c] == color:
                image[r][c] = newColor
                if r >= 1: dfs(r-1, c)
                if r+1 < R: dfs(r+1, c)
                if c >= 1: dfs(r, c-1)
                if c+1 < C: dfs(r, c+1)

        dfs(sr, sc)
        return image


#   Single Element in a Sorted Array
# 
# You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.
# 
# Follow up: Your solution should run in O(log n) time and O(1) space.
# 
#  
# 
# Example 1:
# 
# Input: nums = [1,1,2,3,3,4,4,8,8]
# Output: 2
# 
# Example 2:
# 
# Input: nums = [3,3,7,7,10,11,11]
# Output: 10
# 
#  
# 
# Constraints:
# 
#     1 <= nums.length <= 10^5
#     0 <= nums[i] <= 10^5
# 
# 

# In[ ]:


class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        return sum(set(nums))*2-sum(nums)


#   Remove K Digits
# 
# Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.
# 
# Note:
# 
#     The length of num is less than 10002 and will be ≥ k.
#     The given num does not contain any leading zero.
# 
# Example 1:
# 
# Input: num = "1432219", k = 3
# Output: "1219"
# Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
# 
# Example 2:
# 
# Input: num = "10200", k = 1
# Output: "200"
# Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
# 
# Example 3:
# 
# Input: num = "10", k = 2
# Output: "0"
# Explanation: Remove all the digits from the number and it is left with nothing whic

# In[ ]:


class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if len(num) == k:
            return '0'
        for i in range(k):
            j = 0
            while num[j]<=num[j+1]:
                j += 1
                if j == len(num)-1:
                    break
            num = num[:j]+num[j+1:]

        return str(int(num))


# May 14th  Implement Trie (Prefix Tree)
# 
# Implement a trie with insert, search, and startsWith methods.
# 
# Example:
# 
# Trie trie = new Trie();
# 
# trie.insert("apple");
# trie.search("apple");   // returns true
# trie.search("app");     // returns false
# trie.startsWith("app"); // returns true
# trie.insert("app");   
# trie.search("app");     // returns true
# 
# Note:
# 
#     You may assume that all inputs are consist of lowercase letters a-z.
#     All inputs are guaranteed to be non-empty strings.
# 
# 

# In[ ]:


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.vol = {}
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self.vol[word] = 1
        print('success')
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if word in self.vol.keys():
            return True
        return False
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        for keys in self.vol:
            if prefix in keys:
                for i in range(len(keys)):
                    if prefix == keys[:i+1]:
                        return True
        return False


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


# May 15th
# Maximum Sum Circular Subarray
# 
# Given a circular array C of integers represented by A, find the maximum possible sum of a non-empty subarray of C.
# 
# Here, a circular array means the end of the array connects to the beginning of the array.  (Formally, C[i] = A[i] when 0 <= i < A.length, and C[i+A.length] = C[i] when i >= 0.)
# 
# Also, a subarray may only include each element of the fixed buffer A at most once.  (Formally, for a subarray C[i], C[i+1], ..., C[j], there does not exist i <= k1, k2 <= j with k1 % A.length = k2 % A.length.)
# 
#  
# 
# Example 1:
# 
# Input: [1,-2,3,-2]
# Output: 3
# Explanation: Subarray [3] has maximum sum 3
# 
# Example 2:
# 
# Input: [5,-3,5]
# Output: 10
# Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10
# 
# Example 3:
# 
# Input: [3,-1,2,-1]
# Output: 4
# Explanation: Subarray [2,-1,3] has maximum sum 2 + (-1) + 3 = 4
# 
# Example 4:
# 
# Input: [3,-2,2,-3]
# Output: 3
# Explanation: Subarray [3] and [3,-2,2] both have maximum sum 3
# 
# Example 5:
# 
# Input: [-2,-3,-1]
# Output: -1
# Explanation: Subarray [-1] has maximum sum -1
# 
#  
# 
# Note:
# 
#     -30000 <= A[i] <= 30000
#     1 <= A.length <= 30000
# 
# https://leetcode.com/explore/challenge/card/may-leetcoding-challenge/536/week-3-may-15th-may-21st/3330/

# In[ ]:


class Solution:
    def maxSubarraySumCircular(self, A: List[int]) -> int:
        # if the max subarray includes both the beginning and
        # end, we can find the minimal subarray and subtract
        # it from the sum
        running_max = lambda a, x: max(a + x, x)
        maximal = max(accumulate(A, running_max))
        s = sum(A)
        running_min = lambda a, x: min(a + x, x)
        minimal = min(accumulate(A, running_min))
        minimal = min(minimal, 0)
        # handle edge cases with all negative elements
        return maximal if s == minimal else max(maximal, s - minimal)


# May 16th 
#   Odd Even Linked List
# 
# Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
# 
# You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.
# 
# Example 1:
# 
# Input: 1->2->3->4->5->NULL
# Output: 1->3->5->2->4->NULL
# 
# Example 2:
# 
# Input: 2->1->3->5->6->4->7->NULL
# Output: 2->3->6->7->1->5->4->NULL
# 
#  
# 
# Constraints:
# 
#     The relative order inside both the even and odd groups should remain as it was in the input.
#     The first node is considered odd, the second node even and so on ...
#     The length of the linked list is between [0, 10^4].
# https://leetcode.com/explore/challenge/card/may-leetcoding-challenge/536/week-3-may-15th-may-21st/3331/
# 

# In[ ]:


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        order = 1
        newlist=[]
        tail= head
        if head == None:
            return head
        
        while tail.next != None:
            newlist.append(tail.val)
            tail = tail.next

        
        newlist.append(tail.val)
        fuck=[]
        for i in range(len(newlist)):
            if i%2 == 0:
                fuck.append(newlist[i])
                
                
        for i in range(len(newlist)):
            if i%2 == 1:
                fuck.append(newlist[i])   
                

        
        def assign(head,list,start):
            if head != None:
                head.val = list[start]
                assign(head.next,list,start+1)
                
        assign(head,fuck,0)
            
        return head
                
            


#   Find All Anagrams in a String
# 
# Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
# 
# Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.
# 
# The order of output does not matter.
# 
# Example 1:
# 
# Input:
# s: "cbaebabacd" p: "abc"
# 
# Output:
# [0, 6]
# 
# Explanation:
# The substring with start index = 0 is "cba", which is an anagram of "abc".
# The substring with start index = 6 is "bac", which is an anagram of "abc".
# 
# Example 2:
# 
# Input:
# s: "abab" p: "ab"
# 
# Output:
# [0, 1, 2]
# 
# Explanation:
# The substring with start index = 0 is "ab", which is an anagram of "ab".
# The substring with start index = 1 is "ba", which is an anagram of "ab".
# The substring with start index = 2 is "ab", which is an anagram of "ab".
# https://leetcode.com/explore/challenge/card/may-leetcoding-challenge/536/week-3-may-15th-may-21st/3332/
# 

# In[ ]:


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        slength = len(s)
        pop = []
        plength = len(p)
        for i in range(slength):
            if i+plength <= slength:
                spart = s[i:i+plength]
                if set(spart) == set(p):
                    for elements in set(spart):
                        if spart.count(elements) != p.count(elements):
                            print('found a fake')
                            break
                        pop.append(i)
        pop = list(dict.fromkeys(pop))
        

        return pop
                
            


# In[14]:


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxvalue=  max(nums)
        size = len(nums)
        max_so_far = maxvalue
        max_ending_here = 0
        
        for i in range (0,size):
            max_ending_here=max_ending_here+nums[i]
            if max_ending_here <0:
                max_ending_here=0
            elif (max_so_far < max_ending_here):
                max_so_far = max_ending_here
        return max_so_far
    


# #4>. Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
# 
# Example:
# 
# Input: [0,1,0,3,12]
# Output: [1,3,12,0,0]
# 
# Note:
# 
#     You must do this in-place without making a copy of the array.
#     Minimize the total number of operations.
# 

# In[4]:


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        size = len(nums)
        cheat =[]
        zeroter=0
        x=0
        for i in range(size):           
            #print(nums[x])
            if nums[x] ==0:
                #print(nums[x],"is a zero, removing")
                nums.remove(0)
                zeroter +=1
                #print('current list is', nums)
                x -=1
            x +=1
        for i in range(zeroter):
            nums.append(0)
        


# #5>. Given an array of strings, group anagrams together.
# 
# Example:
# 
# Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
# Output:
# [
#   ["ate","eat","tea"],
#   ["nat","tan"],
#   ["bat"]
# ]
# 
# Note:
# 
#     All inputs will be in lowercase.
#     The order of your output does not matter.
# 
# 

# In[37]:


from collections import defaultdict
class Solution(object):
    def groupAnagrams(self, strs):
        ans = defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()


# In[38]:


x=Solution()
x.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])


# In[40]:


ans = defaultdict(list)


# Single Element in a Sorted Array
# 
# You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.
# 
#  
# 
# Example 1:
# 
# Input: [1,1,2,3,3,4,4,8,8]
# Output: 2
# 
# Example 2:
# 
# Input: [3,3,7,7,10,11,11]
# Output: 10

# In[111]:


#best answer:
    
nums = [1,1,3,3,6,6,7,7,5,8,9,9,8]
sum(set(nums))*2-sum(nums)


# Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.
# 
# Note:
# 
#     The length of num is less than 10002 and will be ≥ k.
#     The given num does not contain any leading zero.
# 
# Example 1:
# 
# Input: num = "1432219", k = 3
# Output: "1219"
# Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
# 
# Example 2:
# 
# Input: num = "10200", k = 1
# Output: "200"
# Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
# 
# Example 3:
# 
# Input: num = "10", k = 2
# Output: "0"
# Explanation: Remove all the digits from the number and it is left with nothing which is 0.
# 

# In[ ]:


num = '237461823'
k=2


# In[ ]:


class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if len(num) == k:
            return '0'

        #print('test value',num)
        #print('times of selection needs to be done',k)
        #print('length of num',len(num))

        for i in range(k):
            #print(i+1,'times selection.',k-i-1,'times left')
            j = 0

            while num[j]<=num[j+1]:
                #print('value',num[j],'is smaller than ',num[j+1])
                #print('pass',num[j],'location is:',j+1)
                j += 1
                if j == len(num)-1:
                    #print('reached the last one, break and delete this one')
                    break
                    
            #print('found the num need to be removed', num[j],'removing')
            #print(num[:j],' ' ,num[j+1:])
            num = num[:j]+num[j+1:]
            #print(num)
        
        return str(int(num))


# Implement a trie with insert, search, and startsWith methods.
# 
# Example:
# 
# Trie trie = new Trie();
# 
# trie.insert("apple");
# trie.search("apple");   // returns true
# trie.search("app");     // returns false
# trie.startsWith("app"); // returns true
# trie.insert("app");   
# trie.search("app");     // returns true
# 
# Note:
# 
#     You may assume that all inputs are consist of lowercase letters a-z.
#     All inputs are guaranteed to be non-empty strings.
# 
# 

# In[ ]:


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.vol = {}
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self.vol[word] = 1
        print('success')
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if word in self.vol.keys():
            return True
        return False
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        for keys in self.vol:
            if prefix in keys:   # why i do this? because if we do not make a pre-judge, the time will used more, so we do not have to check every character
                for i in range(len(keys)):
                    if prefix == keys[:i+1]:
                        return True
        return False


#   Find All Anagrams in a String
# 
# Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
# 
# Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.
# 
# The order of output does not matter.
# 
# Example 1:
# 
# Input:
# s: "cbaebabacd" p: "abc"
# 
# Output:
# [0, 6]
# 
# Explanation:
# The substring with start index = 0 is "cba", which is an anagram of "abc".
# The substring with start index = 6 is "bac", which is an anagram of "abc".
# 
# Example 2:
# 
# Input:
# s: "abab" p: "ab"
# 
# Output:
# [0, 1, 2]
# 
# Explanation:
# The substring with start index = 0 is "ab", which is an anagram of "ab".
# The substring with start index = 1 is "ba", which is an anagram of "ab".
# The substring with start index = 2 is "ab", which is an anagram of "ab".
# 

# In[48]:


def findAnagrams( s: str, p: str):
    slength = len(s)
    pop = []
    plength = len(p)
    for i in range(slength):
        #print(i,'times')
        #print(s[i:i+plength])
        #print(i,i+plength)
        if i+plength <= slength:
            spart = s[i:i+plength]
            if set(spart) == set(p):  
                for elements in set(spart):
                    if spart.count(elements) != p.count(elements):
                        print('found a fake')
                        break
                    pop.append(i)
                        
    pop = list(dict.fromkeys(pop))

    return pop


# In[49]:


findAnagrams('ababababab','aab')


#   Permutation in String
# 
# Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1. In other words, one of the first string's permutations is the substring of the second string.
# 
#  
# 
# Example 1:
# 
# Input: s1 = "ab" s2 = "eidbaooo"
# Output: True
# Explanation: s2 contains one permutation of s1 ("ba").
# 
# Example 2:
# 
# Input:s1= "ab" s2 = "eidboaoo"
# Output: False
# 
#  
# 
# Note:
# 
#     The input strings only contain lower case letters.
#     The length of both given strings is in range [1, 10,000].
# 
#    Hide Hint #1  
# Obviously, brute force will result in TLE. Think of something else.
#    Hide Hint #2  
# How will you check whether one string is a permutation of another string?
#    Hide Hint #3  
# One way is to sort the string and then compare. But, Is there a better way?
#    Hide Hint #4  
# If one string is a permutation of another string then they must one common metric. What is that?
#    Hide Hint #5  
# Both strings must have same character frequencies, if one is permutation of another. Which data structure should be used to store frequencies?
#    Hide Hint #6  
# What about hash table? An array of size 26?

# In[24]:


from collections import Counter
def checkInclusion(s1: str, s2: str):
    m=""
    for i in range(0,len(s2)-len(s1)+1):
        print(i)
      
        m=s2[i:i+len(s1)]
        print(m,Counter(m),Counter(s1))
        if Counter(m)==Counter(s1):
            return True
    return False


# In[25]:


checkInclusion('ab','abc')


# May 19th  Online Stock Span
# 
# Write a class StockSpanner which collects daily price quotes for some stock, and returns the span of that stock's price for the current day.
# 
# The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backwards) for which the price of the stock was less than or equal to today's price.
# s
# For example, if the price of a stock over the next 7 days were [100, 80, 60, 70, 60, 75, 85], then the stock spans would be [1, 1, 1, 2, 1, 4, 6].
# 
#  
# 
# Example 1:
# 
# Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
# Output: [null,1,1,1,2,1,4,6]
# Explanation: 
# First, S = StockSpanner() is initialized.  Then:
# S.next(100) is called and returns 1,
# S.next(80) is called and returns 1,
# S.next(60) is called and returns 1,
# S.next(70) is called and returns 2,
# S.next(60) is called and returns 1,
# S.next(75) is called and returns 4,
# S.next(85) is called and returns 6.
# 
# Note that (for example) S.next(75) returned 4, because the last 4 prices
# (including today's price of 75) were less than or equal to today's price.
# 
#  
# 
# Note:
# 
#     Calls to StockSpanner.next(int price) will have 1 <= price <= 10^5.
#     There will be at most 10000 calls to StockSpanner.next per test case.
#     There will be at most 150000 calls to StockSpanner.next across all test cases.
#     The total time limit for this problem has been reduced by 75% for C++, and 50% for all other languages.
# 
# https://leetcode.com/explore/challenge/card/may-leetcoding-challenge/536/week-3-may-15th-may-21st/3334/

# there are two test which are interesting:
# 
# ubmission Result: Wrong Answer 
# Input: ["StockSpanner","next","next","next","next","next"]
# [[],[31],[41],[48],[59],[79]]
# Output: [null,1,1,2,3,4]
# Expected: [null,1,2,3,4,5]

# Your input
# 
# ["StockSpanner","next","next","next","next","next","next","next"]
# [[],[100],[80],[60],[70],[60],[75],[85]]
# Your answer
# 
# [null,1,2,2,3,2,5,7]
# 
# Expected answer
# 
# [null,1,1,1,2,1,4,6]

# In[87]:


class StockSpanner:

    def __init__(self):
        self.history =[]
        self.min = 100000
        self.index = 1
        
    def next(self, price: int) -> int:
        self.history.append(price)
        days =len(self.history)
        self.index =0
        print('haivng new thing')
        for i in range(1,days+1):
            if self.history[-i] <= price:
                self.index += 1
            else:
                break
                
        return self.index


# above solution is correct however it got TLE while running with a more complicated string, below I attached a solution is so called "monotonic stack"

# In[94]:


class StockSpanner:

    def __init__(self):
        
        # maintain a monotonic stack for stock entry
        
		## definition of stock entry:
        # first parameter is price quote
        # second parameter is price span
        self.monotone_stack = []
              
        
        
    def next(self, price: int) -> int:

        stack = self.monotone_stack
        
        cur_price_quote, cur_price_span = price, 1
        
        # Compute price span in stock data with monotonic stack
        while stack and stack[-1][0] <= cur_price_quote:
            
            prev_price_quote, prev_price_span = stack.pop()
            
            # update current price span with history data in stack
            cur_price_span += prev_price_span
        
        # Update latest price quote and price span
        stack.append( (cur_price_quote, cur_price_span) )
        
        return cur_price_span

    


# In[99]:


tester = StockSpanner()
tester.next(31)
tester.next(41)
tester.next(40)


# In[100]:


tester.monotone_stack


# hMay 21st  Count Square Submatrices with All Ones
# 
# Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.
# 
#  
# 
# Example 1:
# 
# Input: matrix =
# [
#   [0,1,1,1],
#   [1,1,1,1],
#   [0,1,1,1]
# ]
# Output: 15
# Explanation: 
# There are 10 squares of side 1.
# There are 4 squares of side 2.
# There is  1 square of side 3.
# Total number of squares = 10 + 4 + 1 = 15.
# 
# Example 2:
# 
# Input: matrix = 
# [
#   [1,0,1],
#   [1,1,0],
#   [1,1,0]
# ]
# Output: 7
# Explanation: 
# There are 6 squares of side 1.  
# There is 1 square of side 2. 
# Total number of squares = 6 + 1 = 7.
# 
#  
# 
# Constraints:
# 
#     1 <= arr.length <= 300
#     1 <= arr[0].length <= 300
#     0 <= arr[i][j] <= 1
# 
# 

# In[ ]:


class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        if matrix is None or len(matrix) == 0:
             return 0
        
        rows = len(matrix)
        cols = len(matrix[0])
         
        result = 0
         
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 1:   
                    if r == 0 or c == 0: # Cases with first row or first col
                        result += 1      # The 1 cells are square on its own               
                    else:                # Other cells
                        cell_val = min(matrix[r-1][c-1], matrix[r][c-1], matrix[r-1][c]) + matrix[r][c]
                        result += cell_val
                        matrix[r][c] = cell_val #**memoize the updated result**
        return result  


# May 22nd  Sort Characters By Frequency
# 
# Given a string, sort it in decreasing order based on the frequency of characters.
# 
# Example 1:
# 
# Input:
# "tree"
# 
# Output:
# "eert"
# 
# Explanation:
# 'e' appears twice while 'r' and 't' both appear once.
# So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
# 
# Example 2:
# 
# Input:
# "cccaaa"
# 
# Output:
# "cccaaa"
# 
# Explanation:
# Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
# Note that "cacaca" is incorrect, as the same characters must be together.
# 
# Example 3:
# 
# Input:
# "Aabb"
# 
# Output:
# "bbAa"
# 
# Explanation:
# "bbaA" is also a valid answer, but "Aabb" is incorrect.
# Note that 'A' and 'a' are treated as two different characters.
# 

# In[73]:


test ='aacccbb'


# In[74]:


from collections import Counter


# In[128]:



def frequencySort(s: str):
    test = Counter(s)
    new = ''
    for i in range(len(tester.most_common())):
        for j in range((tester.most_common()[i][1])):
            print(j)
            new = new+(tester.most_common()[i][0])
    return new


# In[ ]:


#another sollution is shown below:
#one-line
return "".join([char * times for char, times in collections.Counter(str).most_common()])


# In[133]:


def frequencySort(s: str):
    return "".join([char * times for char, times in Counter(s).most_common()])


# In[134]:


frequencySort('aaabvvvvvv')


# May 23rd  Interval List Intersections
# 
# Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.
# 
# Return the intersection of these two interval lists.
# 
# (Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  The intersection of two closed intervals is a set of real numbers that is either empty, or can be represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)
# 
#  
# 
# Example 1:
# 
# Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
# Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
# Reminder: The inputs and the desired output are lists of Interval objects, and not arrays or lists.
# 
#  
# 
# Note:
# 
#     0 <= A.length < 1000
#     0 <= B.length < 1000
#     0 <= A[i].start, A[i].end, B[i].start, B[i].end < 10^9
# 
# NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
# 

# In[179]:



def intervalIntersection(A, B):
    m, n = len(A), len(B)
    i = j = 0
    res = []
    while i < m and j < n:
        if A[i][-1] >= B[j][0] and A[i][0] <= B[j][-1]:
            res.append([max(A[i][0], B[j][0]), min(A[i][-1], B[j][-1])])
        if A[i][-1] < B[j][-1]:
            i += 1
        else:
            j += 1
    return res 


# In[180]:


A = [[0,2],[5,10],[13,23],[24,25]]
B = [[1,5],[8,12],[15,24],[25,26]]
intervalIntersection(A,B)


# May 24th  Construct Binary Search Tree from Preorder Traversal
# 
# Return the root node of a binary search tree that matches the given preorder traversal.
# 
# (Recall that a binary search tree is a binary tree where for every node, any descendant of node.left has a value < node.val, and any descendant of node.right has a value > node.val.  Also recall that a preorder traversal displays the value of the node first, then traverses node.left, then traverses node.right.)
# 
# It's guaranteed that for the given test cases there is always possible to find a binary search tree with the given requirements.
# 
# Example 1:
# 
# Input: [8,5,1,7,10,12]
# Output: [8,5,10,1,7,null,12]
# 
#  
# 
# Constraints:
# 
#     1 <= preorder.length <= 100
#     1 <= preorder[i] <= 10^8
#     The values of preorder are distinct.
# 
# 

# In[200]:


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def bstFromPreorder( preorder):
    root = TreeNode(preorder[0])
    stack = [root]
    for value in preorder[1:]:
        if value < stack[-1].val:
            stack[-1].left = TreeNode(value)
            stack.append(stack[-1].left)
        else:
            while stack and stack[-1].val < value:
                last = stack.pop()
            last.right = TreeNode(value)
            stack.append(last.right)
    return root


# In[202]:


answer = bstFromPreorder(preorder = [8, 5, 1, 7, 10, 12])


# In[206]:





# May 25th  Uncrossed Lines
# 
# We write the integers of A and B (in the order they are given) on two separate horizontal lines.
# 
# Now, we may draw connecting lines: a straight line connecting two numbers A[i] and B[j] such that:
# 
#     A[i] == B[j];
#     The line we draw does not intersect any other connecting (non-horizontal) line.
# 
# Note that a connecting lines cannot intersect even at the endpoints: each number can only belong to one connecting line.
# 
# Return the maximum number of connecting lines we can draw in this way.
# 
#  
# 
# Example 1:
# 
# Input: A = [1,4,2], B = [1,2,4]
# Output: 2
# Explanation: We can draw 2 uncrossed lines as in the diagram.
# We cannot draw 3 uncrossed lines, because the line from A[1]=4 to B[2]=4 will intersect the line from A[2]=2 to B[1]=2.
# 
# Example 2:
# 
# Input: A = [2,5,1,2,5], B = [10,5,2,1,5,2]
# Output: 3
# 
# Example 3:
# 
# Input: A = [1,3,7,1,7,5], B = [1,9,2,5,1]
# Output: 2
# 
#  
# 
# Note:
# 
#     1 <= A.length <= 500
#     1 <= B.length <= 500
#     1 <= A[i], B[i] <= 2000
# 
# 

# In[ ]:


class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:

        dp, m, n = collections.defaultdict(int), len(A), len(B)
        for i in range(m):
            for j in range(n):
                dp[i, j] = max(dp[i - 1, j - 1] + (A[i] == B[j]), dp[i - 1, j], dp[i, j - 1])
        return dp[m - 1, n - 1]


# May 26th  Contiguous Array
# 
# Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
# 
# Example 1:
# 
# Input: [0,1]
# Output: 2
# Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.
# 
# Example 2:
# 
# Input: [0,1,0]
# Output: 2
# Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.
# 
# Note: The length of the given binary array will not exceed 50,000. 

# In[ ]:


class Solution:
    def findMaxLength(self, nums):
        count = 0
        max_length=0
        table = {0: 0}
        for index, num in enumerate(nums, 1):
            if num == 0:
                count -= 1
            else:
                count += 1
            
            if count in table:
                max_length = max(max_length, index - table[count])
            else:
                table[count] = index
        
        return max_length


#   Possible Bipartition
# 
# Given a set of N people (numbered 1, 2, ..., N), we would like to split everyone into two groups of any size.
# 
# Each person may dislike some other people, and they should not go into the same group. 
# 
# Formally, if dislikes[i] = [a, b], it means it is not allowed to put the people numbered a and b into the same group.
# 
# Return true if and only if it is possible to split everyone into two groups in this way.
# 
#  
# 
# Example 1:
# 
# Input: N = 4, dislikes = [[1,2],[1,3],[2,4]]
# Output: true
# Explanation: group1 [1,4], group2 [2,3]
# 
# Example 2:
# 
# Input: N = 3, dislikes = [[1,2],[1,3],[2,3]]
# Output: false
# 
# Example 3:
# 
# Input: N = 5, dislikes = [[1,2],[2,3],[3,4],[4,5],[1,5]]
# Output: false
# 
#  
# 
# Constraints:
# 
#     1 <= N <= 2000
#     0 <= dislikes.length <= 10000
#     dislikes[i].length == 2
#     1 <= dislikes[i][j] <= N
#     dislikes[i][0] < dislikes[i][1]
#     There does not exist i != j for which dislikes[i] == dislikes[j].
# 
# 

# In[ ]:


class Solution:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        
        # Constant defined for color drawing to person
        NOT_COLORED, BLUE, GREEN = 0, 1, -1
        
        # -------------------------------------
        
        def helper( person_id, color ):
            
            # Draw person_id as color
            color_table[person_id] = color
            
            # Draw the_other, with opposite color, in dislike table of current person_id
            for the_other in dislike_table[ person_id ]:
   
                if color_table[the_other] == color:
                    # the_other has the same color of current person_id
                    # Reject due to breaking the relationship of dislike
                    return False

                if color_table[the_other] == NOT_COLORED and (not helper( the_other, -color)):
                    # Other people can not be colored with two different colors. 
					# Therefore, it is impossible to keep dis-like relationship with bipartition.
                    return False
                    
            return True
        
        
        # ------------------------------------------------
		
		
        if N == 1 or not dislikes:
            # Quick response for simple cases
            return True
        
        
        # each person maintain a list of dislike
        dislike_table = collections.defaultdict( list )
        
        # cell_#0 is dummy just for the convenience of indexing from 1 to N
        color_table = [ NOT_COLORED for _ in range(N+1) ]
        
        for p1, p2 in dislikes:
            
            # P1 and P2 dislike each other
            dislike_table[p1].append( p2 )
            dislike_table[p2].append( p1 )
            
        
        # Try to draw dislike pair with different colors in DFS
        for person_id in range(1, N+1):
            
            if color_table[person_id] == NOT_COLORED and (not helper( person_id, BLUE)):
                # Other people can not be colored with two different colors. 
				# Therefore, it is impossible to keep dis-like relationship with bipartition.
                return False 
        
        return True


# May 28th Counting Bits
# 
# Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.
# 
# Example 1:
# 
# Input: 2
# Output: [0,1,1]
# 
# Example 2:
# 
# Input: 5
# Output: [0,1,1,2,1,2]
# 
# Follow up:
# 
#     It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
#     Space complexity should be O(n).
#     Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.
# 

# In[ ]:


class Solution:
    def countBits(self, num: int) -> List[int]:
        
        iniArr = [0]
        if num > 0:
            amountToAdd = 1
            while len(iniArr) < num + 1:
                iniArr.extend([x+1 for x in iniArr])
        
        return iniArr[0:num+1]


# In[ ]:


#an easy version of above solution
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        res = [0]
        while len(res) <= num:
            res += [i+1 for i in res]
        return res[:num+1]


# In[175]:


longrange


# In[175]:


longrange


#   Course Schedule
# 
# There are a total of numCourses courses you have to take, labeled from 0 to numCourses-1.
# 
# Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
# 
# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
# 
#  
# 
# Example 1:
# 
# Input: numCourses = 2, prerequisites = [[1,0]]
# Output: true
# Explanation: There are a total of 2 courses to take. 
#              To take course 1 you should have finished course 0. So it is possible.
# 
# Example 2:
# 
# Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
# Output: false
# Explanation: There are a total of 2 courses to take. 
#              To take course 1 you should have finished course 0, and to take course 0 you should
#              also have finished course 1. So it is impossible.
# 
#  
# 
# Constraints:
# 
#     The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
#     You may assume that there are no duplicate edges in the input prerequisites.
#     1 <= numCourses <= 10^5
# 
#    Hide Hint #1  
# This problem is equivalent to finding if a cycle exists in a directed graph. If a cycle exists, no topological ordering exists and therefore it will be impossible to take all courses.
#    Hide Hint #2  
# Topological Sort via DFS - A great video tutorial (21 minutes) on Coursera explaining the basic concepts of Topological Sort.
#    Hide Hint #3  
# Topological sort could also be done via BFS.

# In[245]:


def canFinish(n, prerequisites):
    print('set degree and G')
    G = [[] for i in range(n)]
    degree = [0] * n
    print('first step')
    for j, i in prerequisites:
        print(j,i)
        G[i].append(j)
        degree[j] += 1
    print(G)
    print(degree)
    print('second step')
    bfs = [i for i in range(n) if degree[i] == 0]
    #group 0 degree people together

    print(bfs)
    print(G)
    for i in bfs:
        for j in G[i]:
            print(G[i])
            degree[j] -= 1
            if degree[j] == 0:
                bfs.append(j)
    print(bfs)
    return len(bfs) == n


# In[247]:


canFinish(10,[[1,0],[2,3],[9,7],[7,9]])


# In[250]:


canFinish(1,[[1,0],[2,3],[9,7],[2,4]])


# In[258]:


def kClosest(self, points, K):
    return heapq.nsmallest(K, points,lambda x, y: x * x + y * y)
#this only works on python2


# In[260]:


def kClosest(points,K):
    return heapq.nsmallest(K, points, lambda x: x[0] * x[0] + x[1] * x[1])


# In[261]:


#this one is more faster
def kClosest(points,K):
        answer = sorted(points, key = lambda x : x[0]**2+x[1]**2 )
        return answer[:K]


# HARD  Edit Distance
# 
# Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.
# 
# You have the following 3 operations permitted on a word:
# 
#     Insert a character
#     Delete a character
#     Replace a character
# 
# Example 1:
# 
# Input: word1 = "horse", word2 = "ros"
# Output: 3
# Explanation: 
# horse -> rorse (replace 'h' with 'r')
# rorse -> rose (remove 'r')
# rose -> ros (remove 'e')
# 
# Example 2:
# 
# Input: word1 = "intention", word2 = "execution"
# Output: 5
# Explanation: 
# intention -> inention (remove 't')
# inention -> enention (replace 'i' with 'e')
# enention -> exention (replace 'n' with 'x')
# exention -> exection (replace 'n' with 'c')
# exection -> execution (insert 'u')
# 
# 

# In[266]:



def minDistance(word1: str, word2: str):
    """Naive recursive solution"""
    if not word1 and not word2:
        return 0
    if not word1:
        return len(word2)
    if not word2:
        return len(word1)
    if word1[0] == word2[0]:
        return minDistance(word1[1:], word2[1:])
    insert = 1 + minDistance(word1, word2[1:])
    delete = 1 + minDistance(word1[1:], word2)
    replace = 1 + minDistance(word1[1:], word2[1:])
    return min(insert, replace, delete)
#this one work but will get LTE


# In[276]:


#advanced
def minDistance(word1, word2):
    """Dynamic programming solution"""
    m = len(word1)
    n = len(word2)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    print(table)
    for i in range(m + 1):
        table[i][0] = i
    print(table)
    for j in range(n + 1):
        table[0][j] = j
    print(table)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
    print(table)
    return table[-1][-1]


# In[307]:


#third solution
def minDistance(word1, word2,cache ={}):
    if not word1 and not word2:
        return 0
    if not len(word1)   or not len(word2):
        return len(word1) or len(word2)
    if word1[0]==word2[0]:
        return minDistance(word1[1:],word2[1:])
    if (word1,word2) not in cache:
        inserted = 1+ minDistance(word1,word2[1:])
        deleted  = 1+ minDistance(word1[1:],word2)
        replaced = 1+ minDistance(word1[1:],word2[1:])
        cache[(word1,word2)]= min(inserted,deleted,replaced)
        print(cache)
    return cache[(word1,word2)]


# In[308]:


minDistance('oe','house')
minDistance('bo','oa')


# #below is an explaination of the solution
# Python solutions and intuition
# 494
# anderson5's avatar
# anderson5
# 522
# 
# Last Edit: October 26, 2018 12:44 PM
# 
# 13.4K VIEWS
# 
# For those having difficulty cracking dynamic programming solutions, I find it easiest to solve by first starting with a naive, but working recursive implementation. It's essential to do so, because dynamic programming is basically recursion with caching. With this workflow, deciphering dynamic programming problems becomes just a little more manageable for us normal people. :)
# 
# Thought process:
# Given two strings, we're tasked with finding the minimum number of transformations we need to make to arrive with equivalent strings. From the get-go, there doesn't seem to be any way around trying all possibilities, and in this, possibilities refers to inserting, deleting, or replacing a character. Recursion is usually a good choice for trying all possilbilities.
# 
# Whenever we write recursive functions, we'll need some way to terminate, or else we'll end up overflowing the stack via infinite recursion. With strings, the natural state to keep track of is the index. We'll need two indexes, one for word1 and one for word2. Now we just need to handle our base cases, and recursive cases.
# What happens when we're done with either word? Some thought will tell you that the minimum number of transformations is simply to insert the rest of the other word. This is our base case. What about when we're not done with either string? We'll either match the currently indexed characters in both strings, or mismatch. In the first case, we don't incur any penalty, and we can continue to compare the rest of the strings by recursing on the rest of both strings. In the case of a mismatch, we either insert, delete, or replace. To recap:
# 
#     base case: word1 = "" or word2 = "" => return length of other string
#     recursive case: word1[0] == word2[0] => recurse on word1[1:] and word2[1:]
#     recursive case: word1[0] != word2[0] => recurse by inserting, deleting, or replacing
# 
# And in Python:
# 
# class Solution:
#     def minDistance(self, word1, word2):
#         """Naive recursive solution"""
#         if not word1 and not word2:
#             return 0
#         if not word1:
#             return len(word2)
#         if not word2:
#             return len(word1)
#         if word1[0] == word2[0]:
#             return self.minDistance(word1[1:], word2[1:])
#         insert = 1 + self.minDistance(word1, word2[1:])
#         delete = 1 + self.minDistance(word1[1:], word2)
#         replace = 1 + self.minDistance(word1[1:], word2[1:])
#         return min(insert, replace, delete)
# 
# With a solution in hand, we're ecstatic and we go to submit our code. All is well until we see the dreaded red text... TIME LIMIT EXCEEDED. What did we do wrong? Let's look at a simple example, and for sake of brevity I'll annotate the minDistance function as md.
# 
# word1 = "horse"
# word2 = "hello"
# 
# The tree of recursive calls, 3 levels deep, looks like the following. I've highlighted recursive calls with multiple invocations. So now we see that we're repeating work. I'm not going to try and analyze the runtime of this solution, but it's exponential.
# 
# md("horse", "hello")
# 	md("orse", "ello")
# 		md("orse", "llo")
# 			md("orse", "lo")
# 			md("rse", "llo") <- 
# 			md("rse", "lo")
# 		md("rse", "ello")
# 			md("rse", "llo") <-
# 			md("se", "ello")
# 			md("se", "llo") <<-
# 		md("rse", "llo")
# 			md("rse", "llo") <-
# 			md("se", "llo") <<-
# 			md("se", "lo")
# 
# The way we fix this is by caching. We save intermediate computations in a dictionary and if we recur on the same subproblem, instead of doing the same work again, we return the saved value. Here is the memoized solution, where we build from bigger subproblems to smaller subproblems (top-down).
# 
# class Solution:
#     def minDistance(self, word1, word2, i, j, memo):
#         """Memoized solution"""
#         if i == len(word1) and j == len(word2):
#             return 0
#         if i == len(word1):
#             return len(word2) - j
#         if j == len(word2):
#             return len(word1) - i
# 
#         if (i, j) not in memo:
#             if word1[i] == word2[j]:
#                 ans = self.minDistance2(word1, word2, i + 1, j + 1, memo)
#             else: 
#                 insert = 1 + self.minDistance2(word1, word2, i, j + 1, memo)
#                 delete = 1 + self.minDistance2(word1, word2, i + 1, j, memo)
#                 replace = 1 + self.minDistance2(word1, word2, i + 1, j + 1, memo)
#                 ans = min(insert, delete, replace)
#             memo[(i, j)] = ans
#         return memo[(i, j)]
# 
# Of course, an interative implementation is usually better than its recursive counterpart because we don't risk blowing up our stack in case the number of recursive calls is very deep. We can also use a 2D array to do essentially the same thing as the dictionary of cached values. When we do this, we build up solutions from smaller subproblems to bigger subproblems (bottom-up). In this case, since we are no longer "recurring" in the traditional sense, we initialize our 2D table with base constraints. The first row and column of the table has known values since if one string is empty, we simply add the length of the non-empty string since that is the minimum number of edits necessary to arrive at equivalent strings. For both the memoized and dynamic programming solutions, the runtime is O(mn) and the space complexity is O(mn) where m and n are the lengths of word1 and word2, respectively.
# 
# class Solution:
#     def minDistance(self, word1, word2):
#         """Dynamic programming solution"""
#         m = len(word1)
#         n = len(word2)
#         table = [[0] * (n + 1) for _ in range(m + 1)]
# 
#         for i in range(m + 1):
#             table[i][0] = i
#         for j in range(n + 1):
#             table[0][j] = j
# 
#         for i in range(1, m + 1):
#             for j in range(1, n + 1):
#                 if word1[i - 1] == word2[j - 1]:
#                     table[i][j] = table[i - 1][j - 1]
#                 else:
#                     table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
#         return table[-1][-1]
# 

# In[310]:


def minDistance(word1,word2):
    if not word1 and not word2:
        return 0
    if not len(word1)   or not len(word2):
        return len(word1) or len(word2)
    if word1[0] == word2[0]:
        return minDistance(word1[1:],word2[1:])
    else:
        delete = 1+minDistance(word1[1:],word2)
        replace = 1+minDistance(word1[1:],word2[1:])
        incert = 1+minDistance(word1,word2[1:])
    return min(delete,replace,incert)


# In[311]:


minDistance('horse','oe')


# JUNE 1st  Invert Binary Tree
# 
# Invert a binary tree.
# 
# Example:
# 
# Input:
# 
#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# 
# Output:
# 
#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
# 
# Trivia:
# This problem was inspired by this original tweet by Max Howell:
# 
#     Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.
# 
# 

# In[ ]:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# In[ ]:


#solution 2
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
            return root
        
#this solution is more clever that you combined two steps together:
#1>. switch two branches
#2>. recurrssive. 


# JUNE 2rd Delete Node in a Linked List
# 
# Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
# 
# Given linked list -- head = [4,5,1,9], which looks like following:
# 
#  
# 
# Example 1:
# 
# Input: head = [4,5,1,9], node = 5
# Output: [4,1,9]
# Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.
# 
# Example 2:
# 
# Input: head = [4,5,1,9], node = 1
# Output: [4,5,9]
# Explanation: You are given the third node with value 1, the linked list should become 4 -> 5 -> 9 after calling your function.
# 
#  
# 
# Note:
# 
#     The linked list will have at least two elements.
#     All of the nodes' values will be unique.
#     The given node will not be the tail and it will always be a valid node of the linked list.
#     Do not return anything from your function.
# 
# 

# In[ ]:


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


#this is a pretty weird question...just remember it


# JUNE 3rd
#   Two City Scheduling
# 
# There are 2N people a company is planning to interview. The cost of flying the i-th person to city A is costs[i][0], and the cost of flying the i-th person to city B is costs[i][1].
# 
# Return the minimum cost to fly every person to a city such that exactly N people arrive in each city.
# 
#  
# 
# Example 1:
# 
# Input: [[10,20],[30,200],[400,50],[30,20]]
# Output: 110
# Explanation: 
# The first person goes to city A for a cost of 10.
# The second person goes to city A for a cost of 30.
# The third person goes to city B for a cost of 50.
# The fourth person goes to city B for a cost of 20.
# 
# The total minimum cost is 10 + 30 + 50 + 20 = 110 to have half the people interviewing in each city.
# 
#  
# 
# Note:
# 
#     1 <= costs.length <= 100
#     It is guaranteed that costs.length is even.
#     1 <= costs[i][0], costs[i][1] <= 1000
# 

# In[ ]:



def twoCitySchedCost(self, costs: List[List[int]]) -> int:
    costA=[]
    costB=[]
    for i in range(len(costs)) :
        costA.append(costs[i][0])
        costB.append(costs[i][1])


    return (costA).sort


# In[338]:


costA=[1,4,3,3,4,9]


# JUNE 4th  Reverse String
# 
# Write a function that reverses a string. The input string is given as an array of characters char[].
# 
# Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
# 
# You may assume all the characters consist of printable ascii characters.
# 
#  
# 
# Example 1:
# 
# Input: ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]
# 
# Example 2:
# 
# Input: ["H","a","n","n","a","h"]
# Output: ["h","a","n","n","a","H"]
# 
# 

# In[ ]:


#there are three solutions
#no.1
s.reverse()
#no.2
s[::-1]
#no.3
class Solution:
    def reverseString(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left, right = left + 1, right - 1


# JUNE 5th  Random Pick with Weight
# 
# Given an array w of positive integers, where w[i] describes the weight of index i, write a function pickIndex which randomly picks an index in proportion to its weight.
# 
# Note:
# 
#     1 <= w.length <= 10000
#     1 <= w[i] <= 10^5
#     pickIndex will be called at most 10000 times.
# 
# Example 1:
# 
# Input: 
# ["Solution","pickIndex"]
# [[[1]],[]]
# Output: [null,0]
# 
# Example 2:
# 
# Input: 
# ["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
# [[[1,3]],[],[],[],[],[]]
# Output: [null,0,1,1,1,0]
# 
# Explanation of Input Syntax:
# 
# The input is two lists: the subroutines called and their arguments. Solution's constructor has one argument, the array w. pickIndex has no arguments. Arguments are always wrapped with a list, even if there aren't any.
# 

# JUNE 6th Queue Reconstruction by Height
# 
# Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue.
# 
# Note:
# The number of people is less than 1,100.
#  
# 
# Example
# 
# Input:
# [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
# 
# Output:
# [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
# 

# In[ ]:


class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people = sorted(people, key = lambda x: (-x[0], x[1]))
        res = []
        for p in people:
            res.insert(p[1], p)
        return res


# JUNE 7th   Coin Change 2
# 
# You are given coins of different denominations and a total amount of money. Write a function to compute the number of combinations that make up that amount. You may assume that you have infinite number of each kind of coin.
# 
#  
# 
# Example 1:
# 
# Input: amount = 5, coins = [1, 2, 5]
# Output: 4
# Explanation: there are four ways to make up the amount:
# 5=5
# 5=2+2+1
# 5=2+1+1+1
# 5=1+1+1+1+1
# 
# Example 2:
# 
# Input: amount = 3, coins = [2]
# Output: 0
# Explanation: the amount of 3 cannot be made up just with coins of 2.
# 
# Example 3:
# 
# Input: amount = 10, coins = [10] 
# Output: 1
# 
#  
# 
# Note:
# 
# You can assume that
# 
#     0 <= amount <= 5000
#     1 <= coin <= 5000
#     the number of coins is less than 500
#     the answer is guaranteed to fit into signed 32-bit integer
# 
# 

# In[ ]:


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:

        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in coins:
            for j in range(1, amount + 1):
               if j >= i:
                   dp[j] += dp[j - i]
        return dp[amount]


# June 8th Power of Two
# 
# Given an integer, write a function to determine if it is a power of two.
# 
# Example 1:
# 
# Input: 1
# Output: true 
# Explanation: 20 = 1
# 
# Example 2:
# 
# Input: 16
# Output: true
# Explanation: 24 = 16
# 
# Example 3:
# 
# Input: 218
# Output: false
# 
# 

# In[ ]:


class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        i = 0
        while pow(2,i) <= n:
            if pow(2,i) == n:
                print('found')
                return True
            i += 1
        return False

