# 26 Remove Duplicates from Sorted Array
# Use two pointers to overwrite duplicates in-place

class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0
        
        i = 0  # Pointer for the place to insert next unique element
        
        for j in range(1, len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        
        return i + 1  # Number of unique elements
