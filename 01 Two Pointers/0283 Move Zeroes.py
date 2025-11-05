# 283 Move Zeroes
# Maintain insert position for non-zero elements


# class Solution:
#     def moveZeroes(self, nums: list[int]) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         non_zero_index = 0

#         # Move all non-zero elements to the front
#         for i in range(len(nums)):
#             if nums[i] != 0:
#                 nums[non_zero_index] = nums[i]
#                 non_zero_index += 1

#         # Fill the rest with zeros
#         for i in range(non_zero_index, len(nums)):
#             nums[i] = 0

def moveZeroes(nums: list[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    non_zero_index = 0

    # Move all non-zero elements to the front
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[non_zero_index] = nums[i]
            non_zero_index += 1

    # Fill the rest with zeros
    for i in range(non_zero_index, len(nums)):
        nums[i] = 0

    return nums
# Example 1:

# Input: 
nums = [0,1,0,3,12]

# Output: [1,3,12,0,0]
# Example 2:

# Input: 
# nums = [0]
# Output: [0]

print(moveZeroes(nums))