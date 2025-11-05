# 15 3Sum
# Sort array, use two pointers for target sum - nums[i]

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # Sort the array
        result = []
        n = len(nums)
        
        for i in range(n - 2):  # Need at least 3 numbers, so stop at n-3
            if i > 0 and nums[i] == nums[i - 1]:  # Skip duplicates for i
                continue
            target = -nums[i]  # Complement needed for sum = 0
            left, right = i + 1, n - 1
            
            while left < right:
                curr_sum = nums[left] + nums[right]
                if curr_sum == target:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    # Skip duplicates for left and right
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif curr_sum < target:
                    left += 1
                else:
                    right -= 1
        
        return result

            