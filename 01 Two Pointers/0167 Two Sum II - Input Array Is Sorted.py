# 167 Two Sum II - Input Array Is Sorted
# Use two pointers moving inward to find pair

class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            curr_sum = numbers[left] + numbers[right]
            if curr_sum == target:
                return [left + 1, right + 1]  # 1-based indexing
            elif curr_sum < target:
                left += 1
            else:
                right -= 1
