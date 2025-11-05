# 11 Container With Most Water
# Two pointers moving inward, maximize area

class Solution:
    def maxArea(self, height: List[int]) -> int:
        arr=height
        left, right = 0, len(arr) - 1
        max_area = 0

        while left < right:
            height = min(arr[left], arr[right])
            width = right - left
            max_area = max(max_area, height * width)

            # Move the pointer from the shorter wall
            if arr[left] < arr[right]:
                left += 1
            else:
                right -= 1

        return max_area