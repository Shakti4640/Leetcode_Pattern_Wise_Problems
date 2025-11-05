# 88 Merge Sorted Array
# Merge from the end to avoid overwriting elements

from typing import List

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        # Start from the end
        i = m - 1  # Pointer for nums1's valid elements
        j = n - 1  # Pointer for nums2
        k = m + n - 1  # Pointer for placement in nums1

        # Merge from the back
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

        # If anything left in nums2 (nums1's elements are already in place)
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
