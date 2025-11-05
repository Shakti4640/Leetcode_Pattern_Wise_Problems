# 27 Remove Element
# Two pointers: one to iterate, one to overwrite non-targets

class Solution:
  def removeElement(self, nums: list[int], val: int) -> int:
      insert_pos = 0

      for num in nums:
          if num != val:
              nums[insert_pos] = num
              insert_pos += 1

      return insert_pos
