# 239 Sliding Window Maximum
# Use deque to keep track of max in window

def max_sliding_window(nums, k):
    from collections import deque
    
    if not nums or k <= 0:
        return []
    
    result = []
    deq = deque()
    
    for i in range(len(nums)):
        # Remove elements not in the current window
        if deq and deq[0] < i - k + 1:
            deq.popleft()
        
        # Remove elements smaller than the current element
        while deq and nums[deq[-1]] < nums[i]:
            deq.pop()
        
        deq.append(i)
        
        # Start adding to result after the first k elements
        if i >= k - 1:
            result.append(nums[deq[0]])
    
    return result